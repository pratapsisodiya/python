"""The Indian market's own rules, which a strategy written for US equity will violate.

Three of them, and they compound.

A weekly short on NSE cannot be cash equity — delivery cannot be held short overnight — so
it has to be a single-stock future. Futures trade in exchange-defined lots. And the lots
are large: one lot of the cheapest Nifty 200 F&O name is around three lakh of notional. Put
those together with a per-name weight cap and there is a hard floor on account size below
which the short sleeve cannot exist at all, whatever the model thinks of the names.

That floor is the thing worth testing. Not because the arithmetic is hard, but because the
alternative to computing it is discovering it from a rejected order on a Monday morning —
and because the earlier version of this code cheerfully emitted "sell 173" of a future,
which is not a quantity any exchange accepts.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from swingbot.config import load_config
from swingbot.data.instruments_nse import (
    NSEInstruments,
    build_snapshot,
    capital_adequacy,
)
from swingbot.execution import build_orders
from swingbot.types import TargetPortfolio, TargetPosition

SESSIONS = (dt.date(2026, 8, 28), dt.date(2026, 8, 31))

#: Shaped exactly like the exchange dump, so `build_snapshot` is exercised on the real
#: layout rather than on a convenient one.
EXCHANGE_ROWS = [
    {
        "segment": "NSE_EQ", "instrument_type": "EQ", "trading_symbol": "RELIANCE",
        "name": "RELIANCE INDUSTRIES LTD", "isin": "INE002A01018",
        "instrument_key": "NSE_EQ|INE002A01018", "lot_size": 1,
        "freeze_quantity": 100000.0,
    },
    {
        "segment": "NSE_EQ", "instrument_type": "EQ", "trading_symbol": "SMALLCO",
        "name": "SMALL COMPANY LTD", "isin": "INE999A01011",
        "instrument_key": "NSE_EQ|INE999A01011", "lot_size": 1,
        "freeze_quantity": 50000.0,
    },
    # Two expiries for the same underlying, with a revised lot on the far month. The
    # front month is the one that governs an order placed this week.
    {
        "segment": "NSE_FO", "instrument_type": "FUT", "asset_symbol": "RELIANCE",
        "trading_symbol": "RELIANCE FUT 24 SEP 26", "lot_size": 500, "expiry": 1_000,
    },
    {
        "segment": "NSE_FO", "instrument_type": "FUT", "asset_symbol": "RELIANCE",
        "trading_symbol": "RELIANCE FUT 29 OCT 26", "lot_size": 250, "expiry": 2_000,
    },
]


@pytest.fixture()
def instruments(tmp_path):
    frame = build_snapshot(EXCHANGE_ROWS)
    path = tmp_path / "nse_instruments.csv"
    frame.to_csv(path, index=False)
    return NSEInstruments.load(path)


# --------------------------------------------------------------------------------------
# Reading the exchange's own instrument list
# --------------------------------------------------------------------------------------


def test_the_lot_comes_from_the_front_month(instruments):
    """A later expiry can carry a revised lot; this week's order obeys the near one."""
    assert instruments.lot_sizes() == {"RELIANCE": 500}


def test_a_name_with_no_futures_contract_has_no_lot(instruments):
    """And is therefore long-only on NSE, whatever the model thinks of it.

    Reported as absent rather than as 1. For cash equity 1 is correct; for a name with no
    futures contract there is no weekly short available at any size, and conflating the
    two would hide that.
    """
    assert "SMALLCO" not in instruments.lot_sizes()
    assert instruments.get("SMALLCO").is_shortable_via_futures is False
    assert instruments.get("RELIANCE").is_shortable_via_futures is True


def test_instrument_keys_round_trip(instruments):
    """The price provider addresses NSE by ISIN-derived key, not by ticker."""
    assert instruments.instrument_key("RELIANCE") == "NSE_EQ|INE002A01018"
    assert instruments.instrument_key("NOT_LISTED") is None
    assert instruments.instrument_keys(["RELIANCE", "NOT_LISTED"]) == {
        "RELIANCE": "NSE_EQ|INE002A01018"
    }


def test_the_snapshot_is_dated(instruments):
    """Lot sizes are revised, so a run has to be able to say which lots it used."""
    assert instruments.snapshot_date


def test_a_missing_snapshot_degrades_rather_than_crashes(tmp_path):
    """Someone who never refreshed it still gets a signal — with a loud flag, not a stack."""
    assert NSEInstruments.load_if_present(tmp_path / "absent.csv") is None
    with pytest.raises(FileNotFoundError, match="swingbot fetch instruments"):
        NSEInstruments.load(tmp_path / "absent.csv")


# --------------------------------------------------------------------------------------
# The capital floor
# --------------------------------------------------------------------------------------


def test_one_lot_can_exceed_the_entire_position_budget(instruments):
    """The finding that changes what a small account should do.

    RELIANCE in a 500-lot at 1,295 is 647,500 of exposure. A 12 percent cap on a five-lakh
    account allows 60,000. The position is not expensive — it is unreachable, and no
    sizing rule can make it otherwise.
    """
    report = capital_adequacy(
        instruments, {"RELIANCE": 1295.0}, equity=500_000.0, max_weight=0.12
    )
    assert report.tradeable == []
    assert len(report.blocked) == 1

    blocked = report.blocked[0]
    assert blocked.one_lot_notional == pytest.approx(647_500.0)
    # 647,500 / 0.12
    assert blocked.minimum_equity == pytest.approx(5_395_833.0, rel=1e-3)
    assert "NOT ONE" in report.verdict()
    assert "long-only" in report.verdict(), "the verdict has to say what to do instead"


def test_a_large_enough_account_clears_the_floor(instruments):
    report = capital_adequacy(
        instruments, {"RELIANCE": 1295.0}, equity=10_000_000.0, max_weight=0.12
    )
    assert len(report.tradeable) == 1
    assert report.blocked == []
    assert "NOT ONE" not in report.verdict()


def test_a_thin_short_sleeve_is_called_out_as_a_different_strategy(instruments):
    """Being able to short only the cheapest names is not the backtested strategy.

    The book would concentrate in whatever has a small lot rather than in what the model
    dislikes most, and saying "3 of 113 fit" is more useful than silently producing it.
    """
    report = capital_adequacy(
        instruments, {"RELIANCE": 1295.0}, equity=5_500_000.0, max_weight=0.12
    )
    assert len(report.tradeable) == 1
    assert "only 1 of 1" in report.verdict()
    assert "different strategy" in report.verdict()


def test_names_with_no_price_are_skipped_not_assumed(instruments):
    """A missing price is not a zero-notional lot."""
    report = capital_adequacy(instruments, {}, equity=10_000_000.0, max_weight=0.12)
    assert report.constraints == []
    assert "nothing can be checked" in report.verdict()


# --------------------------------------------------------------------------------------
# Lot rounding at the ticket
# --------------------------------------------------------------------------------------


def _book(ticker: str, weight: float, instrument: str) -> TargetPortfolio:
    return TargetPortfolio(
        decision_session=SESSIONS[0],
        entry_session=SESSIONS[1],
        positions=[TargetPosition(ticker=ticker, weight=weight, instrument=instrument)],
    )


def test_a_futures_order_is_rounded_down_to_whole_lots():
    """Down, never up: rounding up would breach the cap the sizing just respected."""
    orders = build_orders(
        _book("RELIANCE", -0.30, "futures"),
        {"RELIANCE": 1295.0}, equity=10_000_000.0,
        lot_sizes={"RELIANCE": 500},
    )
    assert len(orders) == 1
    # 0.30 * 10m / 1295 = 2,316 shares -> 4 lots of 500 = 2,000.
    assert orders[0].quantity == 2000
    assert orders[0].tag == ""


def test_a_cash_equity_order_is_not_subject_to_the_futures_lot():
    """The bug that destroyed the long book, and the reason it is a separate test.

    BRITANNIA's *futures* lot is 125. Applying it to a delivery buy turned a perfectly
    legal 70-share purchase into "below one lot, not tradeable". The cash segment trades
    in single shares; only the derivative has a lot.
    """
    notes: list[str] = []
    orders = build_orders(
        _book("BRITANNIA", 0.065, "equity"),
        {"BRITANNIA": 462.06}, equity=500_000.0,
        lot_sizes={"BRITANNIA": 125},
        notes=notes,
    )
    assert len(orders) == 1
    assert orders[0].quantity == 70, "a cash buy must not be rounded to the F&O lot"
    assert notes == []


def test_an_unknown_derivative_lot_is_flagged_rather_than_guessed():
    """An unrounded futures quantity is not placeable, and the ticket has to say so."""
    orders = build_orders(
        _book("BIOCON", -0.06, "futures"),
        {"BIOCON": 166.38}, equity=500_000.0,
    )
    assert len(orders) == 1
    assert "lot-unknown" in orders[0].tag


def test_a_position_below_one_lot_is_reported_not_silently_dropped():
    """Rounding toward zero makes the position vanish; the note is the only trace."""
    notes: list[str] = []
    build_orders(
        _book("BIOCON", -0.0576, "futures"),
        {"BIOCON": 166.37}, equity=500_000.0,
        derivative_lot_size=500, notes=notes,
    )
    assert notes, "a position the account cannot express must be reported"
    assert "below one lot" in notes[0]
    assert "not tradeable at this size" in notes[0]


# --------------------------------------------------------------------------------------
# The India profile itself
# --------------------------------------------------------------------------------------


def test_the_india_profile_shorts_through_futures_and_the_us_does_not():
    """The whole reason lots matter on one market and not the other."""
    assert load_config("india").market_profile.short_instrument.value == "futures"
    assert load_config("us").market_profile.short_instrument.value == "cash_equity"


def test_the_shipped_universe_carries_real_lot_sizes():
    """The lot table is committed, so a fresh clone knows what it can short.

    Also a canary on the values: they came from the exchange and they vary by an order of
    magnitude, which is precisely why a single market-wide number was never going to work.
    """
    frame = pd.read_csv("config/universe/nifty200.csv")
    assert "lot_size" in frame.columns

    lots = frame["lot_size"].dropna()
    assert len(lots) > 100, "most of the universe should have a declared lot"
    assert lots.min() >= 1
    assert lots.max() / max(lots.min(), 1) > 10, (
        "lot sizes that vary by less than 10x would suggest a fabricated column"
    )


def test_the_instrument_snapshot_ships_with_the_repository():
    """So `swingbot signal --market india` knows the lots without a network call."""
    instruments = NSEInstruments.load()
    assert len(instruments) > 100
    lots = instruments.lot_sizes()
    assert len(lots) > 100
    assert instruments.instrument_key("RELIANCE", ) == "NSE_EQ|INE002A01018"

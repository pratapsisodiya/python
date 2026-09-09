"""Data layer, calendar and execution plumbing.

Less glamorous than the leak tests, and the source of most of the bugs that quietly
corrupt a live signal: a mis-stamped bar, a calendar that drifts on a holiday, or a
position ledger that loses track of what is held.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from swingbot.calendars import TradingCalendar
from swingbot.config import load_config
from swingbot.data import adjust_asof, screen_bars
from swingbot.data.universe import UniverseFile
from swingbot.types import RunMeta, TargetPortfolio, TargetPosition

# ------------------------------------------------------------------------ calendar


def test_decision_entry_exit_are_distinct(calendar, cfg_us):
    """The structural property the whole backtest rests on."""
    grid = calendar.weekly_grid(cfg_us.calendar.hold_sessions)
    assert grid
    for decision in grid:
        assert decision.decision_session < decision.entry_session < decision.exit_session


def test_holiday_friday_falls_back_to_thursday(cfg_us):
    """A missing Friday must not skip the week entirely."""
    sessions = [
        d for d in (dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(120))
        if d.weekday() < 5
    ]
    holiday = next(d for d in sessions if d.weekday() == 4 and d > dt.date(2024, 2, 1))
    sessions.remove(holiday)

    calendar = TradingCalendar.from_config(sessions, cfg_us)
    decisions = calendar.decision_sessions()
    iso = holiday.isocalendar()
    same_week = [d for d in decisions if d.isocalendar()[:2] == iso[:2]]
    assert len(same_week) == 1, "the week with no Friday was dropped"
    assert same_week[0].weekday() == 3, "it should have fallen back to Thursday"


def test_market_close_maps_to_the_right_utc_hour():
    """India closes at 15:30 IST, which is 10:00 UTC. The US closes at 16:00 ET."""
    sessions = [dt.date(2024, 6, 3)]
    india = TradingCalendar.from_config(sessions, load_config("india"))
    us = TradingCalendar.from_config(sessions, load_config("us"))

    assert india.session_close_ts(sessions[0]).hour == 10
    assert us.session_close_ts(sessions[0]).hour == 20  # 16:00 EDT


def test_incomplete_forward_windows_are_dropped(calendar, cfg_us):
    """The final weeks have no exit bar yet and must not produce a truncated label."""
    grid = calendar.weekly_grid(cfg_us.calendar.hold_sessions)
    assert grid[-1].exit_session <= calendar.sessions[-1]


# ---------------------------------------------------------------------- adjustment


def test_adjustment_is_point_in_time():
    """A later split must not retroactively restate an earlier price.

    A vendor's adjusted close is recomputed after every corporate action, so the 2021
    price you read today is not the price anyone saw in 2021. Level features and price
    filters are sensitive to that difference.
    """
    sessions = pd.date_range("2021-01-04", periods=800, freq="B").date
    frame = pd.DataFrame({
        "ticker": "T",
        "session": sessions,
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
        "volume": 1e6, "split_factor": 1.0, "div_cash": 0.0, "is_delisted": False,
        "available_at": pd.to_datetime(sessions, utc=True) + pd.Timedelta(hours=21),
    })
    split_session = sessions[600]
    frame.loc[frame["session"] == split_session, "split_factor"] = 2.0

    early = sessions[300]
    asof_before = adjust_asof(frame, sessions[400])
    fully = adjust_asof(frame, None)

    def close_at(f, session):
        return float(f.loc[f["session"] == session, "close"].iloc[0])

    assert close_at(asof_before, early) == pytest.approx(100.0)
    assert close_at(fully, early) == pytest.approx(50.0)


def test_screen_removes_impossible_bars():
    """Non-positive prices and inconsistent OHLC break everything downstream."""
    frame = pd.DataFrame({
        "ticker": ["T"] * 4,
        "session": pd.date_range("2024-01-01", periods=4, freq="B").date,
        "open": [100.0, 100.0, 100.0, 100.0],
        "high": [101.0, 99.0, 101.0, 101.0],     # row 1 has high < low
        "low": [99.0, 100.0, 99.0, 99.0],
        "close": [100.0, 100.0, -5.0, 100.0],    # row 2 has a negative close
        "volume": [1e6] * 4,
        "split_factor": [1.0] * 4, "div_cash": [0.0] * 4, "is_delisted": [False] * 4,
        "available_at": pd.to_datetime(
            pd.date_range("2024-01-01", periods=4, freq="B"), utc=True
        ),
    })
    assert len(screen_bars(frame)) == 2


def test_screen_removes_unsplit_jumps():
    """A 50% overnight move with no split recorded is a data error, not a return."""
    sessions = pd.date_range("2024-01-01", periods=6, freq="B").date
    closes = [100.0, 101.0, 250.0, 251.0, 252.0, 253.0]
    frame = pd.DataFrame({
        "ticker": "T", "session": sessions,
        "open": closes, "high": [c * 1.01 for c in closes],
        "low": [c * 0.99 for c in closes], "close": closes,
        "volume": 1e6, "split_factor": 1.0, "div_cash": 0.0, "is_delisted": False,
        "available_at": pd.to_datetime(sessions, utc=True),
    })
    assert len(screen_bars(frame)) < len(frame)


# ------------------------------------------------------------------------ universe


def test_universe_has_no_current_tickers_accessor():
    """Asking for "the tickers" without a date is how survivorship bias gets in.

    The API deliberately does not offer it.
    """
    universe = UniverseFile("config/universe/sp500.csv")
    assert not hasattr(universe, "current_tickers")
    assert hasattr(universe, "members_asof")


def test_membership_respects_the_start_date():
    """A name must not be tradeable before it listed."""
    universe = UniverseFile("config/universe/sp500.csv")
    early = set(universe.members_asof(dt.date(2005, 1, 1))["ticker"])
    late = set(universe.members_asof(dt.date(2024, 1, 1))["ticker"])
    assert "META" not in early, "Meta listed in 2012 and cannot be a 2005 member"
    assert "META" in late
    assert early < late


def test_snapshot_universe_is_flagged_as_biased():
    """A file with no real delisting history must say so rather than inflate results."""
    universe = UniverseFile("config/universe/nifty200.csv")
    assert universe.is_survivorship_biased()
    warning = universe.bias_warning()
    assert warning and "survivorship" in warning.lower()


def test_one_exit_does_not_disarm_the_survivorship_warning(tmp_path):
    """The defect: the check used to be "any end_date at all", and one row silenced it.

    Giving LTIM a genuine end date — it merged away — flipped nifty200.csv from
    survivorship-biased to clean while its other 128 names were still precisely the
    survivors of the index as it stands today. A warning that a single edited line can
    switch off is not a warning, and this is the shape of dataset it was written for.
    """
    rows = ["ticker,name,sector,start_date,end_date"]
    rows += [f"T{i:03d},Name {i},Tech,2015-01-01," for i in range(40)]
    rows[1] = "T000,Name 0,Tech,2015-01-01,2020-06-30"  # exactly one exit
    path = tmp_path / "mostly_survivors.csv"
    path.write_text("\n".join(rows) + "\n")

    universe = UniverseFile(path)
    assert universe.n_exits() == 1
    assert universe.is_survivorship_biased(), "one exit in 40 names is not membership history"
    assert "only 1 exit(s) across 40 names" in (universe.bias_warning() or "")


def test_a_file_with_real_membership_history_is_not_flagged(tmp_path):
    """The other side, so the check can still come back clean and mean it."""
    rows = ["ticker,name,sector,start_date,end_date"]
    for i in range(40):
        end = "2020-06-30" if i < 6 else ""
        rows.append(f"T{i:03d},Name {i},Tech,2015-01-01,{end}")
    path = tmp_path / "point_in_time.csv"
    path.write_text("\n".join(rows) + "\n")

    universe = UniverseFile(path)
    assert universe.n_exits() == 6
    assert not universe.is_survivorship_biased()
    assert universe.bias_warning() is None


# ----------------------------------------------------------------------- execution


def test_position_ledger_makes_the_next_diff_correct(tmp_path):
    """The property that turns a signal generator into something that tracks a book.

    Without it, every week's orders would be computed against an empty portfolio and the
    strategy would appear to liquidate and rebuild itself every Friday.
    """
    from swingbot.execution import CSVExecutionAdapter, build_orders

    adapter = CSVExecutionAdapter(tmp_path, equity=1_000_000)
    prices = {"AAA": 100.0, "BBB": 200.0, "CCC": 50.0}
    adapter.set_prices(prices)

    week_one = TargetPortfolio(
        decision_session=dt.date(2024, 3, 15), entry_session=dt.date(2024, 3, 18),
        positions=[
            TargetPosition("AAA", 0.10, 0.8, "equity", "Tech"),
            TargetPosition("BBB", -0.08, -0.7, "equity", "Bank"),
        ],
    )
    meta = RunMeta(
        run_id="r1", market="us", decision_session=week_one.decision_session,
        entry_session=week_one.entry_session, equity=1_000_000, currency="USD",
    )
    orders = build_orders(week_one, prices, 1_000_000, adapter.current_positions())
    adapter.submit(orders, meta)

    held = adapter.current_positions()
    assert held["AAA"] == 1000.0
    assert held["BBB"] == -400.0

    # Week two keeps AAA identical, drops BBB, opens CCC.
    week_two = TargetPortfolio(
        decision_session=dt.date(2024, 3, 22), entry_session=dt.date(2024, 3, 25),
        positions=[
            TargetPosition("AAA", 0.10, 0.8, "equity", "Tech"),
            TargetPosition("CCC", 0.05, 0.6, "equity", "Energy"),
        ],
    )
    next_orders = build_orders(week_two, prices, 1_000_000, adapter.current_positions())
    by_ticker = {o.ticker: o for o in next_orders}

    assert "AAA" not in by_ticker, "an unchanged position generated a trade"
    assert by_ticker["BBB"].side.value == "buy" and by_ticker["BBB"].quantity == 400.0
    assert by_ticker["CCC"].side.value == "buy"


def test_order_files_are_written(tmp_path):
    from swingbot.execution import CSVExecutionAdapter, build_orders

    adapter = CSVExecutionAdapter(tmp_path, equity=500_000)
    prices = {"AAA": 100.0}
    adapter.set_prices(prices)
    portfolio = TargetPortfolio(
        decision_session=dt.date(2024, 3, 15), entry_session=dt.date(2024, 3, 18),
        positions=[TargetPosition("AAA", 0.10, 0.8, "equity", "Tech")],
    )
    meta = RunMeta(
        run_id="r", market="us", decision_session=portfolio.decision_session,
        entry_session=portfolio.entry_session, equity=500_000, currency="USD",
    )
    report = adapter.submit(build_orders(portfolio, prices, 500_000), meta)

    for name in ("orders.csv", "targets.json", "positions.json"):
        assert (tmp_path / name).exists(), name
    assert report.submitted == 1
    assert pd.read_csv(tmp_path / "orders.csv").iloc[0]["ticker"] == "AAA"


def test_share_rounding_never_increases_exposure():
    """Rounding toward zero, so a rounded order never breaches a weight cap."""
    from swingbot.execution.orders import _round_lot

    assert _round_lot(10.9, 1) == 10.0
    assert _round_lot(-10.9, 1) == -10.0
    assert _round_lot(107.0, 25) == 100.0
    assert _round_lot(-107.0, 25) == -100.0


def test_tiny_orders_are_suppressed():
    """A two-share trade pays a full round trip to move the book by nothing."""
    from swingbot.execution import build_orders

    portfolio = TargetPortfolio(
        decision_session=dt.date(2024, 3, 15), entry_session=dt.date(2024, 3, 18),
        positions=[TargetPosition("AAA", 0.0001, 0.1, "equity", "Tech")],
    )
    orders = build_orders(
        portfolio, {"AAA": 100.0}, 1_000_000, {}, min_order_value=5_000
    )
    assert orders == []


# ---------------------------------------------------------------------------- misc


def test_config_layering_and_hashing():
    base = load_config("us")
    overridden = load_config("us", set_values=["portfolio.n_long=12"])
    assert overridden.portfolio.n_long == 12
    assert overridden.config_hash() != base.config_hash()

    # Paths and notification settings must not affect the reproducibility hash.
    cosmetic = load_config("us", set_values=["run.log_level=DEBUG"])
    assert cosmetic.config_hash() == base.config_hash()


def test_market_aliases_resolve():
    for alias in ("in", "india", "nse"):
        assert load_config(alias).market_profile.name == "india"
    for alias in ("us", "usa", "nasdaq"):
        assert load_config(alias).market_profile.name == "us"


def test_unknown_market_is_rejected():
    with pytest.raises(ValueError, match="Unknown market"):
        load_config("atlantis")


def test_blend_weights_default_to_ninety_ten():
    """The split the user asked for, as a config default rather than an emergent property."""
    price, news = load_config("us").model.blend.normalised
    assert price == pytest.approx(0.9)
    assert news == pytest.approx(0.1)

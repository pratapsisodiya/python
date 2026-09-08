"""Portfolio construction and risk limits.

Limits are checked against randomly generated forecast vectors rather than one hand-picked
case, because the limits interact: capping a name lowers gross, capping a sector lowers it
again, and re-normalising gross can re-breach both. A single example proves nothing about
a fixed point.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from swingbot.config import load_config
from swingbot.portfolio import PortfolioConstructor, RiskLimits, apply_limits, check_limits
from swingbot.portfolio.risk import RiskState, update_kill_switch

SESSIONS = (dt.date(2024, 3, 15), dt.date(2024, 3, 18))


@pytest.mark.parametrize("seed", range(25))
@pytest.mark.parametrize("n_sectors", [2, 4, 8])
def test_limits_are_never_violated(seed, n_sectors):
    """Random forecasts, random sector structure: no limit may ever be breached.

    Two sectors under a 35% sector cap makes a 100% gross book infeasible, which is
    deliberately included: the correct response is to run under-leveraged, never to
    breach the cap to reach the leverage target.
    """
    rng = np.random.default_rng(seed)
    n = 40
    tickers = [f"T{i:02d}" for i in range(n)]
    weights = pd.Series(rng.normal(0, 0.4, n), index=tickers)
    sectors = pd.Series([f"S{i % n_sectors}" for i in range(n)], index=tickers)

    limits = RiskLimits()
    adjusted, _ = apply_limits(weights, limits=limits, sectors=sectors)
    problems = check_limits(adjusted, limits=limits, sectors=sectors)
    assert not problems, f"seed={seed} sectors={n_sectors}: {problems}"


def test_gross_is_never_inflated_past_a_binding_cap():
    """With two sectors and a 35% cap, gross must stay near 70%, not be forced to 100%."""
    tickers = [f"T{i:02d}" for i in range(20)]
    weights = pd.Series(np.linspace(-1, 1, 20), index=tickers)
    sectors = pd.Series([f"S{i % 2}" for i in range(20)], index=tickers)

    limits = RiskLimits(gross_leverage=1.0, max_sector_weight=0.35)
    adjusted, notes = apply_limits(weights, limits=limits, sectors=sectors)

    assert adjusted.abs().sum() <= 0.71
    assert any("under-leveraged" in n for n in notes), (
        "an infeasible leverage target must be reported, not silently accepted"
    )


@pytest.mark.parametrize("market", ["us", "india"])
def test_long_short_book_is_built_for_both_markets(market):
    cfg = load_config(market)
    constructor = PortfolioConstructor.from_config(cfg)
    rng = np.random.default_rng(3)
    tickers = [f"T{i:02d}" for i in range(60)]

    portfolio = constructor.build(
        pd.Series(rng.normal(0, 1, 60), index=tickers),
        decision_session=SESSIONS[0],
        entry_session=SESSIONS[1],
        volatility=pd.Series(np.abs(rng.normal(0.35, 0.1, 60)), index=tickers),
        sectors=pd.Series([f"S{i % 8}" for i in range(60)], index=tickers),
    )
    assert portfolio.n_long == cfg.portfolio.n_long
    assert portfolio.n_short == cfg.portfolio.n_short
    # Gross leverage is a ceiling, not a target. Volatility targeting decides how big the
    # book should be and the leverage limit caps it, so a book that comes in under the
    # limit is correct rather than a shortfall. Asserting equality here would re-encode
    # the bug that made `target_vol_annual` inert.
    assert portfolio.gross <= cfg.portfolio.gross_leverage + 1e-6
    assert portfolio.gross > 0.0


def test_india_shorts_use_futures_and_us_shorts_do_not():
    """NSE delivery cannot be held short overnight, so the instrument must differ."""
    rng = np.random.default_rng(3)
    tickers = [f"T{i:02d}" for i in range(60)]
    scores = pd.Series(rng.normal(0, 1, 60), index=tickers)
    vol = pd.Series(np.abs(rng.normal(0.35, 0.1, 60)), index=tickers)
    sectors = pd.Series([f"S{i % 8}" for i in range(60)], index=tickers)

    instruments = {}
    for market in ("us", "india"):
        portfolio = PortfolioConstructor.from_config(load_config(market)).build(
            scores, decision_session=SESSIONS[0], entry_session=SESSIONS[1],
            volatility=vol, sectors=sectors,
        )
        instruments[market] = {p.instrument for p in portfolio.positions if p.weight < 0}

    assert instruments["india"] == {"futures"}
    assert "futures" not in instruments["us"]


def test_long_only_when_shorts_disabled():
    cfg = load_config("india", set_values=["market_profile.short_instrument=none"])
    assert not cfg.shorts_allowed
    portfolio = PortfolioConstructor.from_config(cfg).build(
        pd.Series(np.linspace(-1, 1, 60), index=[f"T{i:02d}" for i in range(60)]),
        decision_session=SESSIONS[0], entry_session=SESSIONS[1],
    )
    assert portfolio.n_short == 0


def test_no_trade_band_suppresses_small_changes():
    """A tiny weight change must not generate a trade."""
    cfg = load_config("us", set_values=["portfolio.no_trade_band=0.02"])
    constructor = PortfolioConstructor.from_config(cfg)
    rng = np.random.default_rng(11)
    tickers = [f"T{i:02d}" for i in range(60)]
    scores = pd.Series(rng.normal(0, 1, 60), index=tickers)
    vol = pd.Series(np.abs(rng.normal(0.35, 0.1, 60)), index=tickers)

    first = constructor.build(
        scores, decision_session=SESSIONS[0], entry_session=SESSIONS[1], volatility=vol
    )
    # Re-run with almost identical scores; the band should hold most positions.
    second = constructor.build(
        scores + rng.normal(0, 0.01, 60), decision_session=SESSIONS[0],
        entry_session=SESSIONS[1], volatility=vol, previous_weights=first.weights,
    )
    held = sum(
        1 for t, w in second.weights.items()
        if t in first.weights and w == pytest.approx(first.weights[t])
    )
    assert held > 0, "the no-trade band never held a position"


def test_band_never_opens_a_new_position():
    """The band may keep an existing weight; it must never create one from nothing."""
    cfg = load_config("us", set_values=["portfolio.no_trade_band=0.05"])
    constructor = PortfolioConstructor.from_config(cfg)
    tickers = [f"T{i:02d}" for i in range(60)]
    scores = pd.Series(np.linspace(-1, 1, 60), index=tickers)

    portfolio = constructor.build(
        scores, decision_session=SESSIONS[0], entry_session=SESSIONS[1],
        previous_weights={},
    )
    assert all(abs(w) > 1e-9 for w in portfolio.weights.values())


def test_shorts_only_taken_on_negative_scores():
    """A 'short' whose score is positive is a bet the model never made."""
    cfg = load_config("us")
    constructor = PortfolioConstructor.from_config(cfg)
    # Every score positive: there should be no shorts at all.
    tickers = [f"T{i:02d}" for i in range(60)]
    portfolio = constructor.build(
        pd.Series(np.linspace(0.1, 1.0, 60), index=tickers),
        decision_session=SESSIONS[0], entry_session=SESSIONS[1],
    )
    assert portfolio.n_short == 0


def test_kill_switch_scales_then_flattens_then_recovers():
    state = RiskState()
    path = [1.0, 1.05, 1.0, 0.96, 0.93]
    for equity in path:
        state = update_kill_switch(state, equity, scale_at=0.08, flat_at=0.15)
    assert state.scale == 0.5, "gross should be halved after an 8% drawdown"

    for equity in (0.88, 0.85):
        state = update_kill_switch(state, equity, scale_at=0.08, flat_at=0.15)
    assert state.scale == 0.0, "the book should be flat after a 15% drawdown"
    assert state.cooldown_remaining > 0


def test_kill_switch_can_be_disabled():
    state = RiskState()
    for equity in (1.0, 0.5):
        state = update_kill_switch(state, equity, enabled=False)
    assert state.scale == 1.0


def test_flat_risk_scale_produces_no_positions():
    portfolio = PortfolioConstructor.from_config(load_config("us")).build(
        pd.Series(np.linspace(-1, 1, 60), index=[f"T{i:02d}" for i in range(60)]),
        decision_session=SESSIONS[0], entry_session=SESSIONS[1], risk_scale=0.0,
    )
    assert not portfolio.positions
    assert any("kill-switch" in n for n in portfolio.notes)


def test_ineligible_names_are_excluded():
    tickers = [f"T{i:02d}" for i in range(60)]
    eligible = pd.Series([i % 2 == 0 for i in range(60)], index=tickers)
    portfolio = PortfolioConstructor.from_config(load_config("us")).build(
        pd.Series(np.linspace(-1, 1, 60), index=tickers),
        decision_session=SESSIONS[0], entry_session=SESSIONS[1], eligible=eligible,
    )
    assert all(int(p.ticker[1:]) % 2 == 0 for p in portfolio.positions)


def test_volatility_targeting_actually_changes_book_size():
    """A tighter volatility target must produce a smaller book.

    This was inert for the whole first version of the system: `apply_limits` normalised
    gross to `gross_leverage` on entry, so the scale that volatility targeting had just
    applied was immediately erased and a 5 percent target produced a byte-identical book
    to a 30 percent one. The config advertised it and the README described it.
    """
    rng = np.random.default_rng(5)
    tickers = [f"T{i:02d}" for i in range(40)]
    scores = pd.Series(rng.normal(0, 1, 40), index=tickers)
    vol = pd.Series(np.abs(rng.normal(0.35, 0.10, 40)), index=tickers)
    sectors = pd.Series([f"S{i % 8}" for i in range(40)], index=tickers)

    def gross_at(target_vol: float) -> float:
        cfg = load_config(
            "us",
            set_values=[
                f"portfolio.target_vol_annual={target_vol}",
                # Loosen the caps so this measures volatility targeting rather than
                # whichever concentration limit binds first.
                "portfolio.max_weight=0.9",
                "portfolio.max_sector_weight=1.0",
                "portfolio.max_net_exposure=1.0",
            ],
        )
        book = PortfolioConstructor.from_config(cfg).build(
            scores, decision_session=SESSIONS[0], entry_session=SESSIONS[1],
            volatility=vol, sectors=sectors,
        )
        return book.gross

    tight, loose = gross_at(0.04), gross_at(0.25)
    assert tight < loose, (
        f"a 4% volatility target produced gross {tight:.3f} and a 25% target "
        f"{loose:.3f}. Volatility targeting is not sizing the book."
    )


def test_gross_leverage_is_never_exceeded_even_at_a_high_vol_target():
    """The ceiling still holds when volatility targeting wants a bigger book."""
    rng = np.random.default_rng(6)
    tickers = [f"T{i:02d}" for i in range(40)]
    cfg = load_config(
        "us",
        set_values=["portfolio.target_vol_annual=2.0", "portfolio.max_weight=0.9",
                    "portfolio.max_sector_weight=1.0"],
    )
    book = PortfolioConstructor.from_config(cfg).build(
        pd.Series(rng.normal(0, 1, 40), index=tickers),
        decision_session=SESSIONS[0], entry_session=SESSIONS[1],
        volatility=pd.Series(np.full(40, 0.05), index=tickers),
        sectors=pd.Series([f"S{i % 8}" for i in range(40)], index=tickers),
    )
    assert book.gross <= cfg.portfolio.gross_leverage + 1e-6


def test_participation_cap_truncates_and_reports():
    """The capacity cap must bind on an illiquid book and say so.

    `max_participation_adv` shipped in the config, documented as capping orders at a
    fraction of average daily volume, and nothing in the pipeline read it.
    """
    rng = np.random.default_rng(7)
    tickers = [f"T{i:02d}" for i in range(30)]
    scores = pd.Series(rng.normal(0, 1, 30), index=tickers)
    vol = pd.Series(np.abs(rng.normal(0.35, 0.1, 30)), index=tickers)
    # Deliberately thin names against a large account: every order should be capacity
    # constrained.
    adv = pd.Series(np.full(30, 50_000.0), index=tickers)

    cfg = load_config("us", set_values=["portfolio.max_participation_adv=0.05"])
    book = PortfolioConstructor.from_config(cfg).build(
        scores, decision_session=SESSIONS[0], entry_session=SESSIONS[1],
        volatility=vol, adv_notional=adv, equity=50_000_000.0,
    )
    assert any("participation cap" in n for n in book.notes), (
        f"a 50m account trading names with 50k daily volume was not capacity capped; "
        f"notes were {book.notes}"
    )
    assert book.gross < cfg.portfolio.gross_leverage, (
        "a capacity-constrained book must run smaller, not be rescaled back to full gross"
    )


def test_capacity_cap_is_not_rescaled_away():
    """Truncated orders must stay truncated.

    Re-normalising gross after a capacity truncation would re-inflate exactly the
    positions the cap just declared unreachable, which is the subtle way a capacity
    control becomes decorative.
    """
    rng = np.random.default_rng(8)
    tickers = [f"T{i:02d}" for i in range(30)]
    scores = pd.Series(rng.normal(0, 1, 30), index=tickers)
    adv = pd.Series(np.full(30, 500_000.0), index=tickers)

    def gross_at(cap: float) -> float:
        cfg = load_config("us", set_values=[f"portfolio.max_participation_adv={cap}"])
        return PortfolioConstructor.from_config(cfg).build(
            scores, decision_session=SESSIONS[0], entry_session=SESSIONS[1],
            adv_notional=adv, equity=2_000_000.0,
        ).gross

    # Sized so the tight cap still clears the dust threshold: an allowed weight below
    # `min_position_weight` empties the book entirely and both sides would read zero,
    # which would pass a naive inequality for the wrong reason.
    tight, loose = gross_at(0.02), gross_at(1.0)
    assert tight > 0.0, "the tight cap emptied the book; choose a less extreme fixture"
    assert tight < loose, (
        f"a tighter participation cap gave gross {tight:.4f} versus {loose:.4f} — "
        "truncated orders are being rescaled back up"
    )


# --------------------------------------------------------------------------------------
# The Kelly ceiling
#
# It shipped in the config as a risk control and could not fire, for two independent
# reasons. `construct._size` handed `kelly_cap` the cross-sectional score as `mu` — a rank
# in roughly [-0.5, 0.5], not a return — which puts the implied ceiling an order of
# magnitude above any weight the book runs. And the cap was applied *before* the gross
# normalisation, which multiplies the whole vector and hands straight back whatever the
# cap took off.
#
# Each test below fails if either regression returns.
# --------------------------------------------------------------------------------------


def _kelly_book(fraction: float, *, mu_scale: float = 0.01, n: int = 30, seed: int = 11):
    rng = np.random.default_rng(seed)
    tickers = [f"K{i:02d}" for i in range(n)]
    scores = pd.Series(rng.normal(0, 1, n), index=tickers)
    vol = pd.Series(np.abs(rng.normal(0.35, 0.05, n)) + 0.10, index=tickers)
    expected = scores * mu_scale

    cfg = load_config(
        "us",
        set_values=[
            "portfolio.sizing=kelly",
            f"portfolio.kelly_fraction={fraction}",
            # Loosened so the ceiling under test is the thing that binds, not the
            # per-name cap that would clip the same weights for a different reason.
            "portfolio.max_weight=0.90",
            "portfolio.max_sector_weight=1.0",
            "portfolio.max_net_exposure=1.0",
        ],
    )
    book = PortfolioConstructor.from_config(cfg).build(
        scores, decision_session=SESSIONS[0], entry_session=SESSIONS[1],
        volatility=vol, expected_returns=expected,
    )
    return book, expected, vol


def test_kelly_ceiling_is_never_exceeded():
    """Every final weight must sit under its own fractional-Kelly limit.

    This is the assertion the ordering bug fails: capping and then re-normalising gross
    restores the very weights the cap removed, so the book ends up above the ceiling
    while still *looking* like it was capped.
    """
    fraction = 0.10
    book, expected, vol = _kelly_book(fraction)
    assert book.positions, "fixture produced an empty book"

    for position in book.positions:
        mu = float(expected[position.ticker])
        sigma = float(vol[position.ticker])
        ceiling = min(abs(mu) / sigma**2 * fraction, 1.0)
        assert abs(position.weight) <= ceiling + 1e-9, (
            f"{position.ticker} carries weight {position.weight:+.4f} against a "
            f"fractional-Kelly ceiling of {ceiling:.4f}"
        )


def test_kelly_fraction_changes_the_book():
    """A tighter fraction must produce a smaller book. Otherwise it is decorative."""
    tight, _, _ = _kelly_book(0.05)
    loose, _, _ = _kelly_book(5.0)
    assert tight.gross < loose.gross, (
        f"kelly_fraction 0.05 gave gross {tight.gross:.4f} and 5.0 gave "
        f"{loose.gross:.4f} — the ceiling is not binding"
    )
    assert any("Kelly ceiling" in n for n in tight.notes), (
        f"a binding Kelly ceiling was not reported; notes were {tight.notes}"
    )


def test_kelly_without_a_calibrated_mu_falls_back_loudly():
    """No expected return means no ceiling — and the book has to say so.

    Silently sizing as though a ceiling had been applied is the failure mode this whole
    pass exists to remove, so the absence of calibration is a reported note, never an
    assumed zero. A zero `mu` would imply a ceiling of zero and delete the book.
    """
    rng = np.random.default_rng(12)
    tickers = [f"K{i:02d}" for i in range(30)]
    scores = pd.Series(rng.normal(0, 1, 30), index=tickers)
    vol = pd.Series(np.abs(rng.normal(0.35, 0.05, 30)) + 0.10, index=tickers)

    cfg = load_config("us", set_values=["portfolio.sizing=kelly"])
    book = PortfolioConstructor.from_config(cfg).build(
        scores, decision_session=SESSIONS[0], entry_session=SESSIONS[1],
        volatility=vol,
    )
    assert book.positions, "the fallback emptied the book instead of sizing it"
    assert any("NO Kelly ceiling" in n for n in book.notes), (
        f"an uncalibrated Kelly book did not announce the fallback; notes were {book.notes}"
    )


def test_a_name_without_a_calibrated_mu_is_left_uncapped_not_deleted():
    """A missing estimate declines to cap. It is not evidence that the edge is zero."""
    fraction = 0.10
    rng = np.random.default_rng(13)
    tickers = [f"K{i:02d}" for i in range(30)]
    scores = pd.Series(rng.normal(0, 1, 30), index=tickers)
    vol = pd.Series(np.full(30, 0.35), index=tickers)
    expected = scores * 0.01
    # The three strongest longs lose their estimate. They are the names most likely to be
    # deleted by a `fillna(0.0)`, which is exactly what used to happen.
    missing = list(scores.sort_values(ascending=False).index[:3])
    expected.loc[missing] = np.nan

    cfg = load_config(
        "us",
        set_values=[
            "portfolio.sizing=kelly",
            f"portfolio.kelly_fraction={fraction}",
            "portfolio.max_weight=0.90",
            "portfolio.max_sector_weight=1.0",
            "portfolio.max_net_exposure=1.0",
        ],
    )
    book = PortfolioConstructor.from_config(cfg).build(
        scores, decision_session=SESSIONS[0], entry_session=SESSIONS[1],
        volatility=vol, expected_returns=expected,
    )
    held = {p.ticker for p in book.positions}
    assert held & set(missing), (
        "names with no calibrated expected return were dropped from the book; a missing "
        "mu must leave a position uncapped, not zero it"
    )
    assert any("uncapped" in n for n in book.notes), book.notes

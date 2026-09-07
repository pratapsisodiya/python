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
    assert portfolio.gross == pytest.approx(cfg.portfolio.gross_leverage, abs=0.05)


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

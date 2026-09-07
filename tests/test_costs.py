"""Cost model behaviour.

Costs decide whether a weekly strategy is real, so the model has to be right in shape as
well as in magnitude. These tests pin the properties that matter and the per-market
statutory charges that are easy to get wrong.
"""

from __future__ import annotations

import pandas as pd
import pytest

from swingbot.backtest.costs import CostModel, round_trip_bps
from swingbot.config import load_config


def test_india_costs_exceed_us_costs():
    """India must be materially more expensive: STT is charged on both sides."""
    india = round_trip_bps(load_config("india"))
    us = round_trip_bps(load_config("us"))
    assert india > us * 2, (
        f"India round trip {india:.1f} bps versus US {us:.1f} bps. India charges 0.1% STT "
        "each side on delivery, so it should be several times the US cost."
    )
    assert 40 < india < 100, f"India round trip {india:.1f} bps is outside a plausible range"
    assert 10 < us < 40, f"US round trip {us:.1f} bps is outside a plausible range"


def test_india_shorts_are_cheaper_than_longs():
    """Futures carry lower statutory charges than cash delivery."""
    cfg = load_config("india")
    assert round_trip_bps(cfg, is_short=True) < round_trip_bps(cfg)


def test_stt_is_charged_on_both_sides_in_india():
    cfg = load_config("india")
    model = CostModel.from_config(cfg, scale=1.0)
    buy = model.trade_cost(notional=1e6, is_buy=True, is_short_leg=False)
    sell = model.trade_cost(notional=1e6, is_buy=False, is_short_leg=False)
    assert buy.taxes > 0 and sell.taxes > 0, "STT must apply to both sides on delivery"


def test_cost_is_monotone_in_size():
    """A larger order can never cost less."""
    model = CostModel.from_config(load_config("us"), scale=1.0)
    previous = -1.0
    for notional in (1e4, 1e5, 1e6, 1e7):
        cost = model.trade_cost(
            notional=notional, is_buy=True, is_short_leg=False, adv_notional=1e8
        ).total
        assert cost > previous
        previous = cost


def test_impact_grows_sublinearly():
    """Square-root impact: doubling size must raise impact by less than double."""
    model = CostModel.from_config(load_config("us"), scale=1.0)
    small = model._impact_bps(1e6, 0.30, 1e8)
    large = model._impact_bps(4e6, 0.30, 1e8)
    assert large > small
    assert large < small * 4.0, (
        f"impact went from {small:.2f} to {large:.2f} bps on a 4x order. Square-root "
        "impact should roughly double, not quadruple."
    )


def test_spread_scales_with_volatility_and_stays_sane():
    """A wider daily range implies a wider spread, but within believable bounds."""
    model = CostModel.from_config(load_config("us"), scale=1.0)
    narrow = model._half_spread_bps(0.01)
    wide = model._half_spread_bps(0.06)
    assert wide > narrow
    assert narrow <= 5.0, f"a 1% daily range implies {narrow:.1f} bps half-spread, too wide"
    assert wide <= 50.0, "the spread cap is not binding"


def test_spread_floor_applies_without_range_data():
    cfg = load_config("us")
    model = CostModel.from_config(cfg, scale=1.0)
    assert model._half_spread_bps(None) == cfg.costs.min_half_spread_bps


def test_cost_scale_is_linear_on_trade_costs():
    """The sensitivity sweep depends on this scaling exactly."""
    cfg = load_config("us")
    base = CostModel.from_config(cfg, scale=1.0).trade_cost(
        notional=1e6, is_buy=True, is_short_leg=False, adv_notional=1e8
    ).total
    doubled = CostModel.from_config(cfg, scale=2.0).trade_cost(
        notional=1e6, is_buy=True, is_short_leg=False, adv_notional=1e8
    ).total
    assert doubled == pytest.approx(base * 2.0, rel=1e-9)


def test_borrow_accrues_with_holding_period():
    model = CostModel.from_config(load_config("us"), scale=1.0)
    week = model.holding_cost(short_notional=1e6, sessions_held=5).total
    month = model.holding_cost(short_notional=1e6, sessions_held=21).total
    assert month > week * 3


def test_futures_roll_only_charged_on_long_holds():
    """A one-week hold does not cross a monthly roll; a one-month hold does."""
    model = CostModel.from_config(load_config("india"), scale=1.0)
    assert model.holding_cost(short_notional=1e6, sessions_held=5, rolls=0).roll == 0.0
    assert model.holding_cost(short_notional=1e6, sessions_held=21, rolls=1).roll > 0.0


def test_zero_turnover_costs_nothing():
    """Holding an unchanged book must be free apart from carry."""
    model = CostModel.from_config(load_config("us"), scale=1.0)
    weights = pd.Series({"A": 0.2, "B": 0.3})
    cost, turnover = model.rebalance_cost(weights, weights, equity=1e6)
    assert turnover == pytest.approx(0.0)
    assert cost.commission == 0.0 and cost.spread == 0.0


def test_rebalance_turnover_is_one_way():
    """Turnover is half the sum of absolute weight changes, by convention."""
    model = CostModel.from_config(load_config("us"), scale=1.0)
    previous = pd.Series({"A": 0.5, "B": 0.5})
    target = pd.Series({"A": 0.0, "C": 0.5, "B": 0.5})
    _, turnover = model.rebalance_cost(target, previous, equity=1e6)
    assert turnover == pytest.approx(0.5)

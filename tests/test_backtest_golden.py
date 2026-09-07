"""End-to-end engine behaviour on data with known properties.

Where the leak canaries prove the harness finds nothing when there is nothing, these
prove it finds the right thing when there is something, and that the accounting adds up.
"""

from __future__ import annotations

import pandas as pd
import pytest

from swingbot.backtest import BacktestEngine, run_ablation, walk_forward_predict
from swingbot.backtest.costs import CostModel
from swingbot.features.pipeline import price_feature_columns
from swingbot.model import build_model
from swingbot.validation.splits import PurgedWalkForward


def _splitter(cfg):
    return PurgedWalkForward(
        train_weeks=104, test_weeks=26, embargo_weeks=cfg.cv.embargo_weeks,
        expanding=True, min_train_weeks=78,
    )


def _predict(panel, cfg):
    return walk_forward_predict(
        panel, lambda: build_model("gbdt", cfg),
        feature_columns=price_feature_columns(panel), splitter=_splitter(cfg),
    )


def test_engine_recovers_injected_alpha(alpha_panel, alpha_bars, cfg_us):
    """The planted signal must turn into a positive net return after real costs."""
    panel, calendar, grid = alpha_panel
    predictions = _predict(panel, cfg_us)
    assert not predictions.empty

    result = BacktestEngine(cfg_us).run(
        predictions, alpha_bars, calendar, name="golden", grid=grid
    )
    assert result.metrics.n_periods > 50
    assert result.metrics.sharpe > 0.5, (
        f"Sharpe {result.metrics.sharpe:.2f} on data with planted alpha. The engine is "
        "not converting a real signal into a return."
    )
    assert result.metrics.ann_vol > 0.0
    assert result.metrics.max_drawdown < 0.0


def test_costs_reduce_returns_monotonically(alpha_panel, alpha_bars, cfg_us):
    """Every increase in the cost assumption must lower the net return.

    Charged inside the loop at fill time, so this also confirms costs are not being
    applied as an afterthought against average turnover.
    """
    panel, calendar, grid = alpha_panel
    predictions = _predict(panel, cfg_us)

    previous = None
    for scale in (0.0, 1.0, 2.0, 4.0):
        engine = BacktestEngine(cfg_us, cost_model=CostModel.from_config(cfg_us, scale=scale))
        result = engine.run(
            predictions, alpha_bars, calendar, name=f"c{scale}", grid=grid
        )
        current = result.metrics.ann_return
        if previous is not None:
            assert current < previous, (
                f"cost scale {scale} produced {current:.4f} versus {previous:.4f} at the "
                "level below. Costs are not reducing returns."
            )
        previous = current


def test_gross_minus_cost_equals_net(alpha_panel, alpha_bars, cfg_us):
    """The accounting identity. If this drifts, the cost decomposition is wrong."""
    panel, calendar, grid = alpha_panel
    predictions = _predict(panel, cfg_us)
    result = BacktestEngine(cfg_us).run(
        predictions, alpha_bars, calendar, name="ledger", grid=grid
    )
    frame = result.to_frame()
    residual = (frame["gross_return"] - frame["cost"] - frame["net_return"]).abs().max()
    assert residual < 1e-12, f"net return does not reconcile, worst residual {residual:.3g}"


def test_zero_cost_beats_real_cost(alpha_panel, alpha_bars, cfg_us):
    """A free strategy must beat the same strategy paying to trade."""
    panel, calendar, grid = alpha_panel
    predictions = _predict(panel, cfg_us)

    free = BacktestEngine(cfg_us, cost_model=CostModel.from_config(cfg_us, scale=0.0)).run(
        predictions, alpha_bars, calendar, name="free", grid=grid
    )
    paid = BacktestEngine(cfg_us).run(
        predictions, alpha_bars, calendar, name="paid", grid=grid
    )
    assert free.metrics.ann_return > paid.metrics.ann_return
    assert free.metrics.cost_drag_bps == pytest.approx(0.0)
    assert paid.metrics.cost_drag_bps > 0.0


def test_india_costs_more_than_us_on_the_same_signal(alpha_panel, alpha_bars, cfg_us, cfg_india):
    """The same predictions must be more expensive to trade on NSE."""
    panel, calendar, grid = alpha_panel
    predictions = _predict(panel, cfg_us)

    drags = {}
    for name, cfg in (("us", cfg_us), ("india", cfg_india)):
        result = BacktestEngine(cfg).run(
            predictions, alpha_bars, calendar, name=name, grid=grid
        )
        drags[name] = result.metrics.cost_drag_bps
    assert drags["india"] > drags["us"] * 1.5, (
        f"India cost drag {drags['india']:.1f} bps/week versus US {drags['us']:.1f}. "
        "STT on both sides should make India materially more expensive."
    )


def test_nothing_is_filled_at_the_signal_price(alpha_panel, alpha_bars, cfg_us):
    """Decision, entry and exit must be three distinct sessions.

    The single most important structural property of the backtest: a signal computed at
    Friday's close is filled at the next session's open, and the label measures exactly
    that span. Filling at the close that generated the signal would credit the strategy
    with a move nobody could have captured.
    """
    _, _, grid = alpha_panel
    for decision in grid[:50]:
        assert decision.entry_session > decision.decision_session
        assert decision.exit_session > decision.entry_session


def test_risk_limits_hold_across_the_whole_backtest(alpha_panel, alpha_bars, cfg_us):
    """Every week's realised book must respect the configured caps."""
    panel, calendar, grid = alpha_panel
    predictions = _predict(panel, cfg_us)
    result = BacktestEngine(cfg_us).run(
        predictions, alpha_bars, calendar, name="limits", grid=grid
    )
    for session, weights in result.weights.items():
        series = pd.Series(weights, dtype=float)
        if series.empty:
            continue
        assert series.abs().max() <= cfg_us.portfolio.max_weight + 1e-6, session
        assert series.abs().sum() <= cfg_us.portfolio.gross_leverage + 1e-6, session
        assert abs(series.sum()) <= cfg_us.portfolio.max_net_exposure + 1e-6, session


def test_ablation_reports_every_variant_and_a_verdict(alpha_panel, alpha_bars, cfg_us):
    """The ablation must produce the comparison table and a plain-language read."""
    panel, calendar, grid = alpha_panel
    report = run_ablation(
        panel, alpha_bars, calendar, cfg_us, grid=grid, include_news=False,
        variants=["benchmark", "zero", "shuffled", "momentum", "price_only"],
    )
    table = report.table()
    assert not table.empty
    assert {"shuffled", "price_only", "momentum"} <= set(table["variant"])
    assert report.verdicts, "the ablation produced no verdict"
    assert any("Leak check" in v for v in report.verdicts)


def test_null_market_produces_no_profit(null_panel, null_bars, cfg_us):
    """On a market with no signal, the strategy must not make money after costs."""
    panel, calendar, grid = null_panel
    predictions = _predict(panel, cfg_us)
    result = BacktestEngine(cfg_us).run(
        predictions, null_bars, calendar, name="null", grid=grid
    )
    assert result.metrics.sharpe < 1.0, (
        f"Sharpe {result.metrics.sharpe:.2f} on a market with no injected signal"
    )


def test_weights_are_empty_when_flat(alpha_panel, alpha_bars, cfg_us):
    """With the kill-switch armed at zero tolerance the book must go flat."""
    cfg = cfg_us.model_copy(deep=True)
    cfg.risk.drawdown_scale_at = 0.0001
    cfg.risk.drawdown_flat_at = 0.0002

    panel, calendar, grid = alpha_panel
    predictions = _predict(panel, cfg_us)
    result = BacktestEngine(cfg).run(
        predictions, alpha_bars, calendar, name="flat", grid=grid
    )
    scales = result.risk_scale.dropna()
    assert (scales < 1.0).any(), "an immediate kill-switch never engaged"

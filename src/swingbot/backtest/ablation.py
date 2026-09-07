"""The ablation matrix.

This is the honest-answer machine. A single Sharpe ratio tells you nothing on its own; the
question is always "compared to what". So every backtest runs the full matrix and the
report leads with the comparison rather than the headline.

The variants, in order of what they rule out:

``benchmark``       Buy and hold the equal-weighted universe. The thing you must beat to
                    justify any of this effort.
``zero``            Hold nothing. The cost floor.
``random``          Random scores at matched turnover and volatility target. Some Sharpe
                    comes from rebalancing and vol targeting alone; this measures it.
``shuffled``        The real model, labels permuted within each week. **Must be near
                    zero.** If it is not, the pipeline leaks and every other number here
                    is void. This is the most important row in the table.
``momentum``        The known, free, published factor. Complexity that cannot beat this
                    has not earned its place.
``price_only``      The honest denominator for the news question.
``news_only``       Is there standalone news signal, or only an interaction?
``price_news``      The product.

The headline claim about news is ``Sharpe(price_news) - Sharpe(price_only)`` with a
bootstrap interval on the *difference*, not on either level. Those two return series share
most of their variance, so the difference is estimated far more precisely than either
Sharpe, and comparing two separately-computed Sharpe ratios throws that precision away.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from ..calendars import TradingCalendar, WeeklyDecision
from ..config import Config
from ..features.pipeline import news_feature_columns, price_feature_columns
from ..model import (
    BlendedModel,
    MomentumOnlyModel,
    RandomSignModel,
    ShuffledLabelModel,
    SubsetModel,
    ZeroModel,
    build_model,
)
from ..types import DECISION_SESSION, TICKER
from ..validation.metrics import sharpe_difference_test
from ..validation.splits import PurgedWalkForward
from .engine import BacktestEngine, walk_forward_predict
from .results import BacktestResult

log = logging.getLogger(__name__)


@dataclass(slots=True)
class AblationReport:
    results: dict[str, BacktestResult] = field(default_factory=dict)
    news_effect: dict[str, float] = field(default_factory=dict)
    verdicts: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)

    def table(self) -> pd.DataFrame:
        if not self.results:
            return pd.DataFrame()
        order = [
            "benchmark",
            "zero",
            "random",
            "shuffled",
            "momentum",
            "price_only",
            "news_only",
            "price_news",
        ]
        rows = [
            self.results[k].summary_row()
            for k in order
            if k in self.results and self.results[k].metrics.n_periods > 0
        ]
        rows += [
            r.summary_row()
            for k, r in self.results.items()
            if k not in order and r.metrics.n_periods > 0
        ]
        return pd.DataFrame(rows)

    def headline(self) -> str:
        return " ".join(self.verdicts) if self.verdicts else "no verdict"


def run_ablation(
    panel: pd.DataFrame,
    bars: pd.DataFrame,
    calendar: TradingCalendar,
    cfg: Config,
    *,
    grid: list[WeeklyDecision] | None = None,
    eligibility: pd.DataFrame | None = None,
    caveats: list[str] | None = None,
    variants: list[str] | None = None,
    include_news: bool = True,
) -> AblationReport:
    """Run every variant on the same folds and compare them."""
    report = AblationReport(caveats=list(caveats or []))
    if panel.empty:
        report.verdicts.append("no data to backtest")
        return report

    price_cols = price_feature_columns(panel)
    news_cols = news_feature_columns(panel) if include_news else []
    all_cols = price_cols + news_cols

    splitter = PurgedWalkForward(
        train_weeks=cfg.cv.train_weeks,
        test_weeks=cfg.cv.test_weeks,
        embargo_weeks=cfg.cv.embargo_weeks,
        expanding=cfg.cv.expanding,
        min_train_weeks=cfg.cv.min_train_weeks,
    )
    engine = BacktestEngine(cfg)
    seed = cfg.run.seed

    def _run(name: str, factory, columns: list[str]) -> BacktestResult | None:
        predictions = walk_forward_predict(
            panel, factory, feature_columns=columns, splitter=splitter
        )
        if predictions.empty:
            log.warning("variant %s produced no out-of-sample predictions", name)
            return None
        return engine.run(
            predictions,
            bars,
            calendar,
            name=name,
            grid=grid,
            eligibility=eligibility,
            caveats=report.caveats,
        )

    selected = variants or [
        "benchmark",
        "zero",
        "random",
        "shuffled",
        "momentum",
        "price_only",
        "news_only",
        "price_news",
    ]
    if not news_cols:
        selected = [v for v in selected if v not in ("news_only", "price_news")]
        if "price_only" in selected:
            selected = [v if v != "price_only" else "price_only" for v in selected]

    for variant in selected:
        try:
            result = _dispatch(variant, panel, bars, calendar, cfg, engine, _run,
                               price_cols, news_cols, all_cols, grid, eligibility,
                               report.caveats, seed)
        except Exception as exc:  # a broken variant must not sink the whole run
            log.warning("variant %s failed: %s", variant, exc)
            continue
        if result is not None:
            report.results[variant] = result

    _judge(report, cfg)
    return report


def _dispatch(
    variant, panel, bars, calendar, cfg, engine, run, price_cols, news_cols, all_cols,
    grid, eligibility, caveats, seed,
):
    if variant == "benchmark":
        return _benchmark(panel, bars, calendar, cfg, grid)
    if variant == "zero":
        return run("zero", ZeroModel, price_cols)
    if variant == "random":
        return run("random", lambda: RandomSignModel(seed=seed), price_cols)
    if variant == "momentum":
        return run("momentum", MomentumOnlyModel, price_cols)
    if variant == "shuffled":
        return run(
            "shuffled",
            lambda: ShuffledLabelModel(build_model(cfg.model.price_model, cfg, seed=seed), seed=seed),
            price_cols,
        )
    if variant == "price_only":
        return run("price_only", lambda: build_model(cfg.model.price_model, cfg, seed=seed), price_cols)
    if variant == "news_only":
        if not news_cols:
            return None
        return run(
            "news_only",
            lambda: SubsetModel(
                build_model(cfg.model.news_model, cfg, seed=seed), news_cols, name="news_only"
            ),
            all_cols,
        )
    if variant == "price_news":
        if not news_cols:
            return None
        pw, nw = cfg.model.blend.normalised
        return run(
            "price_news",
            lambda: BlendedModel(
                build_model(cfg.model.price_model, cfg, seed=seed),
                build_model(cfg.model.news_model, cfg, seed=seed),
                price_columns=price_cols,
                news_columns=news_cols,
                price_weight=pw,
                news_weight=nw,
            ),
            all_cols,
        )
    raise ValueError(f"Unknown ablation variant {variant!r}")


def _benchmark(
    panel: pd.DataFrame,
    bars: pd.DataFrame,
    calendar: TradingCalendar,
    cfg: Config,
    grid: list[WeeklyDecision] | None,
) -> BacktestResult:
    """Equal-weighted buy and hold over the same weeks the strategy trades.

    Restricted to the same weeks so the comparison is like for like. A benchmark measured
    over a longer window would be comparing different market conditions.
    """
    from ..features.labels import forward_return_matrix

    grid = grid if grid is not None else calendar.weekly_grid(cfg.calendar.hold_sessions)
    realised = forward_return_matrix(bars, grid)
    weeks = sorted(set(panel[DECISION_SESSION].unique()) & set(realised))

    rows = []
    for week in weeks:
        names = panel.loc[panel[DECISION_SESSION] == week, TICKER]
        returns = realised[week].reindex(names).dropna()
        if returns.empty:
            continue
        rows.append({DECISION_SESSION: week, "net_return": float(returns.mean())})

    result = BacktestResult(name="benchmark")
    if not rows:
        return result.finalize()

    frame = pd.DataFrame(rows).set_index(DECISION_SESSION).sort_index()
    result.returns = frame["net_return"]
    result.gross_returns = frame["net_return"]
    result.costs = pd.Series(0.0, index=frame.index)
    result.turnover = pd.Series(0.0, index=frame.index)
    result.gross_exposure = pd.Series(1.0, index=frame.index)
    result.net_exposure = pd.Series(1.0, index=frame.index)
    result.risk_scale = pd.Series(1.0, index=frame.index)
    result.notes.append("equal-weighted buy and hold, no costs charged")
    return result.finalize()


def _judge(report: AblationReport, cfg: Config) -> None:
    """Turn the table into plain statements about what is and is not established."""
    results = report.results

    shuffled = results.get("shuffled")
    if shuffled is not None and not shuffled.predictions.empty:
        # The leak test reads the information coefficient, not the Sharpe ratio, and
        # only in the positive direction. Two reasons, both of which matter.
        #
        # Sharpe is contaminated by costs: a shuffled model trades on noise and pays a
        # full round trip for it, so its Sharpe is reliably negative even when the
        # pipeline is perfectly clean. Judging leakage on |Sharpe| would flag every
        # healthy run. IC is a rank correlation between forecast and outcome and is
        # untouched by costs, sizing or leverage.
        #
        # And leakage is directional. Predictive power on labels whose name-to-outcome
        # mapping has been destroyed can only come from information that should not be
        # there. A negative IC is just noise on the other side of zero.
        # Clustered by FOLD, not by week. The shuffled model is refit once per fold, so
        # within a fold its prediction function is fixed and every week shares the sign
        # of that function's IC. Clustering by week would count ~70 correlated weeks per
        # fold as independent and flag a clean pipeline as leaking, which is exactly what
        # it did before this was corrected.
        from ..validation.ic import fold_clustered_ic

        ic = fold_clustered_ic(shuffled.predictions)
        leaking = ic.mean_ic > 0.005 and ic.t_stat > 2.5
        if leaking:
            report.verdicts.append(
                f"LEAK WARNING: shuffled labels still predict, IC {ic.mean_ic:+.4f} "
                f"(t={ic.t_stat:.1f} across {ic.n_periods} fold(s)). The pipeline is "
                "leaking and no other number in this report is valid."
            )
        else:
            report.verdicts.append(
                f"Leak check passed: shuffled labels give IC {ic.mean_ic:+.4f} "
                f"(t={ic.t_stat:.1f} across {ic.n_periods} fold(s), clustered by fold), "
                "indistinguishable from zero as required."
            )

    primary = results.get("price_news") or results.get("price_only")
    momentum = results.get("momentum")
    if (
        primary is not None
        and momentum is not None
        and primary.metrics.sharpe <= momentum.metrics.sharpe
    ):
        report.verdicts.append(
            f"The model (Sharpe {primary.metrics.sharpe:.2f}) does not beat plain "
            f"momentum and reversal (Sharpe {momentum.metrics.sharpe:.2f}). "
            "The added complexity is not earning anything."
        )

    if primary is not None and primary.ic.n_periods > 20:
        report.verdicts.append(
            f"Signal quality: IC {primary.ic.mean_ic:+.4f} (t={primary.ic.t_stat:.1f}, "
            f"IC-IR {primary.ic.ic_ir:.2f}) over {primary.ic.n_periods} weeks "
            f"— {primary.ic.verdict()}."
        )

    random = results.get("random")
    if primary is not None and random is not None and random.metrics.sharpe > 0.2:
        report.verdicts.append(
            f"Note: random scores alone give Sharpe {random.metrics.sharpe:.2f} at this "
            "turnover and volatility target, so subtract that from any claim of skill."
        )

    price_only = results.get("price_only")
    price_news = results.get("price_news")
    if price_only is not None and price_news is not None:
        test = sharpe_difference_test(price_news.returns, price_only.returns)
        report.news_effect = test
        pw, nw = cfg.model.blend.normalised
        if test["p_value"] < 0.05 and test["difference"] > 0:
            report.verdicts.append(
                f"News adds {test['difference']:+.2f} Sharpe "
                f"(95% CI [{test['ci_low']:.2f}, {test['ci_high']:.2f}], p={test['p_value']:.3f}) "
                f"at a {nw:.0%} blend weight."
            )
        else:
            report.verdicts.append(
                f"News does not measurably help: Sharpe difference {test['difference']:+.2f} "
                f"(95% CI [{test['ci_low']:.2f}, {test['ci_high']:.2f}], p={test['p_value']:.3f}). "
                "Treat the news block as unproven on this data."
            )

    benchmark = results.get("benchmark")
    if (
        primary is not None
        and benchmark is not None
        and primary.metrics.sharpe <= benchmark.metrics.sharpe
    ):
        report.verdicts.append(
            f"The strategy (Sharpe {primary.metrics.sharpe:.2f}) does not beat simply "
            f"holding the universe (Sharpe {benchmark.metrics.sharpe:.2f})."
        )


def cost_sensitivity(
    panel: pd.DataFrame,
    bars: pd.DataFrame,
    calendar: TradingCalendar,
    cfg: Config,
    *,
    scales: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0),
    grid: list[WeeklyDecision] | None = None,
    eligibility: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Re-run the primary variant at several cost multiples.

    A strategy whose Sharpe collapses between 1x and 2x costs is not robust: real costs
    are uncertain by at least that much, and the difference between a good and a bad
    execution week is larger than most people assume.
    """
    from .costs import CostModel

    price_cols = price_feature_columns(panel)
    news_cols = news_feature_columns(panel)
    columns = price_cols + news_cols

    splitter = PurgedWalkForward(
        train_weeks=cfg.cv.train_weeks,
        test_weeks=cfg.cv.test_weeks,
        embargo_weeks=cfg.cv.embargo_weeks,
        expanding=cfg.cv.expanding,
        min_train_weeks=cfg.cv.min_train_weeks,
    )

    if news_cols:
        pw, nw = cfg.model.blend.normalised
        factory = lambda: BlendedModel(  # noqa: E731
            build_model(cfg.model.price_model, cfg, seed=cfg.run.seed),
            build_model(cfg.model.news_model, cfg, seed=cfg.run.seed),
            price_columns=price_cols,
            news_columns=news_cols,
            price_weight=pw,
            news_weight=nw,
        )
    else:
        factory = lambda: build_model(cfg.model.price_model, cfg, seed=cfg.run.seed)  # noqa: E731

    predictions = walk_forward_predict(
        panel, factory, feature_columns=columns, splitter=splitter
    )
    if predictions.empty:
        return pd.DataFrame()

    rows = []
    for scale in scales:
        engine = BacktestEngine(cfg, cost_model=CostModel.from_config(cfg, scale=scale))
        result = engine.run(
            predictions, bars, calendar, name=f"cost_{scale}x", grid=grid,
            eligibility=eligibility,
        )
        rows.append(
            {
                "cost_scale": scale,
                "sharpe": round(result.metrics.sharpe, 3),
                "ann_return": round(result.metrics.ann_return, 4),
                "max_dd": round(result.metrics.max_drawdown, 4),
                "cost_bps_per_week": round(result.metrics.cost_drag_bps, 2),
            }
        )
    return pd.DataFrame(rows)

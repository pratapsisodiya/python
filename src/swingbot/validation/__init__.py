"""Validation: purged cross-validation, metrics, information coefficient, deflation."""

from .deflated import (
    DeflatedSharpeResult,
    deflated_sharpe,
    expected_max_sharpe,
    probability_of_backtest_overfitting,
)
from .ic import (
    ICSummary,
    fold_clustered_ic,
    ic_by_group,
    periodic_ic,
    quantile_returns,
    summarize_ic,
)
from .metrics import (
    PerformanceMetrics,
    compute_metrics,
    drawdown_series,
    drawdown_stats,
    sharpe_confidence_interval,
    sharpe_difference_test,
    turnover_from_weights,
)
from .splits import CombinatorialPurgedCV, Fold, PurgedWalkForward, verify_no_overlap
from .trials import Trial, TrialLedger

__all__ = [
    "CombinatorialPurgedCV",
    "DeflatedSharpeResult",
    "Fold",
    "ICSummary",
    "PerformanceMetrics",
    "PurgedWalkForward",
    "Trial",
    "TrialLedger",
    "compute_metrics",
    "deflated_sharpe",
    "drawdown_series",
    "drawdown_stats",
    "expected_max_sharpe",
    "fold_clustered_ic",
    "ic_by_group",
    "periodic_ic",
    "probability_of_backtest_overfitting",
    "quantile_returns",
    "sharpe_confidence_interval",
    "sharpe_difference_test",
    "summarize_ic",
    "turnover_from_weights",
    "verify_no_overlap",
]

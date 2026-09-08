"""Backtesting: the weekly engine, costs, and the ablation matrix."""

from .ablation import AblationReport, cost_sensitivity, run_ablation
from .costs import CostBreakdown, CostModel, round_trip_bps
from .engine import BacktestEngine, walk_forward_predict
from .overfit import Candidate, PBOResult, default_candidates, run_pbo
from .results import BacktestResult

__all__ = [
    "AblationReport",
    "BacktestEngine",
    "BacktestResult",
    "CostBreakdown",
    "CostModel",
    "Candidate",
    "PBOResult",
    "cost_sensitivity",
    "default_candidates",
    "run_pbo",
    "round_trip_bps",
    "run_ablation",
    "walk_forward_predict",
]

"""Backtest result container."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd

from ..validation.ic import ICSummary, summarize_ic
from ..validation.metrics import PerformanceMetrics, compute_metrics, drawdown_series


@dataclass(slots=True)
class BacktestResult:
    """Everything a backtest produced, plus the caveats that qualify it."""

    name: str = "strategy"
    returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    gross_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    costs: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    turnover: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    gross_exposure: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    net_exposure: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    risk_scale: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    weights: dict[date, dict[str, float]] = field(default_factory=dict)
    predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    cost_breakdown: dict[str, float] = field(default_factory=dict)
    metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    #: Signal quality, independent of costs and sizing. The honest leak diagnostic.
    ic: ICSummary = field(default_factory=lambda: ICSummary())
    #: Statements that qualify the numbers, for example survivorship bias.
    caveats: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def equity(self) -> pd.Series:
        if self.returns.empty:
            return self.returns
        return (1.0 + self.returns).cumprod()

    @property
    def drawdown(self) -> pd.Series:
        return drawdown_series(self.equity)

    def finalize(self) -> BacktestResult:
        self.metrics = compute_metrics(
            self.returns,
            turnover=self.turnover,
            costs=self.costs,
            gross=self.gross_exposure,
            net=self.net_exposure,
        )
        if not self.predictions.empty and "prediction" in self.predictions.columns:
            self.ic = summarize_ic(self.predictions)
        return self

    def summary_row(self) -> dict[str, Any]:
        m = self.metrics
        return {
            "variant": self.name,
            "sharpe": round(m.sharpe, 3),
            "sharpe_ci": f"[{m.sharpe_ci_low:.2f}, {m.sharpe_ci_high:.2f}]",
            "ann_return": round(m.ann_return, 4),
            "ann_vol": round(m.ann_vol, 4),
            "max_dd": round(m.max_drawdown, 4),
            "calmar": round(m.calmar, 3),
            "hit_rate": round(m.hit_rate, 3),
            "turnover": round(m.turnover, 3),
            "cost_bps": round(m.cost_drag_bps, 2),
            "ic": round(self.ic.mean_ic, 4),
            "ic_t": round(self.ic.t_stat, 2),
            "n_weeks": m.n_periods,
        }

    def to_frame(self) -> pd.DataFrame:
        """Per-week detail, for the report and for CSV export."""
        if self.returns.empty:
            return pd.DataFrame()
        return pd.DataFrame(
            {
                "net_return": self.returns,
                "gross_return": self.gross_returns.reindex(self.returns.index),
                "cost": self.costs.reindex(self.returns.index),
                "turnover": self.turnover.reindex(self.returns.index),
                "gross_exposure": self.gross_exposure.reindex(self.returns.index),
                "net_exposure": self.net_exposure.reindex(self.returns.index),
                "risk_scale": self.risk_scale.reindex(self.returns.index),
                "equity": self.equity,
                "drawdown": self.drawdown,
            }
        )

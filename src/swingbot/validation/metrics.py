"""Performance metrics for a weekly strategy.

Everything here annualises with 52, not 252, because the return series is weekly. Using
the daily factor on weekly data inflates Sharpe by a factor of about 2.2, which is a
surprisingly common error and always in the flattering direction.

The confidence interval matters as much as the point estimate. Ten years of weekly data
is about 520 observations, and the standard error on a Sharpe ratio from 520 points is
roughly 0.14 even under ideal assumptions. A reported Sharpe of 0.8 with a 95 percent
interval spanning 0.5 to 1.1 is a very different claim from a bare 0.8.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 52.0


@dataclass(slots=True)
class PerformanceMetrics:
    n_periods: int = 0
    total_return: float = 0.0
    cagr: float = 0.0
    ann_return: float = 0.0
    ann_vol: float = 0.0
    sharpe: float = 0.0
    sharpe_ci_low: float = 0.0
    sharpe_ci_high: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_weeks: int = 0
    hit_rate: float = 0.0
    win_loss_ratio: float = 0.0
    best_week: float = 0.0
    worst_week: float = 0.0
    skew: float = 0.0
    turnover: float = 0.0
    cost_drag_bps: float = 0.0
    avg_gross: float = 0.0
    avg_net: float = 0.0
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def compute_metrics(
    returns: pd.Series,
    *,
    turnover: pd.Series | None = None,
    costs: pd.Series | None = None,
    gross: pd.Series | None = None,
    net: pd.Series | None = None,
    periods_per_year: float = PERIODS_PER_YEAR,
    bootstrap: int = 500,
    seed: int = 7,
) -> PerformanceMetrics:
    """Full metric set from a series of periodic (weekly) simple returns."""
    clean = pd.Series(returns).dropna().astype(float)
    metrics = PerformanceMetrics(n_periods=len(clean))
    if clean.empty:
        return metrics

    equity = (1.0 + clean).cumprod()
    years = len(clean) / periods_per_year

    metrics.total_return = float(equity.iloc[-1] - 1.0)
    metrics.cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else 0.0
    metrics.ann_return = float(clean.mean() * periods_per_year)
    metrics.ann_vol = float(clean.std(ddof=1) * np.sqrt(periods_per_year))
    metrics.sharpe = (
        float(metrics.ann_return / metrics.ann_vol) if metrics.ann_vol > 1e-12 else 0.0
    )

    low, high = sharpe_confidence_interval(clean, periods_per_year, n=bootstrap, seed=seed)
    metrics.sharpe_ci_low, metrics.sharpe_ci_high = low, high

    downside = clean[clean < 0.0]
    downside_vol = float(downside.std(ddof=1) * np.sqrt(periods_per_year)) if len(downside) > 1 else 0.0
    metrics.sortino = (
        float(metrics.ann_return / downside_vol) if downside_vol > 1e-12 else 0.0
    )

    dd, dd_weeks = drawdown_stats(equity)
    metrics.max_drawdown = dd
    metrics.max_drawdown_weeks = dd_weeks
    metrics.calmar = float(metrics.cagr / abs(dd)) if abs(dd) > 1e-9 else 0.0

    wins = clean[clean > 0.0]
    losses = clean[clean < 0.0]
    metrics.hit_rate = float(len(wins) / len(clean))
    metrics.win_loss_ratio = (
        float(wins.mean() / abs(losses.mean())) if len(losses) and losses.mean() != 0 else 0.0
    )
    metrics.best_week = float(clean.max())
    metrics.worst_week = float(clean.min())
    metrics.skew = float(clean.skew()) if len(clean) > 3 else 0.0

    if turnover is not None and len(turnover):
        metrics.turnover = float(pd.Series(turnover).dropna().mean())
    if costs is not None and len(costs):
        metrics.cost_drag_bps = float(pd.Series(costs).dropna().mean() * 10_000.0)
    if gross is not None and len(gross):
        metrics.avg_gross = float(pd.Series(gross).dropna().mean())
    if net is not None and len(net):
        metrics.avg_net = float(pd.Series(net).dropna().mean())

    return metrics


def drawdown_stats(equity: pd.Series) -> tuple[float, int]:
    """Maximum drawdown and the length of the longest underwater stretch."""
    if equity.empty:
        return 0.0, 0
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_dd = float(drawdown.min())

    underwater = drawdown < -1e-9
    longest = current = 0
    for flag in underwater:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return max_dd, int(longest)


def drawdown_series(equity: pd.Series) -> pd.Series:
    if equity.empty:
        return equity
    return equity / equity.cummax() - 1.0


def sharpe_confidence_interval(
    returns: pd.Series,
    periods_per_year: float = PERIODS_PER_YEAR,
    *,
    n: int = 500,
    seed: int = 7,
    block: int = 4,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Bootstrap interval for the Sharpe ratio, using overlapping blocks.

    Blocks rather than independent draws because weekly strategy returns are
    autocorrelated through position overlap and volatility clustering. An i.i.d.
    bootstrap would produce an interval that is too narrow, which defeats the purpose of
    computing one.
    """
    clean = pd.Series(returns).dropna().astype(float).to_numpy()
    if len(clean) < 20 or n <= 0:
        return 0.0, 0.0

    rng = np.random.default_rng(seed)
    n_obs = len(clean)
    n_blocks = int(np.ceil(n_obs / block))
    sharpes = np.empty(n, dtype=float)

    for i in range(n):
        starts = rng.integers(0, n_obs, size=n_blocks)
        sample = np.concatenate([_wrap_slice(clean, s, block) for s in starts])[:n_obs]
        std = sample.std(ddof=1)
        sharpes[i] = (
            sample.mean() * periods_per_year / (std * np.sqrt(periods_per_year))
            if std > 1e-12
            else 0.0
        )

    return float(np.quantile(sharpes, alpha / 2.0)), float(
        np.quantile(sharpes, 1.0 - alpha / 2.0)
    )


def _wrap_slice(values: np.ndarray, start: int, length: int) -> np.ndarray:
    """Circular slice, so blocks near the end are not systematically shorter."""
    end = start + length
    if end <= len(values):
        return values[start:end]
    return np.concatenate([values[start:], values[: end - len(values)]])


def sharpe_difference_test(
    returns_a: pd.Series,
    returns_b: pd.Series,
    *,
    periods_per_year: float = PERIODS_PER_YEAR,
    n: int = 1000,
    seed: int = 7,
    block: int = 4,
) -> dict[str, float]:
    """Bootstrap the *difference* in Sharpe between two strategies.

    This is the number that answers "does news actually help", and it is not the same as
    comparing two separately-computed Sharpe ratios. The two return series share most of
    their variance because they trade the same universe in the same weeks, so the
    difference is estimated far more precisely than either level. Resampling the same
    weeks for both preserves that pairing.
    """
    a = pd.Series(returns_a).dropna().astype(float)
    b = pd.Series(returns_b).dropna().astype(float)
    common = a.index.intersection(b.index)
    a, b = a.loc[common].to_numpy(), b.loc[common].to_numpy()

    if len(a) < 20:
        return {"difference": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_value": 1.0}

    def _sharpe(x: np.ndarray) -> float:
        std = x.std(ddof=1)
        return float(x.mean() * np.sqrt(periods_per_year) / std) if std > 1e-12 else 0.0

    observed = _sharpe(a) - _sharpe(b)

    rng = np.random.default_rng(seed)
    n_obs = len(a)
    n_blocks = int(np.ceil(n_obs / block))
    diffs = np.empty(n, dtype=float)
    for i in range(n):
        starts = rng.integers(0, n_obs, size=n_blocks)
        idx = np.concatenate(
            [np.arange(s, s + block) % n_obs for s in starts]
        )[:n_obs]
        diffs[i] = _sharpe(a[idx]) - _sharpe(b[idx])

    centred = diffs - diffs.mean()
    p_value = float((np.abs(centred) >= abs(observed)).mean())

    return {
        "difference": observed,
        "ci_low": float(np.quantile(diffs, 0.025)),
        "ci_high": float(np.quantile(diffs, 0.975)),
        "p_value": p_value,
    }


def turnover_from_weights(
    weights_by_period: list[dict[str, float]],
) -> pd.Series:
    """One-way turnover per rebalance: half the sum of absolute weight changes."""
    out = []
    previous: dict[str, float] = {}
    for weights in weights_by_period:
        tickers = set(previous) | set(weights)
        change = sum(abs(weights.get(t, 0.0) - previous.get(t, 0.0)) for t in tickers)
        out.append(change / 2.0)
        previous = weights
    return pd.Series(out, dtype=float)

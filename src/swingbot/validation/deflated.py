"""Deflated Sharpe ratio and probability of backtest overfitting.

If you try two hundred configurations and report the best one, its Sharpe ratio is not an
estimate of future performance. It is the maximum of two hundred noisy draws, and the
maximum of noise is reliably positive. The deflated Sharpe ratio, from Bailey and López
de Prado, asks the honest question: given that this was the best of ``n`` trials, what is
the probability the true Sharpe is above zero?

The input that makes this work is an accurate trial count, which is why
:mod:`swingbot.validation.trials` records every backtest automatically rather than
trusting anyone to remember how many they ran.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats

EULER_MASCHERONI = 0.5772156649015329


@dataclass(slots=True)
class DeflatedSharpeResult:
    observed_sharpe: float = 0.0
    expected_max_sharpe: float = 0.0
    deflated_sharpe: float = 0.0
    probability: float = 0.0
    n_trials: int = 1
    n_observations: int = 0
    skew: float = 0.0
    kurtosis: float = 3.0
    is_significant: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    def verdict(self) -> str:
        if self.n_observations < 100:
            return "too short a track record to deflate meaningfully"
        if self.probability >= 0.95:
            return "survives deflation for the number of trials run"
        if self.probability >= 0.80:
            return "marginal once the trial count is accounted for"
        return "does not survive deflation; likely selection, not skill"


def expected_max_sharpe(n_trials: int, variance_of_trials: float = 1.0) -> float:
    """Expected maximum Sharpe across ``n_trials`` independent null strategies.

    This is the bar a genuinely worthless strategy clears by luck alone. With 100 trials
    on a decade of weekly data it is well above zero, which is why an unadjusted Sharpe
    of 0.6 from a wide search means very little.
    """
    if n_trials <= 1:
        return 0.0
    sigma = np.sqrt(max(variance_of_trials, 1e-12))
    n = float(n_trials)
    # Bailey and López de Prado's approximation to the expected maximum of n normals.
    term = (1.0 - EULER_MASCHERONI) * stats.norm.ppf(1.0 - 1.0 / n) + (
        EULER_MASCHERONI * stats.norm.ppf(1.0 - 1.0 / (n * np.e))
    )
    return float(sigma * term)


def deflated_sharpe(
    returns: pd.Series,
    *,
    n_trials: int = 1,
    variance_of_trials: float = 1.0,
    periods_per_year: float = 52.0,
    benchmark_sharpe: float | None = None,
) -> DeflatedSharpeResult:
    """Probability the true Sharpe exceeds the selection-adjusted threshold."""
    clean = pd.Series(returns).dropna().astype(float)
    result = DeflatedSharpeResult(n_trials=max(1, n_trials), n_observations=len(clean))
    if len(clean) < 20:
        return result

    std = clean.std(ddof=1)
    if std <= 1e-12:
        return result

    # Per-period Sharpe; annualisation is applied only for reporting.
    sr = float(clean.mean() / std)
    result.observed_sharpe = float(sr * np.sqrt(periods_per_year))
    result.skew = float(clean.skew()) if len(clean) > 3 else 0.0
    result.kurtosis = float(clean.kurt() + 3.0) if len(clean) > 4 else 3.0

    threshold = (
        benchmark_sharpe / np.sqrt(periods_per_year)
        if benchmark_sharpe is not None
        else expected_max_sharpe(result.n_trials, variance_of_trials) / np.sqrt(periods_per_year)
    )
    result.expected_max_sharpe = float(threshold * np.sqrt(periods_per_year))

    n = len(clean)
    # Non-normality correction: negative skew and fat tails inflate a naive Sharpe.
    denominator = np.sqrt(
        max(1.0 - result.skew * sr + ((result.kurtosis - 1.0) / 4.0) * sr**2, 1e-9)
    )
    z = (sr - threshold) * np.sqrt(n - 1) / denominator

    result.probability = float(stats.norm.cdf(z))
    result.deflated_sharpe = float(z)
    result.is_significant = result.probability >= 0.95
    return result


def probability_of_backtest_overfitting(
    in_sample: np.ndarray, out_of_sample: np.ndarray
) -> float:
    """PBO: how often the in-sample best underperforms the median out of sample.

    Takes paired arrays of performance across combinatorial CV paths. A PBO above 0.5
    means the selection procedure is worse than picking at random, which happens more
    often than people expect when many configurations are compared on one dataset.
    """
    in_sample = np.asarray(in_sample, dtype=float)
    out_of_sample = np.asarray(out_of_sample, dtype=float)
    if in_sample.ndim != 2 or in_sample.shape != out_of_sample.shape:
        raise ValueError("in_sample and out_of_sample must be matching (paths, configs)")

    n_paths = in_sample.shape[0]
    if n_paths == 0:
        return 0.0

    failures = 0
    for path in range(n_paths):
        best = int(np.nanargmax(in_sample[path]))
        oos = out_of_sample[path]
        median = np.nanmedian(oos)
        if oos[best] < median:
            failures += 1
    return float(failures / n_paths)


def haircut_sharpe(observed_sharpe: float, n_trials: int, n_observations: int) -> float:
    """The Sharpe that survives after correcting for multiple testing.

    A blunt, quick companion to the full deflation: subtract the expected maximum of the
    null distribution from the observed value.
    """
    if n_observations < 20:
        return 0.0
    expected = expected_max_sharpe(n_trials, 1.0)
    return float(max(0.0, observed_sharpe - expected))

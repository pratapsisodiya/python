"""Position sizing.

The forecast says which names to hold. Sizing says how much, and it is where most of the
realised risk is actually determined. Equal-weighting a book that contains a 15 percent
volatility utility and a 90 percent volatility small cap means the small cap supplies
most of the portfolio's risk regardless of what the model thinks of it.

Three layers, applied in order:

1. **Inverse volatility** so each name contributes comparable risk.
2. **Portfolio volatility targeting** so the book's total risk is stable across regimes,
   using a shrunk covariance rather than the raw sample one.
3. **A fractional Kelly cap** as a ceiling on the whole thing.

On Kelly: the full Kelly fraction assumes the expected return is known. It is estimated,
noisily, from a few hundred weeks, and Kelly sizing on an overestimated edge is the
fastest route to ruin known to finance. The default here is a quarter, and it is a cap
rather than a target.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_WEEKS = 52.0


def inverse_vol_weights(
    scores: pd.Series,
    volatility: pd.Series,
    *,
    floor_quantile: float = 0.10,
) -> pd.Series:
    """Weights proportional to signal sign over volatility.

    The volatility floor matters more than it looks. A name whose trailing volatility is
    near zero, usually because it barely traded, would otherwise attract an enormous
    weight, and a division by a near-zero estimate is how a sizing routine puts the whole
    book into an illiquid name.
    """
    if scores.empty:
        return scores

    vol = volatility.reindex(scores.index).astype(float)
    positive = vol[vol > 0]
    if positive.empty:
        return pd.Series(np.sign(scores), index=scores.index, dtype=float)

    floor = float(positive.quantile(floor_quantile))
    vol = vol.fillna(positive.median()).clip(lower=max(floor, 1e-4))

    raw = np.sign(scores) / vol
    total = raw.abs().sum()
    return raw / total if total > 0 else raw


def equal_weights(scores: pd.Series) -> pd.Series:
    if scores.empty:
        return scores
    signs = np.sign(scores)
    total = np.abs(signs).sum()
    return pd.Series(signs / total if total > 0 else signs, index=scores.index, dtype=float)


def shrunk_covariance(returns: pd.DataFrame, *, shrinkage: float | None = None) -> pd.DataFrame:
    """Ledoit-Wolf style shrinkage toward a constant-correlation target.

    With 104 weekly observations and 15 names the sample covariance is already poorly
    conditioned, and inverting it for a volatility target produces weights that are
    mostly estimation error. Shrinking toward a structured target is not a refinement
    here, it is what makes the number usable.
    """
    clean = returns.dropna(axis=1, how="all").dropna()
    if clean.shape[0] < 10 or clean.shape[1] < 2:
        return pd.DataFrame()

    sample = clean.cov()
    n_obs, n_assets = clean.shape

    std = np.sqrt(np.diag(sample))
    std_outer = np.outer(std, std)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(std_outer > 0, sample.to_numpy() / std_outer, 0.0)
    off_diagonal = corr[~np.eye(n_assets, dtype=bool)]
    mean_corr = float(np.nanmean(off_diagonal)) if off_diagonal.size else 0.0

    target = mean_corr * std_outer
    np.fill_diagonal(target, np.diag(sample))

    if shrinkage is None:
        # More shrinkage when observations are scarce relative to the number of names.
        shrinkage = float(np.clip(n_assets / max(n_obs, 1), 0.1, 0.9))

    blended = shrinkage * target + (1.0 - shrinkage) * sample.to_numpy()
    return pd.DataFrame(blended, index=sample.index, columns=sample.columns)


def portfolio_volatility(weights: pd.Series, covariance: pd.DataFrame) -> float:
    """Annualised ex-ante volatility of a weight vector."""
    if weights.empty or covariance.empty:
        return 0.0
    common = weights.index.intersection(covariance.index)
    if len(common) < 2:
        return 0.0
    w = weights.loc[common].to_numpy(dtype=float)
    cov = covariance.loc[common, common].to_numpy(dtype=float)
    variance = float(w @ cov @ w)
    return float(np.sqrt(max(variance, 0.0)) * np.sqrt(TRADING_WEEKS))


def scale_to_target_vol(
    weights: pd.Series,
    covariance: pd.DataFrame,
    *,
    target_vol: float = 0.12,
    max_scale: float = 3.0,
    fallback_vol: pd.Series | None = None,
) -> tuple[pd.Series, float]:
    """Scale a weight vector so its ex-ante volatility hits the target.

    Returns the scaled weights and the scale applied, so the caller can report leverage
    rather than discovering it in the equity curve.
    """
    if weights.empty:
        return weights, 1.0

    current = portfolio_volatility(weights, covariance)

    if current <= 1e-6 and fallback_vol is not None:
        # No usable covariance: assume the names are uncorrelated, which understates
        # risk, so cap the resulting scale conservatively.
        vol = fallback_vol.reindex(weights.index).fillna(fallback_vol.median())
        current = float(np.sqrt(((weights * vol) ** 2).sum()))

    if current <= 1e-6:
        return weights, 1.0

    scale = float(np.clip(target_vol / current, 1.0 / max_scale, max_scale))
    return weights * scale, scale


def kelly_cap(
    weights: pd.Series,
    expected_returns: pd.Series,
    volatility: pd.Series,
    *,
    fraction: float = 0.25,
) -> pd.Series:
    """Cap each weight at a fraction of its Kelly-optimal size.

    Kelly for a single asset is ``mu / sigma^2``. Both inputs are estimates, ``mu``
    especially, so this is applied as a ceiling on weights that were sized some other
    way, never as the sizing rule itself.

    ``expected_returns`` must be an actual expected return over the hold window, on the
    same scale as ``volatility``. Handing it a cross-sectional rank instead does not
    fail — it just produces a ceiling one or two orders of magnitude too high to ever
    bind, which is what this function did until :mod:`swingbot.model.calibrate` existed
    to supply the real quantity. See ``tests/test_portfolio_limits.py``.
    """
    if weights.empty or fraction <= 0:
        return weights

    mu = expected_returns.reindex(weights.index).astype(float)
    sigma = volatility.reindex(weights.index).astype(float)
    sigma = sigma.fillna(sigma.median()).clip(lower=1e-3)

    kelly = (mu / (sigma**2)).abs() * fraction
    # A name with no expected return gets no ceiling, rather than a ceiling of zero.
    # This previously read ``.fillna(0.0)`` on ``mu``, which turns a missing estimate into
    # a hard instruction to hold nothing — deleting the position instead of declining to
    # cap it. Missing information is not evidence of zero edge.
    ceiling = kelly.clip(upper=1.0).fillna(np.inf)
    capped = np.sign(weights) * np.minimum(weights.abs(), ceiling)
    return capped


def normalize_gross(weights: pd.Series, gross: float = 1.0) -> pd.Series:
    """Rescale so total absolute weight equals ``gross``."""
    if weights.empty:
        return weights
    total = weights.abs().sum()
    return weights * (gross / total) if total > 1e-12 else weights

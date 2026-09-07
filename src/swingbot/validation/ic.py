"""Information coefficient: signal quality, independent of the portfolio.

The IC is the rank correlation between the forecast and the realised return, computed
within each date and then averaged. It is the cleanest measure of whether a signal has
any content, because it is unaffected by position sizing, leverage or risk limits, all of
which can flatter or wreck a Sharpe ratio without changing the underlying prediction.

The critical detail is the standard error. Four hundred names in one week are not four
hundred independent observations: they share the market factor, sector factors, and the
same news flow. The effective sample size is the number of **weeks**, roughly 520 for a
decade. Computing a t-statistic on the pooled name-weeks instead gives a number about
twenty times too large, and this is the specific mistake behind a great many published
retail backtests that look overwhelmingly significant and then do not work.

So every t-statistic here is clustered by week.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats

from ..types import DECISION_SESSION, LABEL


@dataclass(slots=True)
class ICSummary:
    n_periods: int = 0
    mean_ic: float = 0.0
    median_ic: float = 0.0
    ic_std: float = 0.0
    ic_ir: float = 0.0
    t_stat: float = 0.0
    p_value: float = 1.0
    hit_rate: float = 0.0
    decile_spread: float = 0.0
    top_decile_return: float = 0.0
    bottom_decile_return: float = 0.0
    monotonicity: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    def verdict(self) -> str:
        """A plain-language read on whether this signal is worth anything."""
        if self.n_periods < 52:
            return "too few weeks to judge"
        if self.p_value > 0.10:
            return "indistinguishable from noise"
        if abs(self.mean_ic) < 0.01:
            return "statistically detectable but economically tiny"
        if abs(self.mean_ic) < 0.03:
            return "plausible weekly-horizon signal"
        return "unusually strong for this horizon; check for leakage"


def periodic_ic(
    panel: pd.DataFrame,
    prediction_column: str = "prediction",
    label_column: str = LABEL,
    *,
    date_column: str = DECISION_SESSION,
    method: str = "spearman",
    min_names: int = 10,
) -> pd.Series:
    """Rank correlation between prediction and label, one value per date."""
    if panel.empty or prediction_column not in panel.columns:
        return pd.Series(dtype=float)

    frame = panel[[date_column, prediction_column, label_column]].dropna()
    if frame.empty:
        return pd.Series(dtype=float)

    def _corr(group: pd.DataFrame) -> float:
        if len(group) < min_names:
            return np.nan
        pred = group[prediction_column]
        label = group[label_column]
        if pred.nunique() < 2 or label.nunique() < 2:
            return np.nan
        return float(pred.corr(label, method=method))

    return frame.groupby(date_column, sort=True)[
        [prediction_column, label_column]
    ].apply(_corr).dropna()


def summarize_ic(
    panel: pd.DataFrame,
    prediction_column: str = "prediction",
    label_column: str = LABEL,
    *,
    date_column: str = DECISION_SESSION,
    n_quantiles: int = 10,
) -> ICSummary:
    """Full IC summary with week-clustered significance."""
    ic = periodic_ic(
        panel, prediction_column, label_column, date_column=date_column
    )
    summary = ICSummary(n_periods=len(ic))
    if ic.empty:
        return summary

    summary.mean_ic = float(ic.mean())
    summary.median_ic = float(ic.median())
    summary.ic_std = float(ic.std(ddof=1)) if len(ic) > 1 else 0.0
    summary.ic_ir = float(summary.mean_ic / summary.ic_std) if summary.ic_std > 1e-12 else 0.0
    summary.hit_rate = float((ic > 0).mean())

    # Clustered by week: the sample size is the number of periods, not name-weeks.
    if len(ic) > 2 and summary.ic_std > 1e-12:
        summary.t_stat = float(summary.mean_ic / (summary.ic_std / np.sqrt(len(ic))))
        summary.p_value = float(2.0 * (1.0 - stats.t.cdf(abs(summary.t_stat), df=len(ic) - 1)))

    deciles = quantile_returns(
        panel, prediction_column, label_column, date_column=date_column, n_quantiles=n_quantiles
    )
    if not deciles.empty:
        summary.top_decile_return = float(deciles.iloc[-1])
        summary.bottom_decile_return = float(deciles.iloc[0])
        summary.decile_spread = summary.top_decile_return - summary.bottom_decile_return
        summary.monotonicity = _monotonicity(deciles)

    return summary


def quantile_returns(
    panel: pd.DataFrame,
    prediction_column: str = "prediction",
    label_column: str = LABEL,
    *,
    date_column: str = DECISION_SESSION,
    n_quantiles: int = 10,
) -> pd.Series:
    """Mean realised label by prediction quantile, averaged across dates.

    Bucketing within each date rather than pooling keeps the buckets comparable when the
    overall level of returns shifts between calm and volatile periods.
    """
    if panel.empty or prediction_column not in panel.columns:
        return pd.Series(dtype=float)

    frame = panel[[date_column, prediction_column, label_column]].dropna().copy()
    if frame.empty:
        return pd.Series(dtype=float)

    def _bucket(group: pd.DataFrame) -> pd.Series:
        if len(group) < n_quantiles:
            return pd.Series(dtype=float)
        try:
            buckets = pd.qcut(
                group[prediction_column], n_quantiles, labels=False, duplicates="drop"
            )
        except ValueError:
            return pd.Series(dtype=float)
        return group[label_column].groupby(buckets).mean()

    per_date = frame.groupby(date_column, sort=True)[
        [prediction_column, label_column]
    ].apply(_bucket)
    if per_date.empty:
        return pd.Series(dtype=float)
    if isinstance(per_date, pd.DataFrame):
        return per_date.mean(axis=0)
    return per_date.groupby(level=-1).mean()


def _monotonicity(deciles: pd.Series) -> float:
    """Spearman correlation between bucket index and mean return.

    A real signal produces a monotone ladder. A spread driven entirely by one extreme
    bucket, with noise in between, is usually a handful of outliers rather than an
    effect, and this number distinguishes the two.
    """
    if len(deciles) < 3:
        return 0.0
    ranks = np.arange(len(deciles), dtype=float)
    corr = stats.spearmanr(ranks, deciles.to_numpy(dtype=float)).statistic
    return float(corr) if np.isfinite(corr) else 0.0


def fold_clustered_ic(
    panel: pd.DataFrame,
    prediction_column: str = "prediction",
    label_column: str = LABEL,
    *,
    date_column: str = DECISION_SESSION,
    fold_column: str = "fold",
) -> ICSummary:
    """IC summarised with the fold, not the week, as the unit of observation.

    This exists because :func:`summarize_ic` is the wrong test for the shuffled-label
    canary, and the reason is the same mistake one level up that this module exists to
    prevent.

    A walk-forward model is refit once per fold, so within a fold the fitted function is
    *fixed*. Its information coefficient then has the same sign for every week in that
    fold, and those weeks are not independent draws. Clustering by week gives roughly
    twenty times too many effective observations: on real runs the shuffled canary
    produced IC +0.019 at t = 3.4 over 348 weeks, which is t = 1.5 over five folds and
    not significant at all.

    For a genuine signal the week-clustered statistic in :func:`summarize_ic` is the right
    one, because a real edge is expected to persist across refits. For the null-model
    canary the question is specifically "did this particular fitted function get lucky",
    and that is a per-fold question.
    """
    summary = ICSummary()
    if panel.empty or prediction_column not in panel.columns:
        return summary

    if fold_column not in panel.columns:
        # Without folds there is one fitted model, so there is one observation. Fall back
        # to the week-clustered form rather than inventing a fold structure.
        return summarize_ic(
            panel, prediction_column, label_column, date_column=date_column
        )

    per_fold = []
    for _, block in panel.groupby(fold_column, sort=True):
        ic = periodic_ic(block, prediction_column, label_column, date_column=date_column)
        if not ic.empty:
            per_fold.append(float(ic.mean()))

    if not per_fold:
        return summary

    values = np.asarray(per_fold, dtype=float)
    summary.n_periods = len(values)
    summary.mean_ic = float(values.mean())
    summary.median_ic = float(np.median(values))
    summary.hit_rate = float((values > 0).mean())

    if len(values) > 1:
        summary.ic_std = float(values.std(ddof=1))
        if summary.ic_std > 1e-12:
            summary.ic_ir = float(summary.mean_ic / summary.ic_std)
            summary.t_stat = float(
                summary.mean_ic / (summary.ic_std / np.sqrt(len(values)))
            )
            summary.p_value = float(
                2.0 * (1.0 - stats.t.cdf(abs(summary.t_stat), df=len(values) - 1))
            )
    return summary


def ic_by_group(
    panel: pd.DataFrame,
    group_column: str,
    prediction_column: str = "prediction",
    label_column: str = LABEL,
    *,
    date_column: str = DECISION_SESSION,
) -> pd.DataFrame:
    """IC computed separately per group, for example by news event type or sector."""
    if panel.empty or group_column not in panel.columns:
        return pd.DataFrame()

    rows = []
    for value, group in panel.groupby(group_column, sort=True):
        summary = summarize_ic(
            group, prediction_column, label_column, date_column=date_column
        )
        rows.append(
            {
                group_column: value,
                "n_rows": len(group),
                "n_periods": summary.n_periods,
                "mean_ic": summary.mean_ic,
                "ic_ir": summary.ic_ir,
                "t_stat": summary.t_stat,
                "p_value": summary.p_value,
            }
        )
    return pd.DataFrame(rows).sort_values("mean_ic", ascending=False).reset_index(drop=True)

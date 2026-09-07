"""Labels: forward excess return over the exact span the backtest trades.

Three decisions here carry most of the weight.

**Open to open, not close to close.** The label measures the return from the entry
session's open to the exit session's open, which is exactly what the portfolio earns
after being filled at the next open. A close-to-close label would credit the strategy
with the overnight move between the signal and the fill, which nobody could have
captured. This single choice accounts for a large part of the gap between naive
backtests and reality at weekly horizons.

**Excess, not absolute.** The target is the return minus the cross-sectional mean on that
date. That removes the market factor, which this system is not trying to forecast, and
turns the problem into ranking, which is far more stable than predicting levels.

**Every label carries its span.** ``label_t0`` and ``label_t1`` are stored with the row,
because purged cross-validation needs to know which training samples overlap a test
window. Without the span there is no way to purge correctly, and with a hold longer than
the rebalance step the overlap is severe.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from ..calendars import TradingCalendar, WeeklyDecision
from ..types import (
    CLOSE,
    DECISION_SESSION,
    ENTRY_SESSION,
    EXIT_SESSION,
    LABEL,
    LABEL_T0,
    LABEL_T1,
    OPEN,
    SAMPLE_WEIGHT,
    SESSION,
    TICKER,
)


def build_labels(
    bars: pd.DataFrame,
    calendar: TradingCalendar,
    *,
    hold_sessions: int = 5,
    grid: list[WeeklyDecision] | None = None,
    vol_scale: bool = False,
    vol_window: int = 65,
) -> pd.DataFrame:
    """Forward excess log return for every ticker at every decision date.

    Returns one row per ``(decision_session, ticker)`` with the label, its span, and the
    entry and exit sessions the backtest will actually use.
    """
    if bars.empty:
        return pd.DataFrame()

    grid = grid if grid is not None else calendar.weekly_grid(hold_sessions)
    if not grid:
        return pd.DataFrame()

    opens = bars.pivot_table(index=SESSION, columns=TICKER, values=OPEN, aggfunc="last")
    opens = opens.sort_index()

    rows = []
    for decision in grid:
        entry, exit_ = decision.entry_session, decision.exit_session
        if entry not in opens.index or exit_ not in opens.index:
            continue
        entry_px = opens.loc[entry]
        exit_px = opens.loc[exit_]
        valid = entry_px.notna() & exit_px.notna() & (entry_px > 0) & (exit_px > 0)
        if not bool(valid.any()):
            continue
        raw = np.log(exit_px[valid].astype(float) / entry_px[valid].astype(float))
        block = pd.DataFrame(
            {
                TICKER: raw.index.astype(str),
                DECISION_SESSION: decision.decision_session,
                ENTRY_SESSION: entry,
                EXIT_SESSION: exit_,
                "raw_return": raw.to_numpy(dtype=float),
            }
        )
        rows.append(block)

    if not rows:
        return pd.DataFrame()

    labels = pd.concat(rows, ignore_index=True)

    # Cross-sectional excess: subtract the equal-weighted mean of that decision date.
    mean = labels.groupby(DECISION_SESSION, sort=False)["raw_return"].transform("mean")
    labels[LABEL] = labels["raw_return"] - mean

    if vol_scale:
        vol = _trailing_vol(bars, vol_window)
        labels = labels.merge(
            vol.rename("trailing_vol"),
            left_on=[DECISION_SESSION, TICKER],
            right_index=True,
            how="left",
        )
        scale = labels["trailing_vol"].replace(0.0, np.nan) * np.sqrt(len(grid) and 1.0)
        labels[LABEL] = labels[LABEL] / scale
        labels = labels.drop(columns=["trailing_vol"])

    # The span the label occupies, used by purging. Entry to exit, inclusive.
    labels[LABEL_T0] = labels[ENTRY_SESSION]
    labels[LABEL_T1] = labels[EXIT_SESSION]
    labels[SAMPLE_WEIGHT] = 1.0

    return labels.dropna(subset=[LABEL]).reset_index(drop=True)


def _trailing_vol(bars: pd.DataFrame, window: int) -> pd.Series:
    """Trailing realised volatility per ticker and session, shifted to stay causal."""
    frame = bars.sort_values([TICKER, SESSION]).copy()
    log_close = np.log(frame[CLOSE].astype(float).where(lambda s: s > 0))
    frame["_ret"] = log_close - log_close.groupby(frame[TICKER], sort=False).shift(1)
    vol = frame.groupby(TICKER, sort=False)["_ret"].transform(
        lambda s: s.rolling(window, min_periods=window // 2).std().shift(1)
    )
    frame["_vol"] = vol * np.sqrt(252.0)
    return frame.set_index([SESSION, TICKER])["_vol"]


# --------------------------------------------------------------------------------------
# Sample weighting
# --------------------------------------------------------------------------------------


def uniqueness_weights(labels: pd.DataFrame) -> pd.Series:
    """Average uniqueness of each label, following López de Prado.

    When a hold is longer than the rebalance step, consecutive labels cover overlapping
    sessions, so the same market move appears in several training rows. Left alone, the
    model treats that as independent confirmation and becomes overconfident, and any
    significance test built on it is wrong.

    The weight is the reciprocal of the average number of labels concurrently spanning
    each session in a label's window, normalised to mean one.
    """
    if labels.empty:
        return pd.Series(dtype=float)

    starts = pd.to_datetime(labels[LABEL_T0])
    ends = pd.to_datetime(labels[LABEL_T1])

    boundaries = pd.concat([starts, ends]).sort_values().unique()
    timeline = pd.Series(0, index=pd.DatetimeIndex(boundaries), dtype=float)

    # Count concurrency by sweeping the interval endpoints.
    counts = pd.Series(0.0, index=timeline.index)
    for start, end in zip(starts, ends, strict=True):
        counts.loc[start:end] += 1.0

    weights = np.empty(len(labels), dtype=float)
    counts_safe = counts.replace(0.0, np.nan)
    for i, (start, end) in enumerate(zip(starts, ends, strict=True)):
        window = counts_safe.loc[start:end]
        weights[i] = float((1.0 / window).mean()) if len(window) else 1.0

    series = pd.Series(weights, index=labels.index, name=SAMPLE_WEIGHT)
    mean = series.mean()
    return series / mean if mean and np.isfinite(mean) else series.fillna(1.0)


def recency_weights(
    decision_sessions: pd.Series, *, half_life_weeks: float = 104.0
) -> pd.Series:
    """Exponential decay so recent regimes count for more.

    Combined multiplicatively with uniqueness. Kept mild by default, because a short
    half-life throws away the sample size that weekly strategies are already short of.
    """
    if decision_sessions.empty:
        return pd.Series(dtype=float)
    stamps = pd.to_datetime(decision_sessions)
    latest = stamps.max()
    age_weeks = (latest - stamps).dt.days / 7.0
    decay = 0.5 ** (age_weeks / max(half_life_weeks, 1e-6))
    mean = decay.mean()
    return decay / mean if mean else decay


def combine_weights(*weights: pd.Series) -> pd.Series:
    """Multiply weight series and renormalise to mean one."""
    combined: pd.Series | None = None
    for w in weights:
        if w is None or w.empty:
            continue
        combined = w.copy() if combined is None else combined * w
    if combined is None:
        return pd.Series(dtype=float)
    mean = combined.mean()
    return combined / mean if mean and np.isfinite(mean) else combined


def label_spans(labels: pd.DataFrame) -> pd.DataFrame:
    """The ``(t0, t1)`` spans purging needs, indexed like the label frame."""
    return labels[[LABEL_T0, LABEL_T1]].copy()


def forward_return_matrix(
    bars: pd.DataFrame, grid: list[WeeklyDecision]
) -> dict[date, pd.Series]:
    """Realised open-to-open return per decision date, for the backtest.

    The backtest marks positions with this rather than recomputing, which guarantees the
    numbers the model was trained on and the numbers the portfolio earns are the same
    quantity.
    """
    opens = bars.pivot_table(index=SESSION, columns=TICKER, values=OPEN, aggfunc="last")
    opens = opens.sort_index()
    out: dict[date, pd.Series] = {}
    for decision in grid:
        entry, exit_ = decision.entry_session, decision.exit_session
        if entry not in opens.index or exit_ not in opens.index:
            continue
        entry_px = opens.loc[entry]
        exit_px = opens.loc[exit_]
        valid = entry_px.notna() & exit_px.notna() & (entry_px > 0)
        out[decision.decision_session] = (
            exit_px[valid].astype(float) / entry_px[valid].astype(float) - 1.0
        )
    return out

"""Daily bars to weekly bars, anchored on the decision calendar.

Weekly bars are built from the sessions the decision grid actually uses rather than from
a pandas ``W-FRI`` resample. The difference matters: a calendar resample happily emits a
bar for a week the exchange was shut, and it anchors on Friday even when Friday was a
holiday, which shifts every feature by a session in exactly the weeks that tend to be
eventful.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from ..calendars import TradingCalendar
from ..types import (
    AVAILABLE_AT,
    CLOSE,
    DIV_CASH,
    HIGH,
    LOW,
    OPEN,
    SESSION,
    SPLIT_FACTOR,
    TICKER,
    VOLUME,
)


def to_weekly(
    bars: pd.DataFrame,
    calendar: TradingCalendar,
    *,
    decision_sessions: list[date] | None = None,
) -> pd.DataFrame:
    """Aggregate daily bars into weekly bars ending on each decision session.

    The resulting bar for decision session ``D`` covers the sessions from the previous
    decision session (exclusive) through ``D`` (inclusive), so it contains exactly the
    information available at ``D``'s close and no more.
    """
    if bars.empty:
        return pd.DataFrame()

    anchors = decision_sessions or calendar.decision_sessions()
    if not anchors:
        return pd.DataFrame()

    frame = bars.sort_values([TICKER, SESSION]).copy()
    anchor_array = np.array(anchors)
    sessions = frame[SESSION].to_numpy()
    # Each daily session belongs to the first decision session at or after it.
    positions = np.searchsorted(anchor_array, sessions, side="left")
    inside = positions < len(anchor_array)
    frame = frame.loc[inside]
    if frame.empty:
        return pd.DataFrame()
    frame["week_end"] = anchor_array[positions[inside]]

    grouped = frame.groupby([TICKER, "week_end"], sort=True)
    weekly = grouped.agg(
        **{
            OPEN: (OPEN, "first"),
            HIGH: (HIGH, "max"),
            LOW: (LOW, "min"),
            CLOSE: (CLOSE, "last"),
            VOLUME: (VOLUME, "sum"),
            SPLIT_FACTOR: (SPLIT_FACTOR, "prod"),
            DIV_CASH: (DIV_CASH, "sum"),
            AVAILABLE_AT: (AVAILABLE_AT, "max"),
            "n_sessions": (SESSION, "count"),
        }
    ).reset_index()

    weekly = weekly.rename(columns={"week_end": SESSION})
    return weekly.sort_values([TICKER, SESSION]).reset_index(drop=True)


def weekly_returns(weekly: pd.DataFrame) -> pd.DataFrame:
    """Add weekly log return to a weekly bar frame."""
    if weekly.empty:
        return weekly
    out = weekly.sort_values([TICKER, SESSION]).copy()
    log_close = np.log(out[CLOSE].astype(float))
    out["weekly_log_return"] = log_close - log_close.groupby(out[TICKER], sort=False).shift(1)
    return out


def align_panel(frame: pd.DataFrame, sessions: list[date], tickers: list[str]) -> pd.DataFrame:
    """Reindex to a complete session-by-ticker grid, leaving gaps as NaN.

    Gaps are left as NaN rather than forward-filled. Forward-filling a price across a
    halt invents a bar that never traded, and the backtest would happily fill an order
    at it.
    """
    if frame.empty:
        return frame
    grid = pd.MultiIndex.from_product([sessions, tickers], names=[SESSION, TICKER])
    indexed = frame.set_index([SESSION, TICKER])
    return indexed.reindex(grid).reset_index()

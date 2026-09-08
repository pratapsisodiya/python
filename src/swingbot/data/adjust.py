"""Point-in-time corporate-action adjustment.

The subtle bug this module exists to prevent: a vendor's "adjusted close" is recomputed
every time a split happens, so the 2019 price you read today is not the price anyone
could have seen in 2019. Ratio features are invariant to that, but level features are
not, and neither is a price filter. A backtest that reads back-adjusted prices is quietly
using information from the future about which names later split.

So raw prices and factors are stored separately, and :func:`adjust_asof` builds the
adjustment chain from only the factors dated on or before the as-of date. Prices before
the as-of date look exactly as they looked at the time.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from ..types import (
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

PRICE_COLUMNS = (OPEN, HIGH, LOW, CLOSE)


def adjust_asof(
    bars: pd.DataFrame,
    asof: date | None = None,
    *,
    dividends: bool = True,
) -> pd.DataFrame:
    """Return bars adjusted using only corporate actions known at ``asof``.

    The adjustment factor for a session is the product of every split (and dividend
    yield, when enabled) that happens *after* that session and at or before ``asof``.
    Sessions after ``asof`` are left unadjusted, since their own actions are unknown.

    Passing ``asof=None`` adjusts using every action in the frame, which is the standard
    back-adjusted series. That is correct for charting and wrong for backtesting, so
    callers in the feature path always pass a date.
    """
    if bars.empty:
        return bars

    out = bars.sort_values([TICKER, SESSION]).copy()
    split = out[SPLIT_FACTOR].fillna(1.0).to_numpy(dtype=float)
    div = out[DIV_CASH].fillna(0.0).to_numpy(dtype=float)
    close = out[CLOSE].to_numpy(dtype=float)

    if asof is not None:
        sessions = out[SESSION].to_numpy()
        known = np.array([s <= asof for s in sessions], dtype=bool)
        split = np.where(known, split, 1.0)
        div = np.where(known, div, 0.0)

    # Per-session multiplicative factor. A 2-for-1 split has split_factor 2, so prices
    # before it are divided by 2. A dividend reduces the pre-ex price by its yield.
    step = 1.0 / np.where(split > 0, split, 1.0)
    if dividends:
        with np.errstate(divide="ignore", invalid="ignore"):
            yield_ = np.where(close > 0, div / close, 0.0)
        step = step * (1.0 - np.clip(yield_, 0.0, 0.95))

    factors = np.ones(len(out), dtype=float)
    for _, positions in out.groupby(TICKER, sort=False).indices.items():
        idx = np.sort(positions)
        block = step[idx]
        # Cumulative product of every action strictly after each session, computed
        # backwards from the end of the series.
        reversed_cumprod = np.cumprod(block[::-1])[::-1]
        trailing = np.append(reversed_cumprod[1:], 1.0)
        factors[idx] = trailing

    for column in PRICE_COLUMNS:
        if column in out.columns:
            out[column] = out[column].to_numpy(dtype=float) * factors
    if VOLUME in out.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            out[VOLUME] = np.where(factors > 0, out[VOLUME].to_numpy(dtype=float) / factors, 0.0)

    out["adjustment_factor"] = factors
    return out.reset_index(drop=True)


def total_return_series(bars: pd.DataFrame) -> pd.Series:
    """Per-session total return including dividends, from raw prices.

    Used by the backtest to mark positions. Working from raw prices plus the dividend
    on the day keeps the return series independent of any adjustment convention.
    """
    if bars.empty:
        return pd.Series(dtype=float)
    frame = bars.sort_values([TICKER, SESSION])
    close = frame[CLOSE].astype(float)
    prev_close = close.groupby(frame[TICKER], sort=False).shift(1)
    split = frame[SPLIT_FACTOR].fillna(1.0).astype(float)
    div = frame[DIV_CASH].fillna(0.0).astype(float)
    # A split multiplies the share count, so the comparable prior close is scaled.
    adjusted_prev = prev_close / split.where(split > 0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = (close + div) / adjusted_prev - 1.0
    return pd.Series(ret.to_numpy(), index=frame.index, name="total_return")


def apply_delisting_returns(
    equity_curve: pd.Series,
    delistings: dict[str, date],
    weights_by_session: dict[date, dict[str, float]],
    delisting_return: float,
) -> pd.Series:
    """Book a terminal loss on positions held into a delisting.

    Without this, a backtest silently drops failed names on the session they vanish,
    which turns every bankruptcy into a costless exit and every short into free money.
    """
    if equity_curve.empty or not delistings:
        return equity_curve
    out = equity_curve.copy()
    by_session: dict[date, float] = {}
    for ticker, session in delistings.items():
        weights = weights_by_session.get(session, {})
        weight = weights.get(ticker)
        if weight:
            by_session[session] = by_session.get(session, 0.0) + weight * delisting_return
    for session, impact in by_session.items():
        if session in out.index:
            out.loc[session:] = out.loc[session:] * (1.0 + impact)
    return out

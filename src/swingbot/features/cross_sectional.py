"""Cross-sectional normalisation.

Raw features are not comparable across time. A 5 percent monthly momentum means something
different in a calm market than in a crisis, and a model trained on raw values spends its
capacity learning the level of volatility rather than which name is attractive relative to
its peers.

So features are ranked and z-scored **within each decision date**. That has a second
benefit that matters more than the first: a within-date transform is causal by
construction. It uses only rows from that date, so it cannot leak across time no matter
how it is applied. This is the one place where the natural implementation is also the
safe one, and it is worth stating because the tempting alternative, fitting a
``StandardScaler`` on the whole panel, is the single most common leak in retail
backtests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..types import SECTOR, SESSION, TICKER


def cross_sectional_rank(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    date_column: str = SESSION,
    min_names: int = 20,
) -> pd.DataFrame:
    """Rank each column within each date, mapped to ``[-0.5, 0.5]``.

    Ranks rather than raw values because they are robust to the fat tails that make a
    z-score of a single blown-up name dominate a whole cross-section.
    """
    out = frame.copy()
    grouped = out.groupby(date_column, sort=False)
    sizes = grouped[columns[0]].transform("size") if columns else None

    for column in columns:
        ranked = grouped[column].rank(pct=True, method="average") - 0.5
        if sizes is not None:
            ranked = ranked.where(sizes >= min_names)
        out[column] = ranked
    return out


def cross_sectional_zscore(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    date_column: str = SESSION,
    winsorize: float = 0.02,
    min_names: int = 20,
) -> pd.DataFrame:
    """Winsorise then z-score each column within each date."""
    out = frame.copy()
    for column in columns:
        grouped = out.groupby(date_column, sort=False)[column]
        values = out[column]
        if winsorize > 0:
            lower = grouped.transform(lambda s: s.quantile(winsorize))
            upper = grouped.transform(lambda s: s.quantile(1.0 - winsorize))
            values = values.clip(lower=lower, upper=upper)
        mean = values.groupby(out[date_column], sort=False).transform("mean")
        std = values.groupby(out[date_column], sort=False).transform("std")
        count = values.groupby(out[date_column], sort=False).transform("count")
        z = (values - mean) / std.replace(0.0, np.nan)
        out[column] = z.where(count >= min_names)
    return out


def sector_neutralize(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    date_column: str = SESSION,
    sector_column: str = SECTOR,
    min_per_sector: int = 3,
) -> pd.DataFrame:
    """Subtract the date-and-sector mean from each feature.

    Turns "this stock looks strong" into "this stock looks strong for a bank", which is
    what a market-neutral book actually wants. Sectors with too few names on a date are
    left alone, because demeaning a group of two just sets both to plus and minus the
    same number and manufactures a signal from nothing.
    """
    if sector_column not in frame.columns:
        return frame
    out = frame.copy()
    keys = [date_column, sector_column]
    for column in columns:
        grouped = out.groupby(keys, sort=False)[column]
        mean = grouped.transform("mean")
        count = grouped.transform("count")
        out[column] = np.where(count >= min_per_sector, out[column] - mean, out[column])
    return out


def cross_sectional_demean(
    frame: pd.DataFrame, column: str, *, date_column: str = SESSION
) -> pd.Series:
    """Subtract the equal-weighted date mean. Used to build excess-return labels."""
    mean = frame.groupby(date_column, sort=False)[column].transform("mean")
    return frame[column] - mean


def drop_thin_dates(
    frame: pd.DataFrame, *, date_column: str = SESSION, min_names: int = 20
) -> pd.DataFrame:
    """Remove dates with too few names to rank meaningfully.

    A cross-sectional model on eight names is not measuring the same thing as one on
    four hundred, and mixing them makes both the training signal and the reported
    information coefficient unreliable.
    """
    counts = frame.groupby(date_column, sort=False)[TICKER].transform("size")
    return frame.loc[counts >= min_names]

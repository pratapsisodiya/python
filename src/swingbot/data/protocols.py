"""Data-layer protocols.

Two seams. ``PriceProvider`` is where market data comes from, and ``UniverseProvider`` is
where the point-in-time list of tradeable names comes from. Everything downstream is
written against these, so swapping Yahoo for a paid vendor is a one-file change.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class PriceProvider(Protocol):
    """Source of daily bars.

    Implementations return a frame with the columns in :data:`swingbot.types.BAR_COLUMNS`:
    ``ticker, session, open, high, low, close, volume, split_factor, div_cash,
    available_at, is_delisted``.

    Prices are **raw**, not back-adjusted. Split and dividend factors are carried
    alongside so :mod:`swingbot.data.adjust` can build the adjustment chain using only
    the factors known at a given date. A vendor's pre-adjusted close is retroactively
    restated every time a split happens, which quietly rewrites history under a backtest.
    """

    name: str

    def daily_bars(
        self, tickers: Sequence[str], start: date, end: date
    ) -> pd.DataFrame: ...

    def available(self) -> bool:
        """Whether this provider can run right now, without raising."""
        ...


@runtime_checkable
class UniverseProvider(Protocol):
    """Source of point-in-time universe membership.

    There is deliberately no ``current_tickers()`` method. Asking for "the tickers"
    without a date is how survivorship bias gets into a backtest, so the API does not
    offer it.
    """

    name: str

    def members_asof(self, asof: date) -> pd.DataFrame:
        """Names that were members at ``asof``, including ones later delisted.

        Returns columns ``ticker, sector`` at minimum.
        """
        ...

    def delisting_return(self, ticker: str) -> float | None:
        """Terminal return booked when a name leaves the universe involuntarily."""
        ...

    def lot_sizes(self) -> dict[str, int]:
        """Tradeable increment per symbol, for symbols that declare one.

        A symbol absent from the mapping has an *unknown* increment, which is not the
        same as an increment of 1. Cash equity is fine at 1; a single-stock future is
        not, and the order file marks it so nobody sends an unplaceable quantity.
        """
        ...

    def is_survivorship_biased(self) -> bool:
        """True when membership history is unavailable and results are inflated."""
        ...

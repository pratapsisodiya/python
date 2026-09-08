"""Stooq price provider.

A second free source, useful as a cross-check against Yahoo rather than as a primary.
Stooq serves plain CSV and covers US and Indian symbols, though it sits behind a
JavaScript challenge from some IP ranges, in which case the chain simply moves on.

Cross-checking two independent sources is worth the small effort: a single bad split
adjustment in one vendor produces a fake 50 percent weekly return, and a strategy will
find it and load up on it.
"""

from __future__ import annotations

import io
import time
from collections.abc import Sequence
from datetime import date

import pandas as pd

from ..types import (
    AVAILABLE_AT,
    CLOSE,
    DIV_CASH,
    HIGH,
    IS_DELISTED,
    LOW,
    OPEN,
    SESSION,
    SPLIT_FACTOR,
    TICKER,
    VOLUME,
)

_URL = "https://stooq.com/q/d/l/"
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; swingbot/0.1)"}

#: Stooq's market suffixes, keyed by our market name.
_MARKET_SUFFIX = {"us": ".us", "india": ".in"}


class StooqProvider:
    name = "stooq"

    def __init__(
        self,
        *,
        market: str = "us",
        close_hour_utc: int = 21,
        request_delay_seconds: float = 0.3,
        timeout: float = 20.0,
    ) -> None:
        self.market = market
        self.close_hour_utc = close_hour_utc
        self.request_delay_seconds = request_delay_seconds
        self.timeout = timeout

    def available(self) -> bool:
        try:
            import httpx  # noqa: F401
        except ImportError:
            return False
        return True

    def daily_bars(
        self, tickers: Sequence[str], start: date, end: date
    ) -> pd.DataFrame:
        try:
            import httpx
        except ImportError:
            return pd.DataFrame()

        suffix = _MARKET_SUFFIX.get(self.market, "")
        frames = []
        with httpx.Client(headers=_HEADERS, timeout=self.timeout, follow_redirects=True) as client:
            for i, ticker in enumerate(tickers):
                if i:
                    time.sleep(self.request_delay_seconds)
                symbol = f"{ticker}{suffix}".lower().replace("-", ".")
                try:
                    response = client.get(
                        _URL,
                        params={
                            "s": symbol,
                            "i": "d",
                            "d1": start.strftime("%Y%m%d"),
                            "d2": end.strftime("%Y%m%d"),
                        },
                    )
                    response.raise_for_status()
                    text = response.text
                except Exception:
                    continue

                # A JS challenge or an unknown symbol both come back as HTML.
                if not text.lstrip().lower().startswith("date,"):
                    continue

                try:
                    raw = pd.read_csv(io.StringIO(text))
                except Exception:
                    continue
                if raw.empty:
                    continue

                raw.columns = [c.strip().lower() for c in raw.columns]
                frame = pd.DataFrame(
                    {
                        TICKER: ticker,
                        SESSION: pd.to_datetime(raw["date"], errors="coerce").dt.date,
                        OPEN: pd.to_numeric(raw.get("open"), errors="coerce"),
                        HIGH: pd.to_numeric(raw.get("high"), errors="coerce"),
                        LOW: pd.to_numeric(raw.get("low"), errors="coerce"),
                        CLOSE: pd.to_numeric(raw.get("close"), errors="coerce"),
                        VOLUME: pd.to_numeric(raw.get("volume"), errors="coerce").fillna(0.0),
                    }
                ).dropna(subset=[SESSION, CLOSE])
                if frame.empty:
                    continue
                frame[SPLIT_FACTOR] = 1.0
                frame[DIV_CASH] = 0.0
                frame[IS_DELISTED] = False
                frame[AVAILABLE_AT] = pd.to_datetime(
                    frame[SESSION], utc=True
                ) + pd.Timedelta(hours=self.close_hour_utc)
                frames.append(frame)

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True).sort_values([TICKER, SESSION])

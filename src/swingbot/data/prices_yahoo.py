"""Yahoo Finance price provider.

Optional and quarantined behind the protocol on purpose. Yahoo is convenient and free,
and it is also survivorship-biased (delisted tickers simply vanish) and its adjusted
prices are retroactively restated after every corporate action. That is fine for a smoke
test and not fine for a capital decision.

This provider therefore requests **unadjusted** prices plus the corporate-action events
separately, so :mod:`swingbot.data.adjust` can build a point-in-time adjustment chain
rather than trusting a number that changes under the backtest.

It also rate-limits itself. Yahoo returns HTTP 429 aggressively to shared and cloud IPs,
so the chain treats a failure here as "try the next provider" rather than an error.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from datetime import UTC, date, datetime

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

_CHART_URL = "https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}


class YahooProvider:
    """Daily bars from Yahoo's chart endpoint."""

    name = "yahoo"

    def __init__(
        self,
        *,
        symbol_suffix: str = "",
        close_hour_utc: int = 21,
        request_delay_seconds: float = 0.4,
        timeout: float = 20.0,
        max_retries: int = 2,
    ) -> None:
        self.symbol_suffix = symbol_suffix
        self.close_hour_utc = close_hour_utc
        self.request_delay_seconds = request_delay_seconds
        self.timeout = timeout
        self.max_retries = max_retries

    def available(self) -> bool:
        try:
            import httpx  # noqa: F401
        except ImportError:
            return False
        return True

    # ------------------------------------------------------------------------ public

    def daily_bars(
        self, tickers: Sequence[str], start: date, end: date
    ) -> pd.DataFrame:
        try:
            import httpx
        except ImportError:
            return pd.DataFrame()

        frames = []
        with httpx.Client(headers=_HEADERS, timeout=self.timeout, follow_redirects=True) as client:
            for i, ticker in enumerate(tickers):
                if i:
                    time.sleep(self.request_delay_seconds)
                frame = self._fetch_one(client, ticker, start, end)
                if not frame.empty:
                    frames.append(frame)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True).sort_values([TICKER, SESSION])

    # ----------------------------------------------------------------------- private

    def _symbol(self, ticker: str) -> str:
        if self.symbol_suffix and not ticker.endswith(self.symbol_suffix):
            return f"{ticker}{self.symbol_suffix}"
        return ticker

    def _fetch_one(self, client, ticker: str, start: date, end: date) -> pd.DataFrame:
        params = {
            "period1": int(datetime.combine(start, datetime.min.time(), UTC).timestamp()),
            "period2": int(datetime.combine(end, datetime.max.time(), UTC).timestamp()),
            "interval": "1d",
            "events": "div,split",
            "includeAdjustedClose": "true",
        }
        url = _CHART_URL.format(symbol=self._symbol(ticker))

        payload = None
        for attempt in range(self.max_retries + 1):
            try:
                response = client.get(url, params=params)
                if response.status_code == 429:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                response.raise_for_status()
                payload = response.json()
                break
            except Exception:
                if attempt >= self.max_retries:
                    return pd.DataFrame()
                time.sleep(1.0 * (attempt + 1))
        if payload is None:
            return pd.DataFrame()

        try:
            result = payload["chart"]["result"][0]
            stamps = result["timestamp"]
            quote = result["indicators"]["quote"][0]
        except (KeyError, IndexError, TypeError):
            return pd.DataFrame()

        sessions = [
            datetime.fromtimestamp(int(ts), UTC).date() for ts in stamps
        ]
        frame = pd.DataFrame(
            {
                TICKER: ticker,
                SESSION: sessions,
                OPEN: quote.get("open"),
                HIGH: quote.get("high"),
                LOW: quote.get("low"),
                CLOSE: quote.get("close"),
                VOLUME: quote.get("volume"),
            }
        ).dropna(subset=[CLOSE])
        if frame.empty:
            return frame

        events = result.get("events", {}) or {}
        frame[SPLIT_FACTOR] = 1.0
        frame[DIV_CASH] = 0.0

        session_pos = {s: i for i, s in enumerate(frame[SESSION].tolist())}
        for raw in (events.get("splits") or {}).values():
            session = datetime.fromtimestamp(int(raw["date"]), UTC).date()
            idx = session_pos.get(session)
            if idx is not None:
                denom = float(raw.get("denominator") or 1.0)
                numer = float(raw.get("numerator") or 1.0)
                if denom:
                    frame.iat[idx, frame.columns.get_loc(SPLIT_FACTOR)] = numer / denom
        for raw in (events.get("dividends") or {}).values():
            session = datetime.fromtimestamp(int(raw["date"]), UTC).date()
            idx = session_pos.get(session)
            if idx is not None:
                frame.iat[idx, frame.columns.get_loc(DIV_CASH)] = float(raw.get("amount") or 0.0)

        frame[IS_DELISTED] = False
        frame[AVAILABLE_AT] = pd.to_datetime(frame[SESSION], utc=True) + pd.Timedelta(
            hours=self.close_hour_utc
        )
        return frame.reset_index(drop=True)

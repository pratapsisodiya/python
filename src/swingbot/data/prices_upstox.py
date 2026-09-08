"""Daily NSE bars from Upstox's public historical-candle API.

Preferred over Yahoo for Indian equities, for three reasons that matter to this system.

**It is the exchange's own data, one hop away.** An Indian broker's feed of NSE candles,
rather than a US aggregator's re-publication of it. Symbol coverage, corporate-action
handling and the session calendar are all NSE's rather than approximately NSE's.

**Timestamps are already in IST.** Yahoo returns epoch seconds that have to be converted,
and a one-off error there shifts a whole series by a session — the exact failure the
point-in-time layer exists to catch. Upstox returns ISO strings with ``+05:30`` on them.

**It does not rate-limit a cloud address into uselessness.** Yahoo answers a single
request and then returns 429 to a hundred, which is fatal for a 130-name universe.

The one thing it does not give is corporate actions: the candles are unadjusted and there
is no split or dividend stream. That is stated rather than papered over — ``split_factor``
and ``div_cash`` come back as neutral, and :mod:`swingbot.data.adjust` therefore has
nothing to adjust. For a weekly cross-sectional strategy on large caps that is a real but
bounded problem: an unadjusted split shows up as a single enormous one-week return, which
the winsorising in the feature layer clips, and the liquidity screen drops the name. It is
still a reason to prefer a vendor with an action history if you have one, and the report
says so.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from datetime import date
from urllib.parse import quote

import pandas as pd

from ..types import (
    AVAILABLE_AT,
    BAR_COLUMNS,
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

log = logging.getLogger(__name__)

_CANDLE_URL = "https://api.upstox.com/v2/historical-candle/{key}/day/{to}/{from_}"

#: The API returns nothing for a window that reaches too far back. Daily history is
#: available from roughly 2018; asking for 2015 silently yields an empty list rather than
#: an error, which is worse than a refusal, so the floor is explicit here.
EARLIEST_SESSION = date(2018, 1, 1)


class UpstoxProvider:
    """Daily bars for NSE equities, keyed by Upstox instrument key."""

    name = "upstox"

    def __init__(
        self,
        instrument_keys: dict[str, str] | None = None,
        *,
        close_hour_utc: int = 10,
        request_delay_seconds: float = 0.12,
        timeout: float = 25.0,
        max_retries: int = 2,
        snapshot_path: str | None = None,
    ) -> None:
        self.close_hour_utc = close_hour_utc
        self.request_delay_seconds = request_delay_seconds
        self.timeout = timeout
        self.max_retries = max_retries
        self._keys = dict(instrument_keys or {})
        self._snapshot_path = snapshot_path

    # ---------------------------------------------------------------- instrument keys

    def keys_for(self, tickers: Sequence[str]) -> dict[str, str]:
        """Instrument keys for the requested tickers, loading the snapshot on demand."""
        if self._keys:
            return {t: k for t in tickers if (k := self._keys.get(t))}

        from .instruments_nse import NSEInstruments

        instruments = (
            NSEInstruments.load_if_present(self._snapshot_path)
            if self._snapshot_path
            else NSEInstruments.load_if_present()
        )
        if instruments is None:
            return {}
        self._keys = {
            ticker: key
            for ticker in tickers
            if (key := instruments.instrument_key(ticker))
        }
        return dict(self._keys)

    def available(self) -> bool:
        """True when instrument keys can be resolved at all.

        Deliberately does not probe the network. A provider that reports itself
        unavailable because of one slow request would be skipped by the chain and the
        run would silently fall through to synthetic data.
        """
        from .instruments_nse import NSEInstruments

        return NSEInstruments.load_if_present(
            self._snapshot_path
        ) is not None if self._snapshot_path else NSEInstruments.load_if_present() is not None

    # ------------------------------------------------------------------------- fetching

    def daily_bars(
        self, tickers: Sequence[str], start: date, end: date
    ) -> pd.DataFrame:
        import httpx

        keys = self.keys_for(list(tickers))
        if not keys:
            log.warning(
                "upstox: no instrument keys for the requested tickers; run "
                "`swingbot fetch instruments --market india`"
            )
            return pd.DataFrame(columns=list(BAR_COLUMNS))

        floor = max(start, EARLIEST_SESSION)
        if floor > start:
            log.info(
                "upstox: daily history starts around %s, so %s was raised to it",
                EARLIEST_SESSION, start,
            )

        frames = []
        with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
            for ticker, key in keys.items():
                frame = self._fetch_one(client, ticker, key, floor, end)
                if not frame.empty:
                    frames.append(frame)
                time.sleep(self.request_delay_seconds)

        if not frames:
            return pd.DataFrame(columns=list(BAR_COLUMNS))

        out = pd.concat(frames, ignore_index=True)
        log.info(
            "upstox: %d bar(s) for %d of %d ticker(s)",
            len(out), out[TICKER].nunique(), len(tickers),
        )
        return out

    def _fetch_one(
        self, client, ticker: str, key: str, start: date, end: date
    ) -> pd.DataFrame:
        url = _CANDLE_URL.format(key=quote(key, safe=""), to=end, from_=start)

        for attempt in range(self.max_retries + 1):
            try:
                response = client.get(url)
                if response.status_code == 429:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                response.raise_for_status()
                payload = response.json()
            except Exception as exc:
                if attempt >= self.max_retries:
                    log.warning("upstox: %s failed: %s", ticker, exc)
                    return pd.DataFrame(columns=list(BAR_COLUMNS))
                time.sleep(1.0 * (attempt + 1))
                continue

            candles = (payload.get("data") or {}).get("candles") or []
            if not candles:
                return pd.DataFrame(columns=list(BAR_COLUMNS))
            return self._to_frame(ticker, candles)

        return pd.DataFrame(columns=list(BAR_COLUMNS))

    def _to_frame(self, ticker: str, candles: list[list]) -> pd.DataFrame:
        """One symbol's candles as bars.

        Candle layout is ``[timestamp, open, high, low, close, volume, open_interest]``,
        newest first, with an IST offset on the timestamp.
        """
        stamps = pd.to_datetime([row[0] for row in candles], utc=True, format="ISO8601")

        frame = pd.DataFrame(
            {
                TICKER: ticker,
                SESSION: stamps.tz_convert("Asia/Kolkata").date,
                OPEN: [float(row[1]) for row in candles],
                HIGH: [float(row[2]) for row in candles],
                LOW: [float(row[3]) for row in candles],
                CLOSE: [float(row[4]) for row in candles],
                VOLUME: [float(row[5]) for row in candles],
                # Unadjusted: the API carries no corporate-action stream. Neutral factors
                # are the honest representation — pretending to a 1.0 split we verified is
                # different from having no information, and the module docstring says which
                # of the two this is.
                SPLIT_FACTOR: 1.0,
                DIV_CASH: 0.0,
                IS_DELISTED: False,
            }
        )

        # A bar becomes knowable at its own close, expressed in UTC. NSE closes at 15:30
        # IST, which is 10:00 UTC, and `close_hour_utc` carries that from the market
        # profile rather than being assumed here.
        frame[AVAILABLE_AT] = pd.to_datetime(frame[SESSION], utc=True) + pd.Timedelta(
            hours=self.close_hour_utc
        )
        return frame.sort_values(SESSION).reset_index(drop=True)

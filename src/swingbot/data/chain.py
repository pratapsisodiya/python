"""Provider chain with fallback, caching and sanity screening.

Providers are tried in configured order and each one only needs to cover the tickers the
previous ones missed. Everything lands in a parquet cache so a rerun is instant and the
free rate-limited sources are not hammered.

The chain also runs the sanity screen, which is not optional decoration. Bad vendor data
is the single most common source of a fake backtest result: one mis-applied split shows
up as a 50 percent weekly return, and a cross-sectional ranker will find it and put the
whole book into it. Screening for impossible bars is cheaper than explaining an equity
curve that never happened.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from datetime import date

import numpy as np
import pandas as pd

from ..config import Config
from ..io.store import ParquetStore, write_json
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
from .prices_csv import CSVProvider
from .prices_stooq import StooqProvider
from .prices_upstox import UpstoxProvider
from .prices_yahoo import YahooProvider
from .synthetic import SyntheticProvider

log = logging.getLogger(__name__)

#: A single-session move larger than this, with no split recorded, is almost certainly
#: a data error rather than a real return. Kept generous so genuine event moves survive.
MAX_PLAUSIBLE_DAILY_MOVE = 0.60


class ProviderChain:
    """Tries several providers in order, caches the union."""

    name = "chain"

    def __init__(
        self,
        providers: Sequence[object],
        *,
        store: ParquetStore | None = None,
        cache_enabled: bool = True,
    ) -> None:
        self.providers = list(providers)
        self.store = store
        self.cache_enabled = cache_enabled and store is not None
        #: ticker -> provider name, populated by :meth:`daily_bars`.
        self.attribution: dict[str, str] = {}

    # ---------------------------------------------------------------- construction

    @classmethod
    def from_config(cls, cfg: Config, *, store: ParquetStore | None = None) -> ProviderChain:
        close_hour = _close_hour_utc(cfg)
        built: list[object] = []
        for name in cfg.data.providers:
            key = name.strip().lower()
            if key == "csv":
                built.append(
                    CSVProvider(cfg.market_dir / "csv", close_hour_utc=close_hour)
                )
            elif key == "yahoo":
                built.append(
                    YahooProvider(
                        symbol_suffix=cfg.market_profile.symbol_suffix,
                        close_hour_utc=close_hour,
                    )
                )
            elif key == "upstox":
                built.append(
                    UpstoxProvider(close_hour_utc=close_hour)
                )
            elif key == "stooq":
                built.append(
                    StooqProvider(
                        market=cfg.market_profile.name, close_hour_utc=close_hour
                    )
                )
            elif key == "synthetic":
                built.append(
                    SyntheticProvider(seed=cfg.run.seed, close_hour_utc=close_hour)
                )
            else:
                raise ValueError(f"Unknown price provider {name!r}")
        return cls(built, store=store, cache_enabled=cfg.data.cache_enabled)

    def available(self) -> bool:
        return any(getattr(p, "available", lambda: True)() for p in self.providers)

    #: Provider names whose output is generated rather than observed.
    FABRICATED_SOURCES = frozenset({"synthetic", "synthetic-csv"})

    def fabricated_tickers(self) -> list[str]:
        """Tickers whose bars came from a generator rather than a market.

        The one question worth asking about a chain's output, and it has two answers
        because there are two ways to get invented prices. A live chain ending in
        ``synthetic`` will answer for any ticker a real source could not supply — that is
        how three renamed NSE symbols joined 126 genuine ones. And ``swingbot demo``
        writes its generated CSVs into the same directory a user's own exports go, where
        they shadow everything, which is how a whole real fetch got quietly ignored.

        Both come back here, so a single caveat covers both.
        """
        return sorted(
            t for t, p in self.attribution.items() if p in self.FABRICATED_SOURCES
        )

    # ---------------------------------------------------------------------- public

    def daily_bars(
        self,
        tickers: Sequence[str],
        start: date,
        end: date,
        *,
        use_cache: bool = True,
        refresh: bool = False,
    ) -> pd.DataFrame:
        tickers = list(dict.fromkeys(tickers))
        collected: dict[str, pd.DataFrame] = {}
        # Which provider supplied each ticker. Recorded because the chain's whole job is
        # to fall through until something answers, and the last thing in a live chain is
        # usually a generator. Without attribution a run that quietly invented prices for
        # a handful of names looks exactly like one that did not.
        self.attribution = {}

        if use_cache and self.cache_enabled and not refresh:
            cached = self._read_cache()
            if not cached.empty:
                # The sidecar remembers which provider originally answered, so a cache
                # hit does not launder generated prices into apparently real ones.
                remembered = self._read_attribution()
                for ticker, group in cached.groupby(TICKER, sort=False):
                    if ticker in tickers and _covers(group, start, end):
                        collected[str(ticker)] = group
                        self.attribution[str(ticker)] = remembered.get(
                            str(ticker), "cache"
                        )

        missing = [t for t in tickers if t not in collected]

        for provider in self.providers:
            if not missing:
                break
            try:
                if not getattr(provider, "available", lambda: True)():
                    continue
            except Exception:
                continue
            try:
                frame = provider.daily_bars(missing, start, end)
            except Exception as exc:
                log.warning("provider %s failed: %s", getattr(provider, "name", provider), exc)
                continue
            if frame is None or frame.empty:
                continue
            frame = _normalise(frame)
            provider_name = str(getattr(provider, "name", provider))
            for ticker, group in frame.groupby(TICKER, sort=False):
                if str(ticker) in collected:
                    continue
                collected[str(ticker)] = group
                self.attribution[str(ticker)] = provider_name
            got = {str(t) for t in frame[TICKER].unique()}
            missing = [t for t in missing if t not in got]
            log.info(
                "provider %s supplied %d ticker(s), %d still missing",
                getattr(provider, "name", provider),
                len(got),
                len(missing),
            )

        if missing:
            log.warning(
                "no provider supplied %d ticker(s): %s",
                len(missing), ", ".join(missing[:8]) + ("..." if len(missing) > 8 else ""),
            )

        if not collected:
            return pd.DataFrame(columns=list(BAR_COLUMNS))

        out = pd.concat(collected.values(), ignore_index=True)
        out = _normalise(out)
        out = screen_bars(out)

        if self.cache_enabled:
            self._write_cache(out)
        return out

    # ----------------------------------------------------------------------- cache

    def _cache_parts(self) -> tuple[str, ...]:
        return ("raw", "bars.parquet")

    def _attribution_path(self):
        return self.store.path("raw", "attribution.json") if self.store else None

    def _read_cache(self) -> pd.DataFrame:
        if self.store is None:
            return pd.DataFrame()
        frame = self.store.read(*self._cache_parts())
        return _normalise(frame) if not frame.empty else frame

    def _read_attribution(self) -> dict[str, str]:
        """Where each cached ticker's bars originally came from.

        A sidecar rather than a column on the bars, because provenance is a property of
        the *series*, not of each row, and adding a column would change the bar schema
        that :mod:`swingbot.types` pins and the tests assert.

        This exists because losing it was a real defect. Attribution was recorded at fetch
        time and thrown away on the way into parquet, so a cache holding three invented
        NSE symbols alongside 126 real ones came back on the next run indistinguishable
        from a wholly real dataset — and the caveat that should have said so could not
        fire.
        """
        path = self._attribution_path()
        if path is None or not path.exists():
            return {}
        try:
            return {
                str(k): str(v)
                for k, v in json.loads(path.read_text()).get("providers", {}).items()
            }
        except (OSError, ValueError, AttributeError):
            return {}

    def _write_cache(self, frame: pd.DataFrame) -> None:
        if self.store is None:
            return
        self.store.append(frame, *self._cache_parts(), dedupe_on=[TICKER, SESSION])
        self._write_attribution()

    def _write_attribution(self) -> None:
        """Merge this fetch's provenance into the sidecar.

        Merged rather than replaced: one run may fetch a handful of names while the cache
        holds hundreds, and forgetting the rest would make a partial refresh look like a
        clean dataset. ``cache`` is never written — it is not a source, it is where a
        source's answer was kept.
        """
        path = self._attribution_path()
        if path is None:
            return
        merged = self._read_attribution()
        merged.update(
            {t: p for t, p in self.attribution.items() if p and p != "cache"}
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, {"providers": merged})


# --------------------------------------------------------------------------------------
# normalisation and screening
# --------------------------------------------------------------------------------------


def _close_hour_utc(cfg: Config) -> int:
    """UTC hour of the market close, used to stamp ``available_at`` on each bar."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    hour, _, minute = cfg.market_profile.close_local_time.partition(":")
    local = datetime(
        2024, 6, 3, int(hour), int(minute or 0), tzinfo=ZoneInfo(cfg.market_profile.timezone)
    )
    return int(pd.Timestamp(local).tz_convert("UTC").hour)


def _covers(group: pd.DataFrame, start: date, end: date) -> bool:
    """Whether a cached group spans enough of the requested window to be reused."""
    if group.empty:
        return False
    sessions = pd.to_datetime(group[SESSION])
    lo, hi = sessions.min().date(), sessions.max().date()
    # Allow a week of slack at each end for holidays and weekends.
    return lo <= start + pd.Timedelta(days=7).to_pytimedelta() and hi >= end - pd.Timedelta(
        days=7
    ).to_pytimedelta()


def _normalise(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce a bar frame to the canonical schema and dtypes."""
    if frame.empty:
        return pd.DataFrame(columns=list(BAR_COLUMNS))
    out = frame.copy()
    if SPLIT_FACTOR not in out.columns:
        out[SPLIT_FACTOR] = 1.0
    if DIV_CASH not in out.columns:
        out[DIV_CASH] = 0.0
    if IS_DELISTED not in out.columns:
        out[IS_DELISTED] = False

    out[TICKER] = out[TICKER].astype(str)
    out[SESSION] = pd.to_datetime(out[SESSION]).dt.date
    for col in (OPEN, HIGH, LOW, CLOSE, VOLUME, SPLIT_FACTOR, DIV_CASH):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out[SPLIT_FACTOR] = out[SPLIT_FACTOR].fillna(1.0)
    out[DIV_CASH] = out[DIV_CASH].fillna(0.0)
    out[IS_DELISTED] = out[IS_DELISTED].astype(bool)
    out[AVAILABLE_AT] = pd.to_datetime(out[AVAILABLE_AT], utc=True)

    out = out.dropna(subset=[CLOSE, SESSION])
    out = out.drop_duplicates(subset=[TICKER, SESSION], keep="last")
    return out[list(BAR_COLUMNS)].sort_values([TICKER, SESSION]).reset_index(drop=True)


def screen_bars(frame: pd.DataFrame, *, max_move: float = MAX_PLAUSIBLE_DAILY_MOVE) -> pd.DataFrame:
    """Drop bars that cannot be real.

    Three checks, all cheap and all worth it:

    * Non-positive prices, which break every log return downstream.
    * An OHLC set that is internally inconsistent, meaning high below low or a close
      outside the range.
    * A single-session move beyond ``max_move`` with no split recorded, which is the
      signature of an unadjusted corporate action.
    """
    if frame.empty:
        return frame

    out = frame.copy()
    positive = (out[[OPEN, HIGH, LOW, CLOSE]] > 0).all(axis=1)
    consistent = (out[HIGH] >= out[LOW]) & (
        out[CLOSE].between(out[LOW] * 0.999, out[HIGH] * 1.001)
    )
    out = out.loc[positive & consistent]
    if out.empty:
        return out

    grouped = out.groupby(TICKER, sort=False)[CLOSE]
    log_move = np.log(out[CLOSE]) - np.log(grouped.shift(1))
    unsplit = out[SPLIT_FACTOR].fillna(1.0).eq(1.0)
    implausible = log_move.abs().gt(np.log(1.0 + max_move)) & unsplit
    implausible = implausible.fillna(False)

    if bool(implausible.any()):
        log.warning(
            "screened %d bar(s) with an implausible unsplit move over %.0f%%",
            int(implausible.sum()),
            max_move * 100,
        )
    return out.loc[~implausible].reset_index(drop=True)


def apply_liquidity_filter(
    frame: pd.DataFrame, *, min_price: float, min_adv_notional: float, window: int = 20
) -> pd.DataFrame:
    """Flag rows failing the price and turnover floors.

    Adds ``adv_notional`` and ``liquid`` columns rather than dropping rows, because the
    decision of whether a name was tradeable belongs at the decision date and must use
    only trailing information. The rolling mean is shifted by one session so a name is
    never judged liquid on the strength of a bar it has not printed yet.
    """
    if frame.empty:
        return frame
    out = frame.sort_values([TICKER, SESSION]).copy()
    notional = out[CLOSE] * out[VOLUME]
    out["adv_notional"] = (
        notional.groupby(out[TICKER], sort=False)
        .transform(lambda s: s.rolling(window, min_periods=max(5, window // 4)).mean())
        .groupby(out[TICKER], sort=False)
        .shift(1)
    )
    out["liquid"] = (out[CLOSE] >= min_price) & (
        out["adv_notional"].fillna(0.0) >= min_adv_notional
    )
    return out

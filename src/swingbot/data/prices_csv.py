"""Local CSV / parquet price provider.

The escape hatch that makes the system genuinely vendor-independent. Whatever your data
source, if you can export ``session,open,high,low,close,volume`` you can use this, and it
works with no network at all.

Files live at ``data/<market>/csv/<TICKER>.csv``. A ``session`` column is required;
everything else is filled with sane defaults. If the file carries ``split_factor`` and
``div_cash`` those are used, otherwise the series is assumed already continuous.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from pathlib import Path

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

_ALIASES = {
    "date": SESSION,
    "datetime": SESSION,
    "timestamp": SESSION,
    "time": SESSION,
    "adj close": "adj_close",
    "adj_close": "adj_close",
    "adjclose": "adj_close",
    "o": OPEN,
    "h": HIGH,
    "l": LOW,
    "c": CLOSE,
    "v": VOLUME,
    "vol": VOLUME,
    "qty": VOLUME,
    "shares traded": VOLUME,
    "total traded quantity": VOLUME,
}


#: Written by ``swingbot demo`` beside the CSVs it generates.
#:
#: A generated CSV is byte-indistinguishable from a real export, so without a marker the
#: chain reports "csv supplied 129 tickers" whether the prices came from an exchange or
#: from a random walk. That is not a hypothetical: leftover demo files silently shadowed a
#: real NSE fetch here, and every number downstream was measuring the generator.
GENERATED_MARKER = "GENERATED.json"


class CSVProvider:
    """Reads local CSV or parquet files, one per ticker."""

    def __init__(self, directory: Path | str, *, close_hour_utc: int = 21) -> None:
        self.directory = Path(directory)
        self.close_hour_utc = close_hour_utc

    @property
    def name(self) -> str:
        """``csv``, or ``synthetic-csv`` when the directory is marked as generated.

        A property rather than a class attribute so the answer follows the directory. The
        chain records this name as the provenance of every ticker it supplies, so renaming
        here is what makes the synthetic-data caveat fire on a poisoned CSV directory.
        """
        return "synthetic-csv" if self.is_generated else "csv"

    @property
    def is_generated(self) -> bool:
        return (self.directory / GENERATED_MARKER).exists()

    def available(self) -> bool:
        return self.directory.exists() and any(self._candidates())

    def _candidates(self):
        if not self.directory.exists():
            return []
        return sorted(
            list(self.directory.glob("*.csv"))
            + list(self.directory.glob("*.CSV"))
            + list(self.directory.glob("*.parquet"))
        )

    def _find(self, ticker: str) -> Path | None:
        for suffix in (".csv", ".CSV", ".parquet"):
            candidate = self.directory / f"{ticker}{suffix}"
            if candidate.exists():
                return candidate
        # Tolerate a market suffix in the filename, e.g. RELIANCE.NS.csv
        for candidate in self._candidates():
            stem = candidate.name.rsplit(".", 1)[0]
            if stem.split(".")[0].upper() == ticker.upper():
                return candidate
        return None

    def daily_bars(
        self, tickers: Sequence[str], start: date, end: date
    ) -> pd.DataFrame:
        frames = []
        for ticker in tickers:
            path = self._find(ticker)
            if path is None:
                continue
            frame = self._read_one(path, ticker)
            if frame.empty:
                continue
            mask = (frame[SESSION] >= start) & (frame[SESSION] <= end)
            frames.append(frame.loc[mask])
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True).sort_values([TICKER, SESSION])

    def _read_one(self, path: Path, ticker: str) -> pd.DataFrame:
        raw = (
            pd.read_parquet(path)
            if path.suffix.lower() == ".parquet"
            else pd.read_csv(path)
        )
        if raw.empty:
            return pd.DataFrame()

        raw.columns = [str(c).strip().lower() for c in raw.columns]
        raw = raw.rename(columns={k: v for k, v in _ALIASES.items() if k in raw.columns})

        if SESSION not in raw.columns:
            raise ValueError(f"{path} has no recognisable date column")

        frame = pd.DataFrame()
        frame[SESSION] = pd.to_datetime(raw[SESSION], errors="coerce").dt.date
        for col in (OPEN, HIGH, LOW, CLOSE, VOLUME):
            if col in raw.columns:
                frame[col] = pd.to_numeric(raw[col], errors="coerce")
            elif col == VOLUME:
                frame[col] = 0.0
            elif CLOSE in raw.columns:
                frame[col] = pd.to_numeric(raw[CLOSE], errors="coerce")
            else:
                raise ValueError(f"{path} is missing both {col!r} and {CLOSE!r}")

        frame[TICKER] = ticker
        frame[SPLIT_FACTOR] = (
            pd.to_numeric(raw[SPLIT_FACTOR], errors="coerce").fillna(1.0)
            if SPLIT_FACTOR in raw.columns
            else 1.0
        )
        frame[DIV_CASH] = (
            pd.to_numeric(raw[DIV_CASH], errors="coerce").fillna(0.0)
            if DIV_CASH in raw.columns
            else 0.0
        )
        frame[IS_DELISTED] = (
            raw[IS_DELISTED].astype(bool) if IS_DELISTED in raw.columns else False
        )
        frame[AVAILABLE_AT] = pd.to_datetime(frame[SESSION], utc=True) + pd.Timedelta(
            hours=self.close_hour_utc
        )

        frame = frame.dropna(subset=[SESSION, CLOSE])
        return frame.sort_values(SESSION).reset_index(drop=True)


def write_csv_fixture(frame: pd.DataFrame, directory: Path | str) -> list[Path]:
    """Write a bar panel out as per-ticker CSVs. Used by the demo and by tests."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for ticker, group in frame.groupby(TICKER, sort=True):
        target = directory / f"{ticker}.csv"
        columns = [SESSION, OPEN, HIGH, LOW, CLOSE, VOLUME, SPLIT_FACTOR, DIV_CASH]
        group[columns].to_csv(target, index=False)
        written.append(target)
    return written

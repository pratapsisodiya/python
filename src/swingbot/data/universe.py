"""Point-in-time universe membership.

There is deliberately no way to ask this module for "the tickers". Every query takes a
date, because a universe without a date is how survivorship bias gets into a backtest:
running a 2015 strategy on today's index constituents means trading names that had not
yet succeeded, and never trading the ones that failed.

The membership file carries ``start_date`` and ``end_date``. When ``end_date`` is absent
from the file entirely, the universe is a present-day snapshot and results *are*
inflated. Rather than fail or pretend otherwise, the loader flags it and every report
generated from it is stamped survivorship-biased.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd

from ..types import SECTOR, TICKER

log = logging.getLogger(__name__)

#: Terminal return booked when a name leaves the universe involuntarily and no vendor
#: delisting return is available. Deliberately punitive: bankruptcies should cost money
#: in a backtest, and a zero default makes shorting failures look free.
DEFAULT_DELISTING_RETURN = -0.30


class UniverseFile:
    """Membership loaded from a CSV with ``ticker,name,sector,start_date,end_date``."""

    name = "file"

    def __init__(
        self,
        path: Path | str,
        *,
        max_names: int = 500,
        default_delisting_return: float = DEFAULT_DELISTING_RETURN,
    ) -> None:
        self.path = Path(path)
        self.max_names = max_names
        self.default_delisting_return = default_delisting_return
        self._frame = self._load()

    # ----------------------------------------------------------------------- loading

    def _load(self) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(f"Universe file not found: {self.path}")
        raw = pd.read_csv(self.path)
        raw.columns = [c.strip().lower() for c in raw.columns]
        if TICKER not in raw.columns:
            raise ValueError(f"{self.path} must have a 'ticker' column")

        frame = pd.DataFrame()
        frame[TICKER] = raw[TICKER].astype(str).str.strip()
        frame["name"] = raw["name"].astype(str) if "name" in raw.columns else frame[TICKER]
        frame[SECTOR] = (
            raw[SECTOR].astype(str).str.strip() if SECTOR in raw.columns else "Unknown"
        )
        # Kept as datetime64 rather than object-dtype dates: an all-empty end_date column
        # otherwise re-infers as datetime64 on assignment and then refuses to compare
        # against a plain date.
        frame["start_date"] = (
            pd.to_datetime(raw["start_date"], errors="coerce")
            if "start_date" in raw.columns
            else pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")
        )
        self._has_end_column = "end_date" in raw.columns
        frame["end_date"] = (
            pd.to_datetime(raw["end_date"], errors="coerce")
            if self._has_end_column
            else pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")
        )
        frame["delisting_return"] = (
            pd.to_numeric(raw["delisting_return"], errors="coerce")
            if "delisting_return" in raw.columns
            else pd.NA
        )
        # Optional per-symbol tradeable increment.
        #
        # Needed because a single-stock future does not trade in shares: NSE defines a lot
        # per underlying and revises it, so "sell 173" of a future is not an order anyone
        # can place. A single market-wide number would be wrong for most names and could
        # round a 250-lot symbol up to 500, doubling the intended exposure — worse than
        # not rounding. So it is per symbol, optional, and absence is reported rather than
        # guessed at.
        frame["lot_size"] = (
            pd.to_numeric(raw["lot_size"], errors="coerce")
            if "lot_size" in raw.columns
            else pd.NA
        )
        return frame.drop_duplicates(subset=[TICKER], keep="first").reset_index(drop=True)

    # ------------------------------------------------------------------------ public

    def members_asof(self, asof: date) -> pd.DataFrame:
        """Members at ``asof``, including names that were later delisted."""
        frame = self._frame
        stamp = pd.Timestamp(asof)
        started = frame["start_date"].isna() | (frame["start_date"] <= stamp)
        not_ended = frame["end_date"].isna() | (frame["end_date"] >= stamp)
        out = frame.loc[started & not_ended]
        if self.max_names and len(out) > self.max_names:
            out = out.head(self.max_names)
        return out[[TICKER, "name", SECTOR]].reset_index(drop=True)

    def all_tickers(self) -> list[str]:
        """Every ticker that was ever a member.

        Used only to decide what price history to *download*. Never used to decide what
        is tradeable on a given date; that is what :meth:`members_asof` is for.
        """
        tickers = self._frame[TICKER].tolist()
        return tickers[: self.max_names] if self.max_names else tickers

    def sectors(self) -> dict[str, str]:
        return dict(zip(self._frame[TICKER], self._frame[SECTOR], strict=True))

    def lot_sizes(self) -> dict[str, int]:
        """Per-symbol tradeable increment, for the symbols that declare one.

        Only symbols with a usable value appear. A caller must treat a missing entry as
        "unknown", never as 1: for cash equity 1 is right, but for a derivative it means
        the order has not been rounded to a placeable size and somebody has to check.
        """
        column = self._frame["lot_size"]
        return {
            str(ticker): int(value)
            for ticker, value in zip(self._frame[TICKER], column, strict=True)
            if pd.notna(value) and int(value) > 1
        }

    def delisting_return(self, ticker: str) -> float | None:
        row = self._frame.loc[self._frame[TICKER] == ticker]
        if row.empty:
            return None
        if row["end_date"].isna().all():
            return None
        explicit = row["delisting_return"].iloc[0]
        if pd.notna(explicit):
            return float(explicit)
        return self.default_delisting_return

    def delisting_sessions(self) -> dict[str, date]:
        """Ticker to the date it left the universe, for names that did."""
        ended = self._frame.loc[self._frame["end_date"].notna()]
        return dict(zip(ended[TICKER], ended["end_date"].dt.date, strict=True))

    #: Below this share of dated exits a file is still treated as a present-day snapshot.
    #:
    #: Any nonzero threshold is a judgement, so here is the one behind this number. A
    #: broad index turns over on the order of 5–10% of its members a year, so a file
    #: covering several years of genuine membership history carries exits in the tens of
    #: percent. 5% is roughly one year's turnover: below it, a file cannot be spanning a
    #: backtest-length period and recording what actually left.
    SURVIVORSHIP_EXIT_SHARE = 0.05

    def n_exits(self) -> int:
        """How many names in the file are recorded as having left the universe."""
        if not self._has_end_column:
            return 0
        return int(self._frame["end_date"].notna().sum())

    def is_survivorship_biased(self) -> bool:
        """True when the file is a present-day snapshot rather than membership history.

        Not simply "has no end dates at all", which is the version this started as and
        which turned out to be disarmable by one row: adding a single ``end_date`` to
        nifty200.csv — a genuine change, LTIM merging away — silenced the warning on a
        file where the other 128 names were still exactly the survivors of the index as
        it stands today. The strongest claim a file with one exit supports is that
        somebody edited one line, and the warning existed to catch precisely the dataset
        it then stopped firing on.
        """
        return self.n_exits() < max(1, int(len(self._frame) * self.SURVIVORSHIP_EXIT_SHARE))

    def bias_warning(self) -> str | None:
        if not self.is_survivorship_biased():
            return None
        n_exits, n_total = self.n_exits(), len(self._frame)
        history = (
            "has no delisting history"
            if not n_exits
            else f"records only {n_exits} exit(s) across {n_total} names"
        )
        return (
            f"Universe {self.path.name} {history}, so it is effectively a present-day "
            "snapshot. Backtest results from it are survivorship-biased and optimistic. "
            "Replace it with a point-in-time membership file before trusting any number."
        )


class StaticUniverse:
    """A fixed ticker list. For tests and quick experiments only."""

    name = "static"

    def __init__(
        self,
        tickers: list[str],
        sectors: dict[str, str] | None = None,
        lot_sizes: dict[str, int] | None = None,
    ) -> None:
        self.tickers = list(tickers)
        self._sectors = sectors or {}
        self._lot_sizes = lot_sizes or {}

    def members_asof(self, asof: date) -> pd.DataFrame:  # noqa: ARG002
        return pd.DataFrame(
            {
                TICKER: self.tickers,
                "name": self.tickers,
                SECTOR: [self._sectors.get(t, "Unknown") for t in self.tickers],
            }
        )

    def all_tickers(self) -> list[str]:
        return list(self.tickers)

    def sectors(self) -> dict[str, str]:
        return dict(self._sectors)

    def lot_sizes(self) -> dict[str, int]:
        return dict(self._lot_sizes)

    def delisting_return(self, ticker: str) -> float | None:  # noqa: ARG002
        return None

    def delisting_sessions(self) -> dict[str, date]:
        return {}

    def is_survivorship_biased(self) -> bool:
        return True

    def bias_warning(self) -> str | None:
        return "Static universe: no membership history, results are survivorship-biased."


def load_universe(cfg) -> UniverseFile | StaticUniverse:
    """Build the universe provider named in the config."""
    if cfg.universe.file is None:
        raise ValueError(
            f"No universe file configured for market {cfg.market}. "
            "Set universe.file in config/markets/<market>.yaml"
        )
    universe = UniverseFile(cfg.universe.file, max_names=cfg.universe.max_names)
    warning = universe.bias_warning()
    if warning and cfg.universe.assume_survivorship_biased:
        log.warning(warning)
    return universe


def membership_panel(
    universe, sessions: list[date], *, tickers: list[str] | None = None
) -> pd.DataFrame:
    """Long frame of ``session, ticker, sector`` for every session.

    Built once and joined into the feature panel, so a name is scored only on the dates
    it was actually a member.
    """
    rows = []
    allowed = set(tickers) if tickers is not None else None
    for session in sessions:
        members = universe.members_asof(session)
        if allowed is not None:
            members = members.loc[members[TICKER].isin(allowed)]
        if members.empty:
            continue
        block = members[[TICKER, SECTOR]].copy()
        block["session"] = session
        rows.append(block)
    if not rows:
        return pd.DataFrame(columns=["session", TICKER, SECTOR])
    out = pd.concat(rows, ignore_index=True)
    return out[["session", TICKER, SECTOR]]

"""The point-in-time firewall.

Every leakage guarantee in this system routes through this module. The rules it
enforces:

1. Every timestamp is timezone-aware UTC. A naive timestamp is a bug, not a convenience,
   because it silently compares wrong across a market close.
2. A row is visible at decision time ``t`` only when ``available_at <= t``. Strictly
   ``<=`` for bars, which are stamped at their own close, and strictly ``<`` for news,
   which must not be readable in the same instant it is published. See
   :func:`visible_news`.
3. Feature computation happens inside a :class:`PITGuard`, which checks the inputs
   before the transformer sees them and the outputs after.

The guard catches the honest mistakes. The truncation test in
``tests/test_lookahead_regression.py`` catches the subtle ones, because a transformer
can be handed perfectly valid inputs and still leak by fitting a scaler on the whole
panel.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime

import pandas as pd

from .types import AVAILABLE_AT, SESSION

#: Re-exported so callers need not import it separately; every timestamp here is UTC.
__all__ = ["UTC", "LookAheadError", "assert_pit", "ensure_utc", "visible_bars", "visible_news"]


class LookAheadError(AssertionError):
    """Raised when data newer than the decision time reaches a computation."""


# --------------------------------------------------------------------------------------
# Timestamp hygiene
# --------------------------------------------------------------------------------------


def ensure_utc(value: datetime | pd.Timestamp | str) -> pd.Timestamp:
    """Coerce to a timezone-aware UTC Timestamp, rejecting naive input.

    Naive timestamps are rejected rather than localised, because guessing a timezone is
    exactly the kind of silent assumption that produces an off-by-one-session leak.
    """
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        raise LookAheadError(
            f"Naive timestamp {value!r}. Every timestamp in swingbot must be tz-aware UTC."
        )
    return ts.tz_convert(UTC)


def ensure_utc_series(series: pd.Series) -> pd.Series:
    """Coerce a whole column to tz-aware UTC, rejecting naive values."""
    if len(series) == 0:
        return pd.to_datetime(series, utc=True)
    converted = pd.to_datetime(series, utc=False)
    tz = getattr(converted.dtype, "tz", None)
    if tz is None:
        raise LookAheadError(
            f"Column {series.name!r} contains naive timestamps. "
            "Stamp available_at in UTC at ingestion time."
        )
    return converted.dt.tz_convert(UTC)


def to_session(value: date | datetime | pd.Timestamp | str) -> date:
    """Normalise anything date-like to a plain ``datetime.date``."""
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    return pd.Timestamp(value).date()


# --------------------------------------------------------------------------------------
# Visibility
# --------------------------------------------------------------------------------------


def assert_pit(
    frame: pd.DataFrame,
    asof: pd.Timestamp,
    *,
    context: str = "",
    column: str = AVAILABLE_AT,
) -> None:
    """Raise if any row in ``frame`` could not have been known at ``asof``."""
    if frame.empty or column not in frame.columns:
        return
    asof = ensure_utc(asof)
    stamps = ensure_utc_series(frame[column])
    offenders = stamps > asof
    if bool(offenders.any()):
        n = int(offenders.sum())
        worst = stamps[offenders].max()
        where = f" in {context}" if context else ""
        raise LookAheadError(
            f"{n} row(s){where} are stamped after the decision time. "
            f"asof={asof.isoformat()} latest={worst.isoformat()}"
        )


def as_of_slice(
    frame: pd.DataFrame,
    asof: pd.Timestamp,
    *,
    column: str = AVAILABLE_AT,
    strict: bool = False,
) -> pd.DataFrame:
    """Return only the rows knowable at ``asof``.

    ``strict=True`` excludes rows stamped exactly at ``asof``. That is the correct
    setting for news: an article published at the instant of the close was not readable
    in time to act on that close. Bars use the inclusive form, because a bar is stamped
    at the close it describes.
    """
    if frame.empty:
        return frame
    if column not in frame.columns:
        raise KeyError(f"as_of_slice needs a {column!r} column; got {list(frame.columns)}")
    asof = ensure_utc(asof)
    stamps = ensure_utc_series(frame[column])
    mask = stamps < asof if strict else stamps <= asof
    return frame.loc[mask]


def visible_news(frame: pd.DataFrame, asof: pd.Timestamp, *, column: str = AVAILABLE_AT):
    """News visible at ``asof``, excluding items stamped exactly at the cutoff.

    Separated from :func:`as_of_slice` so the boundary convention lives in one place and
    is asserted by ``tests/test_news_asof_join.py``.
    """
    return as_of_slice(frame, asof, column=column, strict=True)


def visible_bars(frame: pd.DataFrame, asof: pd.Timestamp, *, column: str = AVAILABLE_AT):
    """Bars visible at ``asof``, including the bar stamped exactly at the cutoff."""
    return as_of_slice(frame, asof, column=column, strict=False)


def truncate_sessions(frame: pd.DataFrame, last_session: date, *, column: str = SESSION):
    """Drop every row for a session after ``last_session``.

    Used by the look-ahead regression test to build a physically truncated panel, which
    is a stronger check than filtering downstream: a leaky transformer cannot see data
    that is not in the frame at all.
    """
    if frame.empty or column not in frame.columns:
        return frame
    sessions = pd.to_datetime(frame[column]).dt.date
    return frame.loc[sessions <= last_session]


# --------------------------------------------------------------------------------------
# The guard
# --------------------------------------------------------------------------------------


class PITGuard:
    """Wraps a computation and checks visibility on the way in and on the way out.

    Usage::

        with PITGuard(asof, context="price features") as guard:
            guard.check(bars)
            out = transformer.transform(bars, asof)
            guard.check_output(out)
    """

    __slots__ = ("asof", "context", "_checked")

    def __init__(self, asof: pd.Timestamp, *, context: str = "") -> None:
        self.asof = ensure_utc(asof)
        self.context = context
        self._checked = 0

    def __enter__(self) -> PITGuard:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def check(self, frame: pd.DataFrame, *, column: str = AVAILABLE_AT) -> pd.DataFrame:
        assert_pit(frame, self.asof, context=self.context, column=column)
        self._checked += 1
        return frame

    def check_output(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Verify a transformer did not invent rows dated after the decision time."""
        if frame.empty:
            return frame
        if SESSION in frame.columns:
            sessions = pd.to_datetime(frame[SESSION]).dt.date
            asof_session = self.asof.date()
            late = sessions > asof_session
            if bool(late.any()):
                raise LookAheadError(
                    f"Output of {self.context or 'transformer'} contains "
                    f"{int(late.sum())} row(s) dated after {asof_session}."
                )
        return frame

    @property
    def checks_performed(self) -> int:
        return self._checked


@contextmanager
def pit_scope(asof: pd.Timestamp, *, context: str = "") -> Iterator[PITGuard]:
    guard = PITGuard(asof, context=context)
    yield guard


def assert_all_pit(frames: Iterable[tuple[str, pd.DataFrame]], asof: pd.Timestamp) -> None:
    """Convenience for checking several inputs at once."""
    for name, frame in frames:
        assert_pit(frame, asof, context=name)

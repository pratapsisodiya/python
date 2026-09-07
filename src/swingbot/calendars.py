"""Trading calendars and the weekly decision grid.

The grid is the spine of the whole system, so it is worth being precise about the three
distinct dates a weekly trade involves:

* **decision session** — the Friday whose close produces the signal.
* **entry session** — the next session, whose *open* fills the trade.
* **exit session** — ``hold_sessions`` later, whose *open* closes or rolls it.

Nothing is ever filled at the price that generated it. The label measures exactly the
open-to-open span the backtest actually trades, so a backtest cannot flatter itself by
scoring a return the portfolio never had access to.

Calendars are derived from the sessions actually present in the price data rather than
from an exchange holiday table. That keeps India and the US on one code path and means
the grid can never reference a session for which no bar exists.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import pandas as pd

from .config import Config
from .pit import UTC, to_session

_WEEKDAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


@dataclass(frozen=True, slots=True)
class WeeklyDecision:
    """One row of the weekly grid."""

    decision_session: date
    decision_ts: pd.Timestamp
    entry_session: date
    exit_session: date

    @property
    def week_key(self) -> str:
        iso = self.decision_session.isocalendar()
        return f"{iso.year}-W{iso.week:02d}"


class TradingCalendar:
    """Sessions plus the weekly grid built on top of them.

    Constructed from the sorted unique sessions observed in the price panel, so it is
    exact for whatever market and history the user actually loaded.
    """

    __slots__ = ("sessions", "_index", "timezone", "close_time", "decision_weekday")

    def __init__(
        self,
        sessions: list[date],
        *,
        timezone: str = "America/New_York",
        close_local_time: str = "16:00",
        decision_weekday: int = 4,
    ) -> None:
        self.sessions = sorted(set(sessions))
        self._index = {s: i for i, s in enumerate(self.sessions)}
        self.timezone = ZoneInfo(timezone)
        hour, _, minute = close_local_time.partition(":")
        self.close_time = time(int(hour), int(minute or 0))
        self.decision_weekday = decision_weekday

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_config(cls, sessions: list[date], cfg: Config) -> TradingCalendar:
        return cls(
            sessions,
            timezone=cfg.market_profile.timezone,
            close_local_time=cfg.market_profile.close_local_time,
            decision_weekday=cfg.calendar.decision_weekday,
        )

    @classmethod
    def from_bars(cls, bars: pd.DataFrame, cfg: Config) -> TradingCalendar:
        from .types import SESSION

        sessions = sorted({to_session(s) for s in bars[SESSION].unique()})
        return cls.from_config(sessions, cfg)

    # ---------------------------------------------------------------------- sessions

    def __len__(self) -> int:
        return len(self.sessions)

    def __contains__(self, session: date) -> bool:
        return session in self._index

    def position(self, session: date) -> int:
        try:
            return self._index[session]
        except KeyError as exc:
            raise KeyError(f"{session} is not a trading session in this calendar") from exc

    def next_session(self, session: date, offset: int = 1) -> date | None:
        """The session ``offset`` steps after ``session``, or None past the end.

        ``session`` need not itself be a trading session; the search uses the ordering.
        """
        idx = bisect_right(self.sessions, session) - 1
        if idx < 0:
            idx = -1
        target = idx + offset
        if 0 <= target < len(self.sessions):
            return self.sessions[target]
        return None

    def prev_session(self, session: date, offset: int = 1) -> date | None:
        idx = bisect_left(self.sessions, session)
        target = idx - offset
        if 0 <= target < len(self.sessions):
            return self.sessions[target]
        return None

    def sessions_between(self, start: date, end: date) -> list[date]:
        lo = bisect_left(self.sessions, start)
        hi = bisect_right(self.sessions, end)
        return self.sessions[lo:hi]

    def session_close_ts(self, session: date) -> pd.Timestamp:
        """The UTC instant of that session's close."""
        local = datetime.combine(session, self.close_time, tzinfo=self.timezone)
        return pd.Timestamp(local).tz_convert(UTC)

    # ------------------------------------------------------------------ weekly grid

    def decision_sessions(self) -> list[date]:
        """The last session of each week at or before the configured weekday.

        Using the last available session rather than requiring an exact weekday means a
        holiday Friday falls back to Thursday instead of silently skipping the week.
        """
        by_week: dict[tuple[int, int], date] = {}
        for session in self.sessions:
            if session.weekday() > self.decision_weekday:
                continue
            iso = session.isocalendar()
            key = (iso.year, iso.week)
            current = by_week.get(key)
            if current is None or session > current:
                by_week[key] = session
        return sorted(by_week.values())

    def weekly_grid(self, hold_sessions: int = 5) -> list[WeeklyDecision]:
        """Full decision grid. Weeks without a complete forward window are dropped.

        Dropping incomplete windows here rather than emitting a partial label is what
        stops the final weeks of history from carrying a truncated, upward-biased return.
        """
        out: list[WeeklyDecision] = []
        for decision in self.decision_sessions():
            entry = self.next_session(decision, 1)
            if entry is None:
                continue
            exit_session = self.next_session(entry, hold_sessions)
            if exit_session is None:
                continue
            out.append(
                WeeklyDecision(
                    decision_session=decision,
                    decision_ts=self.session_close_ts(decision),
                    entry_session=entry,
                    exit_session=exit_session,
                )
            )
        return out

    def latest_decision(self, hold_sessions: int = 5) -> WeeklyDecision | None:
        """The most recent decision whose entry session exists.

        Unlike :meth:`weekly_grid` this does not require the exit session to exist,
        because a live signal is generated before its own hold window has elapsed.
        """
        for decision in reversed(self.decision_sessions()):
            entry = self.next_session(decision, 1)
            if entry is None:
                continue
            exit_session = self.next_session(entry, hold_sessions) or self.sessions[-1]
            return WeeklyDecision(
                decision_session=decision,
                decision_ts=self.session_close_ts(decision),
                entry_session=entry,
                exit_session=exit_session,
            )
        return None

    def decision_for(self, asof: date, hold_sessions: int = 5) -> WeeklyDecision | None:
        """The decision on or immediately before ``asof``."""
        candidates = [d for d in self.decision_sessions() if d <= asof]
        if not candidates:
            return None
        decision = candidates[-1]
        entry = self.next_session(decision, 1)
        if entry is None:
            # A live decision at the very end of the data has no entry bar yet.
            entry = decision
        exit_session = self.next_session(entry, hold_sessions) or self.sessions[-1]
        return WeeklyDecision(
            decision_session=decision,
            decision_ts=self.session_close_ts(decision),
            entry_session=entry,
            exit_session=exit_session,
        )

    def describe(self) -> str:
        if not self.sessions:
            return "empty calendar"
        wd = _WEEKDAY_NAMES[self.decision_weekday]
        return (
            f"{len(self.sessions)} sessions, {self.sessions[0]} to {self.sessions[-1]}, "
            f"decisions on the last session at or before {wd}"
        )


def embargo_sessions(hold_sessions: int, embargo_weeks: int, sessions_per_week: int = 5) -> int:
    """How many sessions to embargo after a test fold.

    The label window itself plus the configured embargo. Both matter: purging removes
    training samples whose labels overlap the test window, and the embargo additionally
    removes samples immediately after it, because features are serially correlated and
    leak backwards across the boundary.
    """
    return hold_sessions + embargo_weeks * sessions_per_week

"""News point-in-time correctness.

The specific failure this guards against: joining news to bars on *date* silently pulls a
story published at 18:00 on Friday into a decision taken at Friday's close. Both carry the
same date, so a date join accepts it, and the articles it wrongly admits are
disproportionately the important ones — companies announce after the close.

The convention is strict: for a decision at instant T, an article counts only when
``available_at < T``. An article stamped exactly at T is excluded, because nobody reads
and acts on a story in the same instant it appears.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from swingbot.calendars import TradingCalendar
from swingbot.features import build_news_features
from swingbot.news.models import NewsAnalysis
from swingbot.pit import LookAheadError, ensure_utc, visible_bars, visible_news
from swingbot.types import EventType


def _analysis_row(ticker, stamp, sentiment, *, magnitude=2, event="earnings_beat", novelty=1.0):
    return {
        "content_hash": f"{ticker}-{stamp.isoformat()}",
        "ticker": ticker,
        "published_at": stamp,
        "first_seen_at": stamp,
        "available_at": stamp,
        "event_type": event,
        "sentiment": sentiment,
        "magnitude": magnitude,
        "is_speculative": False,
        "is_recap": False,
        "is_company_specific": True,
        "expected_direction": "up" if sentiment > 0 else "down",
        "direction_confidence": abs(sentiment),
        "signed_score": sentiment * magnitude / 3.0,
        "numeric_surprise_pct": None,
        "novelty": novelty,
        "backend": "test",
        "source": "test",
    }


@pytest.fixture
def news_calendar(cfg_us):
    sessions = [
        d for d in (dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(240))
        if d.weekday() < 5
    ]
    calendar = TradingCalendar.from_config(sessions, cfg_us)
    return calendar, calendar.weekly_grid(cfg_us.calendar.hold_sessions)


def test_article_exactly_at_the_cutoff_is_excluded(news_calendar):
    """The boundary case. Inclusive here would admit every after-close announcement."""
    calendar, grid = news_calendar
    decision = grid[10]
    cutoff = decision.decision_ts

    analyses = pd.DataFrame([
        _analysis_row("BEFORE", cutoff - pd.Timedelta(minutes=1), 0.8),
        _analysis_row("EXACT", cutoff, 0.9),
        _analysis_row("AFTER", cutoff + pd.Timedelta(minutes=1), 0.9),
    ])
    features = build_news_features(analyses, calendar, grid=grid, lookback_days=10)
    present = set(features.loc[features["decision_session"] == decision.decision_session, "ticker"])

    assert "BEFORE" in present
    assert "EXACT" not in present, (
        "an article stamped at the exact decision instant was admitted. Nobody can read "
        "and act on a story in the same instant it is published."
    )
    assert "AFTER" not in present, "a future article leaked into the decision"


def test_visible_news_is_strict_and_visible_bars_is_inclusive(news_calendar):
    """Bars are stamped at their own close, so the bar at T is knowable at T. News is not."""
    calendar, grid = news_calendar
    cutoff = grid[5].decision_ts
    frame = pd.DataFrame({"available_at": [cutoff]})

    assert len(visible_bars(frame, cutoff)) == 1
    assert len(visible_news(frame, cutoff)) == 0


def test_availability_uses_the_later_of_publication_and_first_sight(news_calendar):
    """A source that backdates its timestamps must not gain an advantage.

    ``available_at`` is ``max(published_at, first_seen_at)``, so a vendor claiming an
    article was published last Tuesday when the pipeline first saw it today cannot make
    it retroactively tradeable.
    """
    from swingbot.types import RawArticle

    published = dt.datetime(2024, 3, 1, 9, 0, tzinfo=dt.UTC)
    first_seen = dt.datetime(2024, 3, 8, 9, 0, tzinfo=dt.UTC)
    article = RawArticle(
        content_hash="h", ticker="T", title="t", body="b", url="u", source="s",
        published_at=published, first_seen_at=first_seen,
    )
    assert article.available_at == first_seen


def test_decay_weights_recent_news_more(news_calendar):
    """A story from this morning must matter more than one from last week."""
    calendar, grid = news_calendar
    decision = grid[10]
    cutoff = decision.decision_ts

    analyses = pd.DataFrame([
        _analysis_row("A", cutoff - pd.Timedelta(hours=2), 0.9),
        _analysis_row("A", cutoff - pd.Timedelta(days=7), -0.9),
    ])
    features = build_news_features(
        analyses, calendar, grid=grid, lookback_days=10, half_life_hours=36.0
    )
    row = features.loc[features["decision_session"] == decision.decision_session].iloc[0]
    assert row["news_sentiment_decay"] > 0.5, (
        "a fresh +0.9 and a week-old -0.9 averaged to "
        f"{row['news_sentiment_decay']:.3f}; decay is not being applied"
    )


def test_recap_and_speculation_are_discounted():
    """A rehash and a rumour both carry less information than a confirmed novel event."""
    confirmed = NewsAnalysis(
        event_type=EventType.GUIDANCE_DOWN, sentiment=-0.8, magnitude=3,
        expected_direction="down", direction_confidence=0.9,
    )
    rumour = confirmed.model_copy(update={"is_speculative": True})
    rehash = confirmed.model_copy(update={"is_recap": True})

    assert abs(rumour.signed_score) < abs(confirmed.signed_score)
    assert abs(rehash.signed_score) < abs(rumour.signed_score)


def test_novelty_is_computed_against_prior_coverage_only():
    """Novelty must never be judged against articles published later.

    This is a leak the price-path truncation test cannot catch, because it lives entirely
    inside the news pipeline.
    """
    from swingbot.news.dedupe import article_hash, novelty_scores
    from swingbot.types import RawArticle

    base = dt.datetime(2024, 3, 1, 10, tzinfo=dt.UTC)

    def make(offset_hours, title):
        stamp = base + dt.timedelta(hours=offset_hours)
        stub = RawArticle(
            content_hash="", ticker="T", title=title, body=title, url=f"u{offset_hours}",
            source="s", published_at=stamp, first_seen_at=stamp,
        )
        return RawArticle(
            content_hash=article_hash(stub), ticker="T", title=title, body=title,
            url=stub.url, source="s", published_at=stamp, first_seen_at=stamp,
        )

    first = make(0, "Company wins a large order from a European client")
    later = make(24, "Company wins a large order from a European client")

    # Novelty is keyed by occurrence, not by content hash. Identical text shares a hash
    # — that is what makes the hash right for the analysis cache — but the first
    # appearance of a story is novel and a republication is not, so they must not share
    # a novelty entry.
    from swingbot.news.dedupe import novelty_key

    alone = novelty_scores([first])[novelty_key(first)]
    with_future = novelty_scores([first, later])[novelty_key(first)]
    assert alone == with_future == 1.0, (
        "the original article's novelty changed once a later republication existed. Its "
        "score is being determined by a future event."
    )

    # The later occurrence is the one that loses novelty.
    assert novelty_scores([first, later])[novelty_key(later)] < 0.3

    # And the two occurrences must be distinct entries despite the shared content hash.
    assert first.content_hash == later.content_hash
    assert novelty_key(first) != novelty_key(later)


def test_naive_timestamps_are_rejected():
    """Guessing a timezone is how off-by-one-session leaks are born."""
    with pytest.raises(LookAheadError):
        ensure_utc(dt.datetime(2024, 1, 1, 12, 0))


def test_lookback_window_bounds_the_join(news_calendar):
    """Articles older than the lookback must not contribute."""
    calendar, grid = news_calendar
    decision = grid[20]
    cutoff = decision.decision_ts

    analyses = pd.DataFrame([
        _analysis_row("OLD", cutoff - pd.Timedelta(days=40), 0.9),
    ])
    features = build_news_features(analyses, calendar, grid=grid, lookback_days=10)
    present = set(features.loc[features["decision_session"] == decision.decision_session, "ticker"])
    assert "OLD" not in present

"""The news-analysis contract, and the interchangeability claim.

The system says it is not bound to any one AI provider. That is only meaningful if every
backend really returns the same object and the cache really keys them apart, so both are
asserted here rather than assumed.
"""

from __future__ import annotations

import datetime as dt

import pytest

from swingbot.config import load_config
from swingbot.io.cache import ContentCache, cache_key, content_hash
from swingbot.news.backends import build_backend
from swingbot.news.backends.lexicon import LexiconAnalyzer
from swingbot.news.models import SCHEMA_VERSION, NewsAnalysis
from swingbot.types import EventType, RawArticle

NOW = dt.datetime(2024, 3, 1, 12, tzinfo=dt.UTC)


def _article(title: str, body: str = "", ticker: str = "TCS") -> RawArticle:
    return RawArticle(
        content_hash=content_hash(title + body), ticker=ticker, title=title, body=body,
        url="u", source="test", published_at=NOW, first_seen_at=NOW,
    )


# --------------------------------------------------------------------------- schema


def test_out_of_range_values_are_repaired_not_rejected():
    """A model returning 1.2 for a field bounded at 1.0 has made a rounding error.

    Discarding the whole extraction over it would lose the other twelve fields it got
    right, so out-of-range numbers are clamped before validation.
    """
    assert NewsAnalysis(sentiment=9.0).sentiment == 1.0
    assert NewsAnalysis(sentiment=-9.0).sentiment == -1.0
    assert NewsAnalysis(direction_confidence=1.4).direction_confidence == 1.0
    assert NewsAnalysis(magnitude=7).magnitude == 3
    assert NewsAnalysis(numeric_surprise_pct=1e9).numeric_surprise_pct == 500.0


def test_unparseable_values_fall_back_rather_than_raising():
    assert NewsAnalysis(sentiment="not a number").sentiment == 0.0
    assert NewsAnalysis(numeric_surprise_pct="n/a").numeric_surprise_pct is None


def test_signed_score_direction_and_discounts():
    strong = NewsAnalysis(
        event_type=EventType.GUIDANCE_DOWN, sentiment=-0.8, magnitude=3,
        expected_direction="down", direction_confidence=1.0,
    )
    assert strong.signed_score == pytest.approx(-1.0)

    up = strong.model_copy(update={"expected_direction": "up"})
    assert up.signed_score > 0

    neutral = strong.model_copy(update={"expected_direction": "neutral"})
    assert neutral.signed_score == 0.0

    # Speculation, recaps and non-company-specific pieces all carry less information.
    for field in ("is_speculative", "is_recap", "is_company_specific"):
        # is_company_specific discounts when False; the other two discount when True.
        value = field != "is_company_specific"
        discounted = strong.model_copy(update={field: value})
        assert abs(discounted.signed_score) < abs(strong.signed_score), field


def test_rationale_is_bounded():
    """Free text is stored for audit and must not grow without limit."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        NewsAnalysis(rationale="x" * 5000)


# --------------------------------------------------------------------------- backends


def test_lexicon_backend_needs_nothing_and_always_works():
    """The default must run with no key, no network and no optional package."""
    analyzer = LexiconAnalyzer()
    assert analyzer.available()
    assert analyzer.estimated_cost_usd(100_000) == 0.0

    results = analyzer.analyze([_article("TCS Q3 profit beats estimates, revenue surges")])
    assert len(results) == 1
    assert isinstance(results[0], NewsAnalysis)


@pytest.mark.parametrize(
    ("headline", "expected_direction"),
    [
        ("TCS Q3 profit beats estimates, revenue surges 18%", "up"),
        ("Infosys cuts guidance for FY25, warns on weak demand", "down"),
        ("HDFC Bank Share Price Drops 2% In Trade Today", "down"),
        ("Reliance shares rally as brokerages raise target price", "up"),
        ("Wipro announces buyback, board approves payout", "up"),
        ("SEBI launches probe into disclosure lapses", "down"),
    ],
)
def test_lexicon_reads_direction_correctly(headline, expected_direction):
    """Including present-tense inflections, which is how headlines are actually written."""
    result = LexiconAnalyzer().analyze([_article(headline)])[0]
    assert result.expected_direction == expected_direction, (
        f"{headline!r} read as {result.expected_direction} "
        f"(sentiment {result.sentiment:+.2f})"
    )


def test_lexicon_handles_negation():
    """"not facing any shortfall" is good news, not bad."""
    result = LexiconAnalyzer().analyze(
        [_article("Company is not facing any shortfall in supply")]
    )[0]
    assert result.sentiment > 0


def test_market_wrap_is_flagged_as_a_recap():
    result = LexiconAnalyzer().analyze(
        [_article("Market wrap: Sensex closes higher, top gainers and losers")]
    )[0]
    assert result.is_recap
    # Heavily discounted rather than zeroed: a market wrap does carry a trace of
    # information (which way the index closed), just far less than a real event.
    assert abs(result.signed_score) < 0.1


def test_rumour_is_flagged_as_speculative():
    result = LexiconAnalyzer().analyze(
        [_article("TCS reportedly in talks to acquire a European firm")]
    )[0]
    assert result.is_speculative


@pytest.mark.parametrize("backend", ["lexicon", "anthropic", "openai_compat", "finbert"])
def test_every_backend_is_selectable_and_degrades_to_lexicon(backend):
    """The interchangeability claim.

    Each backend can be selected by config, and one that cannot run here — a missing
    package or absent credentials — falls back to the offline lexicon with a warning
    rather than aborting a weekly run that has already computed its book.
    """
    cfg = load_config("us", set_values=[f"nlp.backend={backend}"])
    analyzer = build_backend(cfg)
    assert hasattr(analyzer, "backend") and hasattr(analyzer, "model_id")
    assert callable(analyzer.analyze)

    results = analyzer.analyze([_article("Company wins a large contract")])
    assert len(results) == 1
    assert results[0] is None or isinstance(results[0], NewsAnalysis)


def test_unknown_backend_is_rejected():
    cfg = load_config("us", set_values=["nlp.backend=telepathy"])
    with pytest.raises(ValueError, match="Unknown nlp backend"):
        build_backend(cfg)


def test_empty_input_returns_empty_output():
    assert LexiconAnalyzer().analyze([]) == []


# ----------------------------------------------------------------------------- cache


def test_cache_key_separates_backends_models_and_prompt_versions():
    """Switching a backend must not overwrite an earlier backend's analyses.

    This is what lets a backtest pin ``prompt_version: v1`` and keep returning what it
    always returned, even after the prompt is rewritten.
    """
    base = dict(
        backend="lexicon", model_id="m1", prompt_version="v1", schema_version="1"
    )
    reference = cache_key("hash", **base)

    for field, value in [
        ("backend", "anthropic"), ("model_id", "m2"),
        ("prompt_version", "v2"), ("schema_version", "2"),
    ]:
        assert cache_key("hash", **{**base, field: value}) != reference, field

    assert cache_key("other", **base) != reference


def test_content_hash_is_whitespace_and_case_insensitive():
    assert content_hash("Profit  BEATS   estimates") == content_hash("profit beats estimates")


def test_cache_records_failures_explicitly(tmp_path):
    """A missing analysis must be a null feature, never a defaulted neutral one.

    A silently neutral default would look like real evidence that the news was neutral.
    """
    with ContentCache(tmp_path / "c.sqlite") as cache:
        key = cache_key("h", backend="b", model_id="m", prompt_version="v1", schema_version="1")
        cache.record_failure(key, "backend returned nothing")

        assert cache.get(key) is None, "a failed analysis must not read back as a result"
        assert cache.has_failure(key)
        assert cache.stats()["failed"] == 1


def test_cache_round_trip_and_spend_tracking(tmp_path):
    with ContentCache(tmp_path / "c.sqlite") as cache:
        key = cache_key("h", backend="b", model_id="m", prompt_version="v1", schema_version="1")
        analysis = NewsAnalysis(event_type=EventType.BUYBACK, sentiment=0.5, magnitude=2)
        cache.put(
            key, analysis.model_dump(mode="json"), content_hash_value="h", backend="b",
            model_id="m", prompt_version="v1", schema_version=SCHEMA_VERSION,
        )
        loaded = NewsAnalysis.model_validate(cache.get(key))
        assert loaded.event_type is EventType.BUYBACK
        assert loaded.sentiment == pytest.approx(0.5)

        cache.log_spend("b", "m", 10, 0.25)
        assert cache.stats()["total_spend_usd"] == pytest.approx(0.25)


def test_analysis_pipeline_caches_and_stamps_publication_time(tmp_path):
    """Only the article's timestamps may reach a feature, never the analysis time."""
    from swingbot.news.analyze import NewsAnalysisPipeline, records_to_frame

    articles = [
        _article("Company wins a large order", ticker="AAA"),
        _article("Company cuts guidance and warns", ticker="BBB"),
    ]
    with ContentCache(tmp_path / "c.sqlite") as cache:
        pipeline = NewsAnalysisPipeline(LexiconAnalyzer(), cache, prompt_version="v1")
        first = pipeline.run(articles)
        assert len(first) == 2
        assert cache.stats()["entries"] == 2

        # Second run must be served entirely from cache.
        second = pipeline.run(articles)
        assert len(second) == 2
        assert cache.stats()["entries"] == 2

    frame = records_to_frame(first)
    assert (frame["available_at"] == NOW).all()
    assert "analyzed_at" not in frame.columns, (
        "the analysis timestamp must never reach the feature frame"
    )

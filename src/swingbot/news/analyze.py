"""The analysis pipeline: raw articles in, cached extraction records out.

Sits between the news providers and the feature layer, and owns three responsibilities
that would otherwise be duplicated in every backend.

**Caching.** Every analysis is keyed by content hash plus backend, model, prompt version
and schema version. Re-running costs nothing for articles already seen, which matters
because a ten-year backfill is the largest single expense in this project and nobody gets
their feature engineering right on the first attempt.

**Point-in-time stamping.** The stored record carries the article's ``published_at`` and
``first_seen_at``, and separately the wall-clock ``analyzed_at``. Only the first two ever
reach a feature. That separation is what allows an analysis run today to feed a backtest
of 2019 without contaminating it.

**Explicit failure.** A backend that fails on an article records a failure rather than a
neutral default. Downstream that becomes a null news feature, which is honest. A silently
defaulted neutral analysis would look like real evidence that the news was neutral.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import UTC, datetime

import pandas as pd

from ..config import Config
from ..io.cache import ContentCache, cache_key
from ..types import RawArticle
from .dedupe import deduplicate, novelty_key, novelty_scores
from .models import SCHEMA_VERSION, ExtractionRecord, NewsAnalysis

log = logging.getLogger(__name__)


class NewsAnalysisPipeline:
    """Deduplicate, score novelty, analyse with caching, persist."""

    def __init__(
        self,
        analyzer,
        cache: ContentCache,
        *,
        prompt_version: str = "v1",
        batch_size: int = 20,
        max_articles: int = 2000,
        novelty_lookback_days: float = 7.0,
    ) -> None:
        self.analyzer = analyzer
        self.cache = cache
        self.prompt_version = prompt_version
        self.batch_size = batch_size
        self.max_articles = max_articles
        self.novelty_lookback_days = novelty_lookback_days

    @classmethod
    def from_config(cls, cfg: Config, analyzer=None) -> NewsAnalysisPipeline:
        from .backends import build_backend

        cache = ContentCache(cfg.market_dir / "news_cache.sqlite")
        return cls(
            analyzer or build_backend(cfg),
            cache,
            prompt_version=cfg.nlp.prompt_version,
            batch_size=cfg.nlp.batch_size,
            max_articles=cfg.nlp.max_articles_per_run,
            novelty_lookback_days=cfg.features.news.lookback_days,
        )

    # ---------------------------------------------------------------------- analysis

    def run(
        self, articles: Sequence[RawArticle], *, dedupe: bool = True
    ) -> list[ExtractionRecord]:
        if not articles:
            return []

        working = deduplicate(articles) if dedupe else list(articles)
        novelty = novelty_scores(working, lookback_days=self.novelty_lookback_days)

        keys = {
            a.content_hash: cache_key(
                a.content_hash,
                backend=self.analyzer.backend,
                model_id=self.analyzer.model_id,
                prompt_version=self.prompt_version,
                schema_version=SCHEMA_VERSION,
            )
            for a in working
        }
        cached = self.cache.get_many(keys.values())

        pending = [
            a
            for a in working
            if keys[a.content_hash] not in cached
            and not self.cache.has_failure(keys[a.content_hash])
        ]
        log.info(
            "%d article(s): %d cached, %d to analyse with %s",
            len(working), len(working) - len(pending), len(pending), self.analyzer.backend,
        )

        if len(pending) > self.max_articles:
            log.warning(
                "capping this run at %d of %d pending article(s)",
                self.max_articles, len(pending),
            )
            pending = pending[: self.max_articles]

        self._analyze_pending(pending, keys)

        # Re-read so freshly analysed items join the cached ones in one place.
        cached = self.cache.get_many(keys.values())

        records: list[ExtractionRecord] = []
        for article in working:
            payload = cached.get(keys[article.content_hash])
            if payload is None:
                continue
            try:
                analysis = NewsAnalysis.model_validate(payload)
            except Exception:
                continue
            records.append(
                ExtractionRecord(
                    content_hash=article.content_hash,
                    ticker=article.ticker,
                    published_at=article.published_at,
                    first_seen_at=article.first_seen_at,
                    analyzed_at=datetime.now(UTC),
                    backend=self.analyzer.backend,
                    model_id=self.analyzer.model_id,
                    prompt_version=self.prompt_version,
                    schema_version=SCHEMA_VERSION,
                    analysis=analysis,
                    source=article.source,
                    url=article.url,
                    novelty=float(novelty.get(novelty_key(article), 1.0)),
                )
            )
        return records

    def _analyze_pending(self, pending: list[RawArticle], keys: dict[str, str]) -> None:
        for start in range(0, len(pending), self.batch_size):
            chunk = pending[start : start + self.batch_size]
            try:
                results = self.analyzer.analyze(chunk)
            except Exception as exc:
                log.warning("analyzer raised on a batch of %d: %s", len(chunk), exc)
                results = [None] * len(chunk)

            for article, analysis in zip(chunk, results, strict=False):
                key = keys[article.content_hash]
                if analysis is None:
                    self.cache.record_failure(
                        key,
                        "analyzer returned no result",
                        content_hash_value=article.content_hash,
                        backend=self.analyzer.backend,
                        model_id=self.analyzer.model_id,
                        prompt_version=self.prompt_version,
                        schema_version=SCHEMA_VERSION,
                    )
                    continue
                self.cache.put(
                    key,
                    analysis.model_dump(mode="json"),
                    content_hash_value=article.content_hash,
                    backend=self.analyzer.backend,
                    model_id=self.analyzer.model_id,
                    prompt_version=self.prompt_version,
                    schema_version=SCHEMA_VERSION,
                )

    # ------------------------------------------------------------------- persistence

    def close(self) -> None:
        self.cache.close()

    def stats(self) -> dict:
        return self.cache.stats()


def records_to_frame(records: Sequence[ExtractionRecord]) -> pd.DataFrame:
    """Flatten extraction records into the frame the feature layer consumes.

    ``available_at`` is computed here, once, as the later of publication and first sight,
    so every downstream consumer uses the same definition of when a story became usable.
    """
    if not records:
        return pd.DataFrame()

    rows = []
    for record in records:
        analysis = record.analysis
        rows.append(
            {
                "content_hash": record.content_hash,
                "ticker": record.ticker,
                "published_at": record.published_at,
                "first_seen_at": record.first_seen_at,
                "available_at": record.available_at,
                "event_type": analysis.event_type.value,
                "sentiment": analysis.sentiment,
                "magnitude": analysis.magnitude,
                "is_speculative": analysis.is_speculative,
                "is_recap": analysis.is_recap,
                "is_company_specific": analysis.is_company_specific,
                "expected_direction": analysis.expected_direction,
                "direction_confidence": analysis.direction_confidence,
                "signed_score": analysis.signed_score,
                "numeric_surprise_pct": analysis.numeric_surprise_pct,
                "novelty": record.novelty,
                "backend": record.backend,
                "source": record.source,
            }
        )
    frame = pd.DataFrame(rows)
    frame["available_at"] = pd.to_datetime(frame["available_at"], utc=True)
    frame["published_at"] = pd.to_datetime(frame["published_at"], utc=True)
    return frame.sort_values(["ticker", "available_at"]).reset_index(drop=True)

"""News ingest, deduplication, and provider-agnostic analysis."""

from .analyze import NewsAnalysisPipeline, records_to_frame
from .backends import build_backend
from .dedupe import (
    article_hash,
    deduplicate,
    novelty_key,
    novelty_scores,
    summarise_coverage,
)
from .models import SCHEMA_VERSION, EntityMention, ExtractionRecord, NewsAnalysis
from .protocols import NewsAnalyzer, NewsProvider
from .providers import JSONLNewsProvider, RSSNewsProvider, build_provider, write_jsonl
from .taxonomy import PROMPT_VERSION, SYSTEM_PROMPT

__all__ = [
    "PROMPT_VERSION",
    "SCHEMA_VERSION",
    "SYSTEM_PROMPT",
    "EntityMention",
    "ExtractionRecord",
    "JSONLNewsProvider",
    "NewsAnalysis",
    "NewsAnalysisPipeline",
    "NewsAnalyzer",
    "NewsProvider",
    "RSSNewsProvider",
    "article_hash",
    "build_backend",
    "build_provider",
    "deduplicate",
    "novelty_key",
    "novelty_scores",
    "records_to_frame",
    "summarise_coverage",
    "write_jsonl",
]

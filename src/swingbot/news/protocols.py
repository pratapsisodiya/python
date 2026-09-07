"""News-layer protocols."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import Protocol, runtime_checkable

from ..types import RawArticle
from .models import NewsAnalysis


@runtime_checkable
class NewsProvider(Protocol):
    """Source of raw articles."""

    name: str

    def fetch(
        self, tickers: Sequence[str], start: datetime, end: datetime
    ) -> Iterable[RawArticle]: ...

    def available(self) -> bool: ...


@runtime_checkable
class NewsAnalyzer(Protocol):
    """Turns article text into structured facts.

    The seam that keeps this system from being tied to any one AI provider. Four
    implementations ship — an offline lexicon, any OpenAI-compatible endpoint, Anthropic,
    and a local transformer — and all four return the same object, so swapping one
    changes nothing downstream.
    """

    backend: str
    model_id: str

    def analyze(self, articles: Sequence[RawArticle]) -> list[NewsAnalysis | None]:
        """One analysis per article, or None where extraction failed.

        Failure is explicit rather than defaulted. A None becomes a null news feature,
        which is honest; a silently-defaulted neutral analysis would look like real
        evidence of neutrality.
        """
        ...

    def available(self) -> bool: ...

    def estimated_cost_usd(self, n_articles: int) -> float: ...

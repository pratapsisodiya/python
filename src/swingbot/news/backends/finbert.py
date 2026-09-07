"""Local transformer backend (FinBERT or any compatible sentiment classifier).

Offline and free once the weights are downloaded, and materially better than the lexicon
at reading sentiment from unusual phrasings. It classifies sentiment only, so event type
still comes from the lexicon's keyword rules — a deliberate hybrid rather than a
pretence that a three-class sentiment model can identify a rights issue.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from ...types import RawArticle
from ..models import NewsAnalysis
from .lexicon import LexiconAnalyzer

log = logging.getLogger(__name__)


class FinbertAnalyzer:
    """Transformer sentiment, lexicon event classification."""

    backend = "finbert"

    def __init__(self, *, model: str = "ProsusAI/finbert", batch_size: int = 16) -> None:
        self.model_id = model
        self.batch_size = batch_size
        self._pipeline = None
        self._lexicon = LexiconAnalyzer()

    def available(self) -> bool:
        try:
            import transformers  # noqa: F401
        except ImportError:
            return False
        return True

    def estimated_cost_usd(self, n_articles: int) -> float:  # noqa: ARG002
        return 0.0

    def _ensure_pipeline(self):
        if self._pipeline is None:
            from transformers import pipeline

            self._pipeline = pipeline(
                "text-classification", model=self.model_id, truncation=True, max_length=512
            )
        return self._pipeline

    def analyze(self, articles: Sequence[RawArticle]) -> list[NewsAnalysis | None]:
        if not articles:
            return []

        # Start from the lexicon result so event type, speculation and recap flags are
        # populated, then overwrite sentiment with the transformer's reading.
        base = self._lexicon.analyze(articles)
        try:
            classifier = self._ensure_pipeline()
        except Exception as exc:
            log.warning("finbert unavailable, falling back to lexicon only: %s", exc)
            return base

        texts = [f"{a.title}. {a.body[:1500]}" for a in articles]
        out: list[NewsAnalysis | None] = []
        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start : start + self.batch_size]
            try:
                scores = classifier(chunk)
            except Exception as exc:
                log.warning("finbert batch failed: %s", exc)
                out.extend(base[start : start + self.batch_size])
                continue
            for offset, score in enumerate(scores):
                analysis = base[start + offset]
                if analysis is None:
                    out.append(None)
                    continue
                out.append(self._merge(analysis, score))
        return out

    @staticmethod
    def _merge(analysis: NewsAnalysis, score: dict) -> NewsAnalysis:
        label = str(score.get("label", "")).lower()
        confidence = float(score.get("score", 0.0))
        sentiment = (
            confidence if label.startswith("pos")
            else -confidence if label.startswith("neg")
            else 0.0
        )
        direction = "up" if sentiment > 0.15 else "down" if sentiment < -0.15 else "neutral"
        return analysis.model_copy(
            update={
                "sentiment": sentiment,
                "expected_direction": direction,
                "direction_confidence": round(abs(sentiment), 3),
                "rationale": f"finbert {label} {confidence:.2f}; {analysis.rationale}",
            }
        )

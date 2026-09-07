"""The news-analysis contract.

``NewsAnalysis`` is a single Pydantic model used for three things at once: the schema an
LLM is asked to fill, the validation applied to whatever it returns, and the record
persisted to disk. Keeping them one definition means a backend cannot drift from storage,
and adding a field is one edit rather than three.

The field set is chosen to be **extractable from the document alone**. Every question here
can be answered by reading the article, without knowing what the stock did afterwards.
That is the line that separates a reliable extraction task from a forecasting task
disguised as one, and it is why there is no ``expected_return`` field: a language model
asked for one would answer from what it remembers about the outcome.

Note what ``rationale`` is for. It is stored for audit so a human can check the extraction
made sense, and it is never a feature. Free text in a feature matrix is an invitation to
overfit.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..types import EntityRole, EventType

#: Bumped when the field set changes. Part of the cache key, so old analyses stay valid
#: under their old schema instead of being silently reinterpreted.
SCHEMA_VERSION = "1"


class EntityMention(BaseModel):
    model_config = ConfigDict(extra="ignore")

    ticker: str
    role: EntityRole = EntityRole.PRIMARY


class NewsAnalysis(BaseModel):
    """Structured facts extracted from one article."""

    model_config = ConfigDict(extra="ignore")

    event_type: EventType = EventType.OTHER
    #: Tone toward the primary entity, -1 hostile to +1 favourable.
    sentiment: float = Field(default=0.0, ge=-1.0, le=1.0)
    #: Materiality: 0 none, 1 minor, 2 notable, 3 major.
    magnitude: Literal[0, 1, 2, 3] = 0
    #: "Reportedly", "in talks", "could" — rumour rather than fact.
    is_speculative: bool = False
    #: Restates information already public, so it carries no new content.
    is_recap: bool = False
    #: About this company, rather than a macro piece that merely names it.
    is_company_specific: bool = True
    expected_direction: Literal["up", "down", "neutral"] = "neutral"
    direction_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    entities: list[EntityMention] = Field(default_factory=list)
    #: Only when the article itself states actual versus expected.
    numeric_surprise_pct: float | None = None
    #: Audit trail. Never a feature.
    rationale: str = Field(default="", max_length=500)

    # Clamping runs in "before" mode, ahead of the range constraints declared above.
    # The constraints stay because they are what the JSON schema shows a language model,
    # and telling it the valid range is most of what keeps output in range. But a model
    # that returns 1.2 for a field bounded at 1.0 has made a rounding error, not a
    # category error, and discarding the whole extraction over it loses the other twelve
    # fields it got right. Repair, then validate.
    @field_validator("sentiment", mode="before")
    @classmethod
    def _clamp_sentiment(cls, v):
        try:
            return float(max(-1.0, min(1.0, float(v))))
        except (TypeError, ValueError):
            return 0.0

    @field_validator("direction_confidence", mode="before")
    @classmethod
    def _clamp_confidence(cls, v):
        try:
            return float(max(0.0, min(1.0, float(v))))
        except (TypeError, ValueError):
            return 0.0

    @field_validator("magnitude", mode="before")
    @classmethod
    def _clamp_magnitude(cls, v):
        try:
            return int(max(0, min(3, round(float(v)))))
        except (TypeError, ValueError):
            return 0

    @field_validator("numeric_surprise_pct", mode="before")
    @classmethod
    def _sane_surprise(cls, v):
        if v is None:
            return None
        try:
            # A model reporting a 4000 percent surprise has misread a table.
            return float(max(-500.0, min(500.0, float(v))))
        except (TypeError, ValueError):
            return None

    @property
    def signed_score(self) -> float:
        """A single directional number combining direction, confidence and materiality.

        Discounted for speculation and for recaps, because a rumour and a rehash both
        carry less information than a confirmed, novel event, and the aggregation layer
        should not have to rediscover that.
        """
        direction = {"up": 1.0, "down": -1.0, "neutral": 0.0}[self.expected_direction]
        score = direction * self.direction_confidence * (self.magnitude / 3.0)
        if self.is_speculative:
            score *= 0.5
        if self.is_recap:
            score *= 0.3
        if not self.is_company_specific:
            score *= 0.4
        return float(score)


class ExtractionRecord(BaseModel):
    """A stored analysis with the timestamps that make it point-in-time safe."""

    model_config = ConfigDict(extra="ignore")

    content_hash: str
    ticker: str
    #: What the source claims. Drives availability.
    published_at: datetime
    #: When this pipeline first saw it. Also drives availability.
    first_seen_at: datetime
    #: Wall clock of the analysis. Audit only — never reaches a feature, which is what
    #: lets an analysis run today feed a backtest of 2019 without contaminating it.
    analyzed_at: datetime
    backend: str
    model_id: str
    prompt_version: str
    schema_version: str = SCHEMA_VERSION
    analysis: NewsAnalysis
    source: str = ""
    url: str = ""
    #: Novelty against the same ticker's recent coverage, 1.0 = wholly new.
    novelty: float = 1.0

    @property
    def available_at(self) -> datetime:
        return max(self.published_at, self.first_seen_at)


def analysis_json_schema() -> dict:
    """JSON schema for the LLM backends that accept one."""
    return NewsAnalysis.model_json_schema()

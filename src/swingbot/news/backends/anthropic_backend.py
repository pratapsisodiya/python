"""Anthropic (Claude) analyzer backend.

One of four interchangeable backends. Nothing in this file is load-bearing for the rest
of the system: the news layer talks to the ``NewsAnalyzer`` protocol, so this can be
swapped for the OpenAI-compatible backend, a local transformer or the offline lexicon
without touching a single line downstream.

Two implementation notes worth stating.

**Structured output, not free text.** The request pins the response to the
:class:`NewsAnalysis` Pydantic model, so what comes back is validated against the schema
rather than parsed out of prose. A model that returns a malformed field fails loudly here
instead of producing a plausible-looking feature that is subtly wrong.

**Batch for backfill, sync for the weekly run.** Analysing ten years of news is the single
largest cost in this project, and the Batches API does it at half price. The weekly run is
a few hundred articles and wants an answer now, so it goes through the synchronous path.
Batch results come back unordered, so they are keyed by ``custom_id``, never by position.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Sequence

from ...types import RawArticle
from ..models import NewsAnalysis
from ..taxonomy import SYSTEM_PROMPT, user_prompt

log = logging.getLogger(__name__)

#: Per-million-token prices used only for the spend estimate and cap. Update alongside
#: the model default; being approximately right is enough to stop a runaway backfill.
_PRICING = {
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-fable-5-1": (10.0, 50.0),
}
_DEFAULT_PRICING = (5.0, 25.0)

#: Rough token counts for one article: the cached system prompt is excluded because it is
#: charged at a tenth of the rate after the first call.
_EST_INPUT_TOKENS = 900
_EST_OUTPUT_TOKENS = 220


class AnthropicAnalyzer:
    """Claude-backed extraction with caching, retry and a spend cap."""

    backend = "anthropic"

    def __init__(
        self,
        *,
        model: str = "claude-opus-5",
        api_key_env: str = "ANTHROPIC_API_KEY",
        max_retries: int = 3,
        timeout: float = 60.0,
        max_spend_usd: float = 5.0,
    ) -> None:
        self.model_id = model
        self.api_key_env = api_key_env
        self.max_retries = max_retries
        self.timeout = timeout
        self.max_spend_usd = max_spend_usd
        self._client = None
        self._spent = 0.0

    # ------------------------------------------------------------------ availability

    def available(self) -> bool:
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False
        # An unset key does not mean no credentials: the SDK also resolves an
        # `ant auth login` profile. Constructing the client is the honest check.
        try:
            self._ensure_client()
        except Exception:
            return False
        return True

    def _ensure_client(self):
        if self._client is None:
            import anthropic

            key = os.environ.get(self.api_key_env)
            self._client = (
                anthropic.Anthropic(api_key=key, timeout=self.timeout, max_retries=self.max_retries)
                if key
                else anthropic.Anthropic(timeout=self.timeout, max_retries=self.max_retries)
            )
        return self._client

    def estimated_cost_usd(self, n_articles: int) -> float:
        inp, out = _PRICING.get(self.model_id, _DEFAULT_PRICING)
        return n_articles * (
            _EST_INPUT_TOKENS * inp / 1e6 + _EST_OUTPUT_TOKENS * out / 1e6
        )

    # ----------------------------------------------------------------------- analyze

    def analyze(self, articles: Sequence[RawArticle]) -> list[NewsAnalysis | None]:
        if not articles:
            return []

        projected = self.estimated_cost_usd(len(articles))
        if self._spent + projected > self.max_spend_usd:
            affordable = self._affordable_count()
            log.warning(
                "spend cap %.2f USD would be exceeded by %d articles (est %.2f); "
                "analysing the first %d only",
                self.max_spend_usd,
                len(articles),
                projected,
                affordable,
            )
            articles = list(articles)[:affordable]
            if not articles:
                return [None] * len(articles)

        client = self._ensure_client()
        out: list[NewsAnalysis | None] = []
        for article in articles:
            out.append(self._analyze_one(client, article))
        return out

    def _affordable_count(self) -> int:
        remaining = max(0.0, self.max_spend_usd - self._spent)
        per_article = max(self.estimated_cost_usd(1), 1e-9)
        return int(remaining / per_article)

    def _analyze_one(self, client, article: RawArticle) -> NewsAnalysis | None:
        request = self._request_kwargs(article)

        for attempt in range(self.max_retries + 1):
            try:
                # messages.parse pins the response to the Pydantic schema and validates
                # it, so a malformed field raises here rather than becoming a quietly
                # wrong feature.
                response = client.messages.parse(**request, output_format=NewsAnalysis)

                if getattr(response, "stop_reason", None) == "refusal":
                    details = getattr(response, "stop_details", None)
                    log.warning(
                        "refusal on %s (%s)",
                        article.content_hash[:12],
                        getattr(details, "category", "unknown"),
                    )
                    return None

                self._record_spend(response)
                parsed = getattr(response, "parsed_output", None)
                if parsed is not None:
                    return parsed
                return self._parse_text(response)

            except AttributeError:
                # An older SDK without messages.parse. Fall back to a plain create with
                # a JSON schema and parse the text ourselves.
                return self._analyze_via_create(client, request, article)
            except Exception as exc:
                if attempt >= self.max_retries:
                    log.warning("analysis failed for %s: %s", article.content_hash[:12], exc)
                    return None
                time.sleep(1.5 * (attempt + 1))
        return None

    def _request_kwargs(self, article: RawArticle) -> dict:
        return {
            "model": self.model_id,
            "max_tokens": 1024,
            # The taxonomy is frozen text and identical on every call, so it is the
            # cacheable prefix. The article, which changes every time, comes after it.
            "system": [
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            # Classification, not reasoning. Low effort is both cheaper and better
            # suited: the answer is in the document, not in a chain of deduction.
            "output_config": {"effort": "low"},
            "messages": [
                {
                    "role": "user",
                    "content": user_prompt(article.ticker, article.title, article.body),
                }
            ],
        }

    def _analyze_via_create(self, client, request: dict, article: RawArticle):
        try:
            payload = dict(request)
            payload["output_config"] = {
                **payload.get("output_config", {}),
                "format": {
                    "type": "json_schema",
                    "schema": NewsAnalysis.model_json_schema(),
                },
            }
            response = client.messages.create(**payload)
            self._record_spend(response)
            return self._parse_text(response)
        except Exception as exc:
            log.warning("fallback analysis failed for %s: %s", article.content_hash[:12], exc)
            return None

    @staticmethod
    def _parse_text(response) -> NewsAnalysis | None:
        for block in getattr(response, "content", []) or []:
            if getattr(block, "type", None) == "text":
                try:
                    return NewsAnalysis.model_validate(json.loads(block.text))
                except Exception:
                    continue
        return None

    def _record_spend(self, response) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            self._spent += self.estimated_cost_usd(1)
            return
        inp, out = _PRICING.get(self.model_id, _DEFAULT_PRICING)
        # Cached reads are charged at roughly a tenth, which is the whole point of
        # putting the frozen taxonomy behind a cache breakpoint.
        cached = getattr(usage, "cache_read_input_tokens", 0) or 0
        fresh = getattr(usage, "input_tokens", 0) or 0
        written = getattr(usage, "cache_creation_input_tokens", 0) or 0
        output = getattr(usage, "output_tokens", 0) or 0
        self._spent += (
            fresh * inp / 1e6
            + written * inp * 1.25 / 1e6
            + cached * inp * 0.1 / 1e6
            + output * out / 1e6
        )

    @property
    def spent_usd(self) -> float:
        return self._spent

    # ------------------------------------------------------------------------- batch

    def analyze_batch(
        self, articles: Sequence[RawArticle], *, poll_seconds: float = 30.0,
        max_wait_seconds: float = 86_400.0,
    ) -> dict[str, NewsAnalysis | None]:
        """Submit a batch and wait for it. Half the cost, for historical backfill.

        Returns a mapping keyed by content hash. Batch results arrive in arbitrary
        order, so they are matched by ``custom_id`` and never by position.
        """
        if not articles:
            return {}
        client = self._ensure_client()

        requests = []
        for article in articles:
            payload = self._request_kwargs(article)
            payload["output_config"] = {
                **payload.get("output_config", {}),
                "format": {
                    "type": "json_schema",
                    "schema": NewsAnalysis.model_json_schema(),
                },
            }
            requests.append(
                {"custom_id": article.content_hash[:64], "params": payload}
            )

        batch = client.messages.batches.create(requests=requests)
        log.info("submitted batch %s with %d article(s)", batch.id, len(requests))

        waited = 0.0
        while waited < max_wait_seconds:
            status = client.messages.batches.retrieve(batch.id)
            if status.processing_status == "ended":
                break
            time.sleep(poll_seconds)
            waited += poll_seconds
        else:
            log.warning("batch %s did not finish within the wait limit", batch.id)
            return {}

        out: dict[str, NewsAnalysis | None] = {}
        for entry in client.messages.batches.results(batch.id):
            key = entry.custom_id
            if entry.result.type != "succeeded":
                out[key] = None
                continue
            out[key] = self._parse_text(entry.result.message)
        return out

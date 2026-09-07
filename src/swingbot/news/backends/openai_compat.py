"""OpenAI-compatible analyzer backend.

Covers a large share of the ecosystem in one file, because the OpenAI chat-completions
shape has become the de facto standard: OpenAI itself, Groq, Together, OpenRouter,
DeepInfra, Fireworks, and locally-run servers such as Ollama and LM Studio all speak it.
Pointing ``base_url`` at any of them is the whole configuration.

That matters for this project specifically. Running a local model through Ollama means the
news layer costs nothing per article and needs no network, which makes a ten-year backfill
practical on a laptop. It will be less accurate than a frontier model; the ablation is
what tells you whether that difference is worth paying for.

Uses the HTTP API directly through ``httpx`` rather than the ``openai`` package, so the
backend works against any compatible endpoint without pulling in a vendor SDK. Structured
output is requested through ``response_format`` where supported, with a plain
JSON-extraction fallback for the many endpoints that ignore it.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections.abc import Sequence

from ...types import RawArticle
from ..models import NewsAnalysis
from ..taxonomy import SYSTEM_PROMPT, user_prompt

log = logging.getLogger(__name__)

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


class OpenAICompatAnalyzer:
    """Any OpenAI-shaped chat-completions endpoint."""

    backend = "openai_compat"

    def __init__(
        self,
        *,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        api_key_env: str = "OPENAI_API_KEY",
        max_retries: int = 3,
        timeout: float = 60.0,
        max_spend_usd: float = 5.0,
        price_per_million_input: float = 0.15,
        price_per_million_output: float = 0.60,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_id = model
        self.api_key_env = api_key_env
        self.max_retries = max_retries
        self.timeout = timeout
        self.max_spend_usd = max_spend_usd
        self.price_in = price_per_million_input
        self.price_out = price_per_million_output
        self._spent = 0.0

    # ------------------------------------------------------------------ availability

    def available(self) -> bool:
        try:
            import httpx  # noqa: F401
        except ImportError:
            return False
        # A local server needs no key, so a missing key is only disqualifying when the
        # endpoint is a hosted one.
        if self._is_local():
            return True
        return bool(os.environ.get(self.api_key_env))

    def _is_local(self) -> bool:
        return any(
            marker in self.base_url
            for marker in ("localhost", "127.0.0.1", "0.0.0.0", "host.docker.internal")
        )

    def estimated_cost_usd(self, n_articles: int) -> float:
        if self._is_local():
            return 0.0
        return n_articles * (900 * self.price_in / 1e6 + 220 * self.price_out / 1e6)

    @property
    def spent_usd(self) -> float:
        return self._spent

    # ----------------------------------------------------------------------- analyze

    def analyze(self, articles: Sequence[RawArticle]) -> list[NewsAnalysis | None]:
        if not articles:
            return []
        import httpx

        projected = self.estimated_cost_usd(len(articles))
        if projected > 0 and self._spent + projected > self.max_spend_usd:
            affordable = int(
                max(0.0, self.max_spend_usd - self._spent)
                / max(self.estimated_cost_usd(1), 1e-9)
            )
            log.warning(
                "spend cap %.2f USD reached; analysing %d of %d article(s)",
                self.max_spend_usd, affordable, len(articles),
            )
            articles = list(articles)[:affordable]

        headers = {"Content-Type": "application/json"}
        key = os.environ.get(self.api_key_env)
        if key:
            headers["Authorization"] = f"Bearer {key}"

        out: list[NewsAnalysis | None] = []
        with httpx.Client(timeout=self.timeout, headers=headers) as client:
            for article in articles:
                out.append(self._analyze_one(client, article))
        return out

    def _analyze_one(self, client, article: RawArticle) -> NewsAnalysis | None:
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": user_prompt(article.ticker, article.title, article.body)
                    + "\n\nRespond with a single JSON object matching the required schema.",
                },
            ],
            "temperature": 0.0,
            "max_tokens": 800,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "news_analysis",
                    "strict": False,
                    "schema": NewsAnalysis.model_json_schema(),
                },
            },
        }

        for attempt in range(self.max_retries + 1):
            try:
                response = client.post(f"{self.base_url}/chat/completions", json=payload)
                if response.status_code == 400 and "response_format" in response.text:
                    # Many compatible servers, especially local ones, reject the schema
                    # form. Drop to plain JSON mode and extract from the text.
                    payload.pop("response_format", None)
                    payload["messages"][-1]["content"] += (
                        " Output only the JSON object, with no commentary or code fence."
                    )
                    continue
                response.raise_for_status()
                body = response.json()
                self._record_spend(body)
                text = body["choices"][0]["message"]["content"]
                return self._parse(text)
            except Exception as exc:
                if attempt >= self.max_retries:
                    log.warning("analysis failed for %s: %s", article.content_hash[:12], exc)
                    return None
                time.sleep(1.5 * (attempt + 1))
        return None

    @staticmethod
    def _parse(text: str) -> NewsAnalysis | None:
        if not text:
            return None
        candidate = text.strip()
        if candidate.startswith("```"):
            candidate = candidate.strip("`")
            candidate = candidate.split("\n", 1)[-1] if "\n" in candidate else candidate
        try:
            return NewsAnalysis.model_validate(json.loads(candidate))
        except Exception:
            pass
        # Last resort: pull the outermost JSON object out of a chatty response.
        match = _JSON_BLOCK.search(text)
        if match:
            try:
                return NewsAnalysis.model_validate(json.loads(match.group(0)))
            except Exception:
                return None
        return None

    def _record_spend(self, body: dict) -> None:
        usage = body.get("usage") or {}
        self._spent += (
            usage.get("prompt_tokens", 0) * self.price_in / 1e6
            + usage.get("completion_tokens", 0) * self.price_out / 1e6
        )

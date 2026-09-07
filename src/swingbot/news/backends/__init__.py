"""Interchangeable news-analysis backends.

All four satisfy the same ``NewsAnalyzer`` protocol and return the same
:class:`~swingbot.news.models.NewsAnalysis`, so the choice is a config line rather than a
code change. That is the whole point: the system is not bound to any one AI provider.
"""

from .lexicon import LexiconAnalyzer

__all__ = ["LexiconAnalyzer", "build_backend"]


def build_backend(cfg):
    """Construct the analyzer named in ``cfg.nlp.backend``.

    Falls back to the lexicon when the requested backend cannot run — a missing package
    or an absent API key — with a warning rather than an exception. A weekly signal run
    should degrade to a free sentiment source, not abort, and the report records which
    backend actually produced the features.
    """
    import logging

    log = logging.getLogger(__name__)
    name = cfg.nlp.backend.strip().lower()

    if name == "lexicon":
        return LexiconAnalyzer()

    if name == "anthropic":
        from .anthropic_backend import AnthropicAnalyzer

        backend = AnthropicAnalyzer(
            model=cfg.nlp.anthropic.model,
            api_key_env=cfg.nlp.anthropic.api_key_env,
            max_retries=cfg.nlp.max_retries,
            timeout=cfg.nlp.timeout_seconds,
            max_spend_usd=cfg.nlp.max_spend_usd,
        )
    elif name == "openai_compat":
        from .openai_compat import OpenAICompatAnalyzer

        backend = OpenAICompatAnalyzer(
            base_url=cfg.nlp.openai_compat.base_url,
            model=cfg.nlp.openai_compat.model,
            api_key_env=cfg.nlp.openai_compat.api_key_env,
            max_retries=cfg.nlp.max_retries,
            timeout=cfg.nlp.timeout_seconds,
            max_spend_usd=cfg.nlp.max_spend_usd,
        )
    elif name == "finbert":
        from .finbert import FinbertAnalyzer

        backend = FinbertAnalyzer(model=cfg.nlp.finbert.model)
    else:
        raise ValueError(
            f"Unknown nlp backend {cfg.nlp.backend!r}. "
            "Known: lexicon, openai_compat, anthropic, finbert"
        )

    if not backend.available():
        log.warning(
            "backend %r is not usable here (missing package or credentials); "
            "falling back to the offline lexicon",
            name,
        )
        return LexiconAnalyzer()
    return backend

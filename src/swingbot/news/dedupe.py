"""Deduplication and novelty scoring.

Financial news is overwhelmingly redundant. One Reuters story about a results
announcement appears on a dozen aggregators within an hour, each with a slightly
different headline. Counted naively, that looks like a burst of twelve independent news
events, and "abnormal news volume" — one of the more predictive news features there is —
becomes a measure of how many sites syndicate a wire service.

Two defences:

**Near-duplicate collapse.** Articles are hashed on character n-grams and grouped by
similarity, keeping the earliest by publication time. Character n-grams rather than words
because headline rewrites reorder and substitute words while preserving most character
sequences.

**Novelty scoring.** Every surviving article is scored against the same ticker's coverage
over the preceding days. A story that repeats last week's news scores near zero and is
downweighted in aggregation, without being discarded — repetition is itself weakly
informative.

Novelty is computed strictly against **prior** articles. Scoring against the full corpus
would let tomorrow's coverage decide how novel today's article was, which is a subtle
leak that the truncation test would not catch, because it lives in the news path rather
than the price path.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Sequence
from datetime import datetime, timedelta

import numpy as np

from ..types import RawArticle

log = logging.getLogger(__name__)

_WORD = re.compile(r"[a-z0-9]+")


def normalise_text(text: str) -> str:
    """Lowercase, collapse whitespace, strip common wire boilerplate."""
    lowered = " ".join(text.lower().split())
    for marker in (
        "(reuters)", "(bloomberg)", "(pti)", "(ani)", "(ians)", "- reuters", "click here",
        "read more", "also read", "advertisement", "subscribe to",
    ):
        lowered = lowered.replace(marker, " ")
    return " ".join(lowered.split())


def char_ngrams(text: str, n: int = 5) -> set[str]:
    """Character n-grams, the similarity unit for near-duplicate detection."""
    cleaned = normalise_text(text)
    if len(cleaned) < n:
        return {cleaned} if cleaned else set()
    return {cleaned[i : i + n] for i in range(len(cleaned) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    if intersection == 0:
        return 0.0
    return intersection / len(a | b)


def article_hash(article: RawArticle) -> str:
    return hashlib.sha256(normalise_text(article.text).encode("utf-8")).hexdigest()


def deduplicate(
    articles: Sequence[RawArticle],
    *,
    threshold: float = 0.60,
    window_hours: float = 72.0,
) -> list[RawArticle]:
    """Collapse near-duplicates, keeping the earliest of each cluster.

    Keeping the earliest matters: the first publication is when the information actually
    became available, and keeping a later syndication would delay the signal by hours for
    no reason.
    """
    if not articles:
        return []

    ordered = sorted(articles, key=lambda a: (a.ticker, a.available_at))
    kept: list[RawArticle] = []
    signatures: list[tuple[str, set[str], object]] = []

    for article in ordered:
        signature = char_ngrams(article.title + " " + article.body[:600])
        duplicate = False
        for ticker, other_sig, seen_at in reversed(signatures):
            if ticker != article.ticker:
                continue
            if article.available_at - seen_at > timedelta(hours=window_hours):
                break
            if jaccard(signature, other_sig) >= threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(article)
            signatures.append((article.ticker, signature, article.available_at))

    removed = len(articles) - len(kept)
    if removed:
        log.info("deduplicated %d of %d article(s)", removed, len(articles))
    return kept


def novelty_key(article: RawArticle) -> tuple[str, datetime]:
    """Identity of one *occurrence* of an article.

    Deliberately not the content hash alone. Two publications of identical text share a
    content hash — that is what makes the hash useful for the analysis cache, since the
    extracted facts are the same either way. But novelty is a property of the occurrence,
    not of the text: the first appearance of a story is novel and a republication a week
    later is not.

    Keying novelty on the content hash alone silently conflated them, and because the
    later occurrence is processed second, its low score overwrote the original's high
    one. The original's novelty then depended on whether the story happened to be
    republished later, which is look-ahead living entirely inside the news path where the
    price-side truncation test cannot see it.
    """
    return (article.content_hash, article.available_at)


def novelty_scores(
    articles: Sequence[RawArticle], *, lookback_days: float = 7.0
) -> dict[tuple[str, datetime], float]:
    """Novelty of each article occurrence against the same ticker's *prior* coverage.

    Returns ``(content_hash, available_at) -> score`` in ``[0, 1]``, where 1 means
    nothing like it has appeared recently. Strictly backward-looking, so a story is never
    judged against coverage that had not yet been published.
    """
    if not articles:
        return {}

    ordered = sorted(articles, key=lambda a: (a.ticker, a.available_at))
    history: dict[str, list[tuple[datetime, set[str]]]] = {}
    out: dict[tuple[str, datetime], float] = {}
    window = timedelta(days=lookback_days)

    for article in ordered:
        signature = char_ngrams(article.title + " " + article.body[:600])
        prior = history.setdefault(article.ticker, [])
        # Only articles strictly before this one, inside the lookback.
        relevant = [
            sig
            for seen_at, sig in prior
            if 0 <= (article.available_at - seen_at).total_seconds() <= window.total_seconds()
        ]
        if not relevant:
            out[novelty_key(article)] = 1.0
        else:
            closest = max(jaccard(signature, sig) for sig in relevant)
            out[novelty_key(article)] = float(max(0.0, 1.0 - closest))
        prior.append((article.available_at, signature))

    return out


def token_overlap(a: str, b: str) -> float:
    """Word-level Jaccard. Cheaper than n-grams, used for quick checks."""
    ta = set(_WORD.findall(a.lower()))
    tb = set(_WORD.findall(b.lower()))
    return jaccard(ta, tb)


def summarise_coverage(articles: Sequence[RawArticle]) -> dict[str, float]:
    """Coverage statistics, reported by ``swingbot doctor``."""
    if not articles:
        return {"n_articles": 0, "n_tickers": 0, "articles_per_ticker": 0.0}
    tickers = {a.ticker for a in articles}
    lengths = np.array([len(a.body) for a in articles], dtype=float)
    return {
        "n_articles": len(articles),
        "n_tickers": len(tickers),
        "articles_per_ticker": round(len(articles) / max(len(tickers), 1), 2),
        "median_body_chars": float(np.median(lengths)),
        "headline_only_frac": float((lengths < 50).mean()),
    }

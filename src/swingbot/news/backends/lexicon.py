"""Offline lexicon backend. No API key, no network, no cost.

The default, deliberately. A system that needs an API key before it does anything useful
is a system most people never get running, so this backend makes the whole pipeline work
the moment the repository is cloned.

It uses a financial-domain word list rather than a general-purpose sentiment lexicon,
which matters more than it sounds. In ordinary English "liability", "cut" and "charge"
are neutral or mildly negative; in financial text they are strongly negative and highly
informative. General lexicons trained on product reviews get this backwards often enough
to be worse than nothing. The word lists here follow the Loughran-McDonald approach of
scoring finance-specific usage.

Event classification is keyword-driven and therefore blunt. It will misclassify unusual
phrasings that a language model would read correctly. That is the honest trade: this
backend is free, deterministic and reproducible forever, and it establishes the floor
that a paid backend has to beat. The ablation will say whether the paid one earns its
cost.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from ...types import EntityRole, EventType, RawArticle
from ..models import EntityMention, NewsAnalysis

# --------------------------------------------------------------------------------------
# Finance-specific sentiment words
# --------------------------------------------------------------------------------------

POSITIVE = frozenset(["beat", "beats", "beating", "exceeded", "exceeds", "outperform", "outperformed", "upgrade", "upgraded", "raised", "raises", "surge", "surged", "soar", "soared", "jump", "jumped", "rally", "rallied", "gain", "gains", "gained", "rose", "rise", "rising", "profit", "profitable", "record", "strong", "stronger", "strength", "robust", "solid", "healthy", "improve", "improved", "improvement", "growth", "grew", "expanding", "expansion", "accelerate", "accelerated", "momentum", "win", "wins", "won", "award", "awarded", "secured", "landed", "bagged", "contract", "order", "approval", "approved", "cleared", "clearance", "authorised", "authorized", "granted", "licence", "license", "buyback", "repurchase", "dividend", "bonus", "special-dividend", "partnership", "alliance", "collaboration", "tie-up", "expansion", "launch", "launched", "unveil", "unveiled", "breakthrough", "milestone", "successful", "success", "optimistic", "confident", "upbeat", "bullish", "turnaround", "recovery", "rebound", "revival", "outperformance", "premium", "up", "higher", "advance", "advances", "buy", "accumulate"])

NEGATIVE = frozenset(["miss", "missed", "missing", "below", "shortfall", "disappoint", "disappointing", "disappointed", "downgrade", "downgraded", "cut", "cuts", "cutting", "lowered", "lower", "reduce", "reduced", "slash", "slashed", "fall", "fell", "falling", "drop", "dropped", "decline", "declined", "plunge", "plunged", "slump", "slumped", "tumble", "tumbled", "sink", "sank", "crash", "crashed", "weak", "weaker", "weakness", "soft", "softer", "sluggish", "loss", "losses", "lossmaking", "deficit", "writedown", "write-off", "impairment", "provision", "debt", "leverage", "default", "insolvency", "bankruptcy", "bankrupt", "liquidation", "restructuring", "probe", "investigation", "inquiry", "raid", "fine", "penalty", "sanction", "violation", "breach", "fraud", "lawsuit", "litigation", "sue", "sued", "sues", "verdict", "injunction", "recall", "halt", "halted", "suspension", "suspended", "shutdown", "strike", "protest", "outage", "disruption", "resign", "resigned", "resignation", "stepping-down", "ouster", "fired", "sacked", "exit", "warning", "warns", "warned", "caution", "cautious", "concern", "concerns", "risk", "risks", "headwind", "headwinds", "dilution", "placement", "stake-sale", "offloaded", "pledged", "bearish", "pessimistic", "downturn", "slowdown", "contraction", "down", "lower", "sell", "reduce", "underperform", "crack", "cracks", "worry", "worries", "pressure"])

INTENSIFIERS = frozenset(
    ["sharply", "steeply", "significantly", "substantially", "materially", "dramatically", "massive", "huge", "record", "surprisingly", "unexpectedly"]
)

NEGATORS = frozenset(["not", "no", "never", "without", "fails", "failed", "unable", "denies", "denied", "refuted"])

SPECULATIVE = (
    "reportedly", "rumour", "rumor", "in talks", "may ", "might ", "could ", "considering",
    "exploring", "weighing", "is said to", "sources said", "people familiar",
    "speculation", "potential", "eyeing", "mulling", "plans to", "expected to",
)

RECAP = (
    "market wrap", "closing bell", "session highlights", "week in review", "recap",
    "at a glance", "top gainers", "top losers", "sensex", "nifty closed", "stocks to watch",
    "market close", "midday", "roundup", "here's what", "what to know",
)

MACRO = (
    "inflation", "interest rate", "rbi ", "federal reserve", "fed ", "gdp", "budget",
    "monetary policy", "repo rate", "bond yield", "crude oil price", "rupee", "dollar index",
    "trade war", "tariff", "recession", "unemployment", "cpi", "ppi",
)

#: Keyword patterns per event type, checked in order. First match wins, so more specific
#: patterns must come first: an "acquisition approval" is an approval, not a takeover.
EVENT_PATTERNS: tuple[tuple[EventType, tuple[str, ...]], ...] = (
    (EventType.MA_TARGET, ("to be acquired", "takeover bid", "acquisition of", "buyout offer", "open offer", "to acquire stake in")),
    (EventType.MA_ACQUIRER, ("acquires", "to acquire", "acquisition", "merger", "buys stake", "amalgamation")),
    (EventType.EARNINGS_BEAT, ("beats estimate", "beat estimates", "above estimate", "tops estimate", "exceeds expectation", "profit beats")),
    (EventType.EARNINGS_MISS, ("misses estimate", "missed estimates", "below estimate", "profit falls", "profit declines", "misses expectation", "net loss")),
    (EventType.EARNINGS_INLINE, ("in line with estimate", "meets estimate", "matches estimate")),
    (EventType.GUIDANCE_UP, ("raises guidance", "raises outlook", "hikes forecast", "upgrades guidance", "raises target")),
    (EventType.GUIDANCE_DOWN, ("cuts guidance", "lowers outlook", "trims forecast", "warns on", "profit warning", "cuts forecast")),
    (EventType.DIVIDEND_CHANGE, ("dividend", "interim payout", "final payout")),
    (EventType.BUYBACK, ("buyback", "share repurchase", "repurchase programme", "repurchase program")),
    (EventType.DILUTION, ("qip", "share sale", "equity raise", "placement", "rights issue", "convertible", "stake sale", "offer for sale")),
    (EventType.REGULATORY_APPROVAL, ("approval", "approved", "cleared by", "gets nod", "receives nod", "clearance", "licence granted")),
    (EventType.REGULATORY_ACTION, ("sebi", "penalty", "fine of", "probe", "investigation", "show cause", "sanction", "banned", "raid", "notice from")),
    (EventType.LITIGATION, ("lawsuit", "sues", "sued", "litigation", "court", "tribunal", "verdict", "arbitration")),
    (EventType.CREDIT_RATING, ("rating", "moody", "fitch", "s&p", "crisil", "icra", "outlook revised")),
    (EventType.ANALYST_ACTION, ("upgrade", "downgrade", "initiates coverage", "price target", "target price", "brokerage", "rated buy", "rated sell")),
    (EventType.CONTRACT_WIN, ("wins order", "bags order", "wins contract", "secures contract", "awarded", "order worth", "lou", "letter of intent")),
    (EventType.PRODUCT_LAUNCH, ("launches", "unveils", "introduces", "rolls out", "new product", "commissions")),
    (EventType.EXEC_CHANGE, ("ceo", "chief executive", "managing director", "cfo", "chairman", "resigns", "appointed", "steps down")),
    (EventType.OPERATIONAL_DISRUPTION, ("fire at", "strike", "shutdown", "outage", "recall", "accident", "explosion", "halt production", "lockout")),
)


#: Event types whose direction is structural rather than a matter of wording.
#:
#: This exists because word-level sentiment gets a whole class of headline exactly
#: backwards. "SEBI launches probe into disclosure lapses" scores neutral on a word list,
#: because "launches" is a positive word in the product sense and cancels "probe" — yet a
#: regulatory investigation is unambiguously bad news whatever verb introduces it. The
#: same trap catches "wins appeal against penalty" and "halts production after fire".
#:
#: Where the event type is known and carries an inherent direction, that direction wins.
#: This is not a tweak to make particular headlines score well: it encodes that the
#: meaning of a corporate event does not depend on which verb a sub-editor chose.
EVENT_SIGN: dict[EventType, float] = {
    EventType.EARNINGS_BEAT: 1.0,
    EventType.EARNINGS_MISS: -1.0,
    EventType.GUIDANCE_UP: 1.0,
    EventType.GUIDANCE_DOWN: -1.0,
    EventType.REGULATORY_APPROVAL: 1.0,
    EventType.REGULATORY_ACTION: -1.0,
    EventType.LITIGATION: -1.0,
    EventType.OPERATIONAL_DISRUPTION: -1.0,
    EventType.CONTRACT_WIN: 1.0,
    EventType.BUYBACK: 1.0,
    EventType.DILUTION: -1.0,
    # A takeover target normally re-rates toward the offer; the acquirer is ambiguous and
    # is deliberately left out.
    EventType.MA_TARGET: 1.0,
}

#: Floor applied when the event's own direction overrides ambiguous wording. Enough to
#: register a clear direction without pretending to a confidence the text did not supply.
EVENT_SIGN_FLOOR = 0.55


#: Suffixes stripped when a word is not found verbatim. Financial headlines are written
#: in the present tense — "Drops 2%", "Wins Order", "Slumps" — while a hand-built word
#: list naturally holds base forms. Without this, "drop" scores and "drops" does not,
#: which is exactly the case that appears in most headlines.
_SUFFIXES = ("s", "es", "ed", "ing", "d")


def _polarity(word: str) -> float:
    """Sentiment polarity of a word, trying light suffix stripping before giving up."""
    if word in POSITIVE:
        return 1.0
    if word in NEGATIVE:
        return -1.0
    for suffix in _SUFFIXES:
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            stem = word[: -len(suffix)]
            if stem in POSITIVE:
                return 1.0
            if stem in NEGATIVE:
                return -1.0
            # "slumped" -> "slump", "climbing" -> "climb": restore a doubled consonant
            # or a dropped terminal e.
            if suffix in ("ed", "ing"):
                if stem + "e" in POSITIVE:
                    return 1.0
                if stem + "e" in NEGATIVE:
                    return -1.0
    return 0.0


class LexiconAnalyzer:
    """Keyword and lexicon based extraction. Deterministic and free."""

    backend = "lexicon"
    model_id = "loughran-mcdonald-style-v1"

    def __init__(self, *, window: int = 4) -> None:
        #: How many words before a sentiment word are checked for a negator.
        self.window = window

    def available(self) -> bool:
        return True

    def estimated_cost_usd(self, n_articles: int) -> float:  # noqa: ARG002
        return 0.0

    def analyze(self, articles: Sequence[RawArticle]) -> list[NewsAnalysis | None]:
        return [self._analyze_one(a) for a in articles]

    # ----------------------------------------------------------------------- private

    def _analyze_one(self, article: RawArticle) -> NewsAnalysis:
        text = article.text
        lowered = text.lower()
        # The headline carries more signal per word than the body, so it is weighted up
        # by being scored twice.
        scored_text = f"{article.title.lower()} {article.title.lower()} {lowered}"

        sentiment, hits = self._sentiment(scored_text)
        event = self._classify(lowered)
        sentiment = self._apply_event_sign(sentiment, event)
        is_recap = any(marker in lowered for marker in RECAP)
        is_speculative = any(marker in lowered for marker in SPECULATIVE)
        is_macro = event is EventType.MACRO or (
            sum(marker in lowered for marker in MACRO) >= 2
        )

        magnitude = self._magnitude(event, hits, len(text))
        if is_recap:
            magnitude = min(magnitude, 1)

        if sentiment > 0.08:
            direction = "up"
        elif sentiment < -0.08:
            direction = "down"
        else:
            direction = "neutral"

        # Confidence rises with how one-sided the wording is and how many hits there
        # were, so a single stray word does not produce a confident call.
        confidence = min(1.0, abs(sentiment) * 1.4 + min(hits, 6) * 0.05)
        if event in (EventType.OPINION_COMMENTARY, EventType.OTHER, EventType.RECAP):
            confidence *= 0.5

        return NewsAnalysis(
            event_type=EventType.RECAP if is_recap and event is EventType.OTHER else event,
            sentiment=sentiment,
            magnitude=magnitude,
            is_speculative=is_speculative,
            is_recap=is_recap,
            is_company_specific=not is_macro,
            expected_direction=direction,
            direction_confidence=round(confidence, 3),
            entities=[EntityMention(ticker=article.ticker, role=EntityRole.PRIMARY)],
            numeric_surprise_pct=self._surprise(lowered),
            rationale=f"lexicon: {hits} sentiment term(s), event {event.value}",
        )

    @staticmethod
    def _apply_event_sign(sentiment: float, event: EventType) -> float:
        """Let a structural event direction override ambiguous wording.

        Applied only when the word-level reading actually disagrees with the event, or is
        too weak to express a direction. Where the wording already agrees, the wording is
        kept, because it carries magnitude information the event type does not.
        """
        expected = EVENT_SIGN.get(event)
        if expected is None:
            return sentiment
        if sentiment * expected >= 0.15:
            return sentiment
        return expected * max(EVENT_SIGN_FLOOR, abs(sentiment))

    def _sentiment(self, text: str) -> tuple[float, int]:
        """Net sentiment in [-1, 1], with negation and intensifier handling."""
        words = re.findall(r"[a-z][a-z'\-]+", text)
        if not words:
            return 0.0, 0

        score = 0.0
        hits = 0
        for i, word in enumerate(words):
            polarity = _polarity(word)
            if polarity == 0.0:
                continue
            hits += 1
            weight = 1.0
            preceding = words[max(0, i - self.window) : i]
            # "not strong" is negative, "no shortfall" is positive.
            if any(w in NEGATORS for w in preceding):
                weight *= -1.0
            if any(w in INTENSIFIERS for w in preceding):
                weight *= 1.5
            score += polarity * weight

        if hits == 0:
            return 0.0, 0
        # Normalise by the square root of hit count: a long article with many terms
        # should score more confidently than a headline, but not proportionally more.
        normalised = score / (hits**0.5 * 2.0)
        return float(max(-1.0, min(1.0, normalised))), hits

    def _classify(self, text: str) -> EventType:
        for event, patterns in EVENT_PATTERNS:
            if any(pattern in text for pattern in patterns):
                return event
        if sum(marker in text for marker in MACRO) >= 2:
            return EventType.MACRO
        if any(marker in text for marker in ("opinion", "view:", "analysis:", "column")):
            return EventType.OPINION_COMMENTARY
        return EventType.OTHER

    def _magnitude(self, event: EventType, hits: int, length: int) -> int:
        major = {
            EventType.MA_TARGET,
            EventType.MA_ACQUIRER,
            EventType.GUIDANCE_DOWN,
            EventType.GUIDANCE_UP,
            EventType.REGULATORY_ACTION,
            EventType.OPERATIONAL_DISRUPTION,
        }
        notable = {
            EventType.EARNINGS_BEAT,
            EventType.EARNINGS_MISS,
            EventType.CONTRACT_WIN,
            EventType.DILUTION,
            EventType.REGULATORY_APPROVAL,
            EventType.LITIGATION,
            EventType.EXEC_CHANGE,
            EventType.BUYBACK,
        }
        if event in major:
            return 3
        if event in notable:
            return 2
        if event in (EventType.OTHER, EventType.RECAP, EventType.OPINION_COMMENTARY):
            return 1 if hits >= 3 or length > 800 else 0
        return 1

    def _surprise(self, text: str) -> float | None:
        """Pull a stated percentage when the article frames it as a versus-expected move."""
        match = re.search(
            r"(beat|miss|above|below|versus|vs\.?|against)\D{0,30}?(\d{1,3}(?:\.\d+)?)\s*(?:per cent|percent|%)",
            text,
        )
        if not match:
            return None
        value = float(match.group(2))
        return -value if match.group(1) in ("miss", "below") else value

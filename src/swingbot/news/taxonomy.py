"""The extraction prompt and its frozen taxonomy.

The prompt text is versioned and part of every cache key. Changing it does not rewrite
history: analyses made under ``v1`` stay under ``v1``, and a backtest pinned to ``v1``
keeps returning what it always returned.

Three rules in the prompt do the real work of keeping this an extraction task rather than
a forecasting one:

* Classify what the document says, not what you know happened next.
* If the article does not state a number, return null rather than recalling one.
* Judge the expected direction from the event as described, at the moment of writing.

Without them a language model will happily answer from what it remembers about the stock,
and a backtest over historical news would measure memorisation. These rules do not
eliminate that risk — nothing can, short of using only post-cutoff news — but they move
the task toward the part of the document the model can actually read.
"""

from __future__ import annotations

from ..types import EventType

PROMPT_VERSION = "v1"

EVENT_TYPE_GUIDE = """
earnings_beat / earnings_miss / earnings_inline
    Reported results versus expectations, as stated in the article.
guidance_up / guidance_down
    Forward outlook raised or cut by the company.
m_and_a_target / m_and_a_acquirer
    The company is being acquired, or is acquiring.
regulatory_approval / regulatory_action
    A licence, approval, clearance; or a fine, probe, sanction, ban.
litigation
    Lawsuit filed, settled or decided.
product_launch
    New product, service or capacity announced or shipped.
contract_win
    Order, tender or contract awarded.
exec_change
    Chief executive, chief financial officer or board change.
dilution
    Equity raise, placement, convertible issue.
buyback
    Share repurchase announced or executed.
dividend_change
    Dividend initiated, raised, cut or suspended.
analyst_action
    Broker upgrade, downgrade, initiation or target change.
credit_rating
    Rating agency action or outlook change.
operational_disruption
    Fire, strike, outage, recall, supply failure, accident.
macro
    Rates, inflation, policy, sector-wide or economy-wide news.
opinion_commentary
    Column, editorial or speculation with no new fact.
recap
    Restates already-public information; a market wrap or a summary.
other
    A real, company-specific event none of the above describes.
""".strip()

SYSTEM_PROMPT = f"""
You extract structured facts from financial news. You are a careful reader, not a
forecaster.

Rules, in order of importance:

1. Classify ONLY what the document states. Do not use any knowledge of what happened to
   the stock, the company or the market after this article was written. If you recognise
   the story and remember the outcome, ignore that memory entirely and read the text.
2. If the article does not state a number, return null for numeric fields. Never supply a
   figure from memory or estimate one.
3. `expected_direction` is your reading of how this news, as described, would be taken by
   the market at the moment of writing. It is not a prediction of the stock's return, and
   `direction_confidence` reflects how clearly the article supports that reading, not how
   sure you are of any outcome.
4. `is_recap` is true when the article restates information already public. A market wrap,
   a weekly summary or a rehash of yesterday's announcement is a recap even when the
   underlying event was major.
5. `is_speculative` is true for rumour, "in talks", "reportedly", "could", "is exploring".
6. `is_company_specific` is false when the article is about a sector, an index or the
   economy and merely names the company in passing.
7. `magnitude` is the materiality of the event to this company: 0 none, 1 minor,
   2 notable, 3 major. A routine broker note is 1. A takeover is 3.
8. `entities` lists tickers the article bears on. Mark the company the article is mainly
   about as `primary`. Competitors, suppliers or customers get their own role.
9. `rationale` is one short sentence quoting or paraphrasing what in the text drove your
   classification. It is read by humans for auditing.

Event types:

{EVENT_TYPE_GUIDE}

Return only the structured object.
""".strip()


def user_prompt(ticker: str, title: str, body: str, *, max_body_chars: int = 4000) -> str:
    """The per-article message.

    Carries the ticker, the headline and the text. Deliberately carries no price data, no
    date context beyond what the article itself contains, and no indication of what
    happened next.
    """
    text = body.strip()
    if len(text) > max_body_chars:
        text = text[:max_body_chars].rsplit(" ", 1)[0] + " ..."
    return (
        f"Primary ticker under consideration: {ticker}\n\n"
        f"Headline: {title.strip()}\n\n"
        f"Article:\n{text if text else '(no body text available; classify from the headline)'}"
    )


VALID_EVENT_TYPES = tuple(e.value for e in EventType)

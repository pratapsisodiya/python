"""Schema contract for the whole system.

Column names live here as constants rather than as string literals scattered through
the codebase, so a rename is a one-line change and a typo is an ImportError instead of
a silent all-NaN column.

Timestamp policy, enforced by :mod:`swingbot.pit`:

* Every timestamp in the system is timezone-aware UTC. Naive timestamps are rejected.
* ``SESSION`` is a calendar date (the trading day), not a timestamp.
* ``AVAILABLE_AT`` is the moment a row could first have been known. It is the only
  field that decides whether a row is visible at a decision time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum
from typing import Any

# --------------------------------------------------------------------------------------
# Column names
# --------------------------------------------------------------------------------------

TICKER = "ticker"
SESSION = "session"
OPEN = "open"
HIGH = "high"
LOW = "low"
CLOSE = "close"
VOLUME = "volume"
SPLIT_FACTOR = "split_factor"
DIV_CASH = "div_cash"
AVAILABLE_AT = "available_at"
IS_DELISTED = "is_delisted"

#: Decision timestamp — the moment the signal is computed (a Friday close).
DECISION_TS = "decision_ts"
#: The session whose close produced the decision.
DECISION_SESSION = "decision_session"
#: The session at whose OPEN the trade is filled. Always after ``DECISION_SESSION``.
ENTRY_SESSION = "entry_session"
#: The session at whose OPEN the position is closed or rolled.
EXIT_SESSION = "exit_session"

SECTOR = "sector"
LABEL = "label"
#: Start and end of the interval a label spans. Used by purging.
LABEL_T0 = "label_t0"
LABEL_T1 = "label_t1"
SAMPLE_WEIGHT = "sample_weight"

BAR_COLUMNS = (
    TICKER,
    SESSION,
    OPEN,
    HIGH,
    LOW,
    CLOSE,
    VOLUME,
    SPLIT_FACTOR,
    DIV_CASH,
    AVAILABLE_AT,
    IS_DELISTED,
)

#: Prefix convention that lets the blender split the feature matrix into blocks
#: without maintaining a second list. Anything not prefixed ``news_`` is a price
#: feature; see :mod:`swingbot.model.blend`.
NEWS_FEATURE_PREFIX = "news_"


# --------------------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------------------


class Side(StrEnum):
    BUY = "buy"
    SELL = "sell"


class ShortInstrument(StrEnum):
    """How a short position is actually expressed in a given market.

    ``CASH_EQUITY`` is a borrowed-stock short (US). ``FUTURES`` is a single-stock
    futures short, which is the only way to carry a weekly short on NSE. ``NONE``
    makes the strategy long-only.
    """

    CASH_EQUITY = "cash_equity"
    FUTURES = "futures"
    NONE = "none"


class OrderType(StrEnum):
    MARKET = "market"
    LIMIT = "limit"


class EventType(StrEnum):
    """Frozen taxonomy of news events.

    Frozen because it is part of the cache key for every news analysis ever computed.
    Adding a member is safe; renaming or removing one invalidates history and must bump
    ``nlp.prompt_version``.
    """

    EARNINGS_BEAT = "earnings_beat"
    EARNINGS_MISS = "earnings_miss"
    EARNINGS_INLINE = "earnings_inline"
    GUIDANCE_UP = "guidance_up"
    GUIDANCE_DOWN = "guidance_down"
    MA_TARGET = "m_and_a_target"
    MA_ACQUIRER = "m_and_a_acquirer"
    REGULATORY_APPROVAL = "regulatory_approval"
    REGULATORY_ACTION = "regulatory_action"
    LITIGATION = "litigation"
    PRODUCT_LAUNCH = "product_launch"
    CONTRACT_WIN = "contract_win"
    EXEC_CHANGE = "exec_change"
    DILUTION = "dilution"
    BUYBACK = "buyback"
    DIVIDEND_CHANGE = "dividend_change"
    ANALYST_ACTION = "analyst_action"
    CREDIT_RATING = "credit_rating"
    OPERATIONAL_DISRUPTION = "operational_disruption"
    MACRO = "macro"
    OPINION_COMMENTARY = "opinion_commentary"
    RECAP = "recap"
    OTHER = "other"


class EntityRole(StrEnum):
    PRIMARY = "primary"
    PEER = "peer"
    SUPPLIER = "supplier"
    CUSTOMER = "customer"
    ACQUIRER = "acquirer"
    TARGET = "target"
    INDEX = "index"


# --------------------------------------------------------------------------------------
# Plain data carriers
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Order:
    """A broker-agnostic instruction.

    Deliberately uses no broker vocabulary. Adapters translate this into whatever
    their API wants; nothing in the core knows what that looks like.
    """

    ticker: str
    side: Side
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    #: Marks a short leg that a market cannot express as cash equity.
    instrument: str = "equity"
    client_order_id: str = ""
    tag: str = ""

    @property
    def signed_quantity(self) -> float:
        return self.quantity if self.side is Side.BUY else -self.quantity


@dataclass(frozen=True, slots=True)
class Fill:
    ticker: str
    session: date
    quantity: float
    price: float
    cost: float
    instrument: str = "equity"
    tag: str = ""


@dataclass(frozen=True, slots=True)
class TargetPosition:
    ticker: str
    weight: float
    score: float = 0.0
    instrument: str = "equity"
    sector: str = ""

    @property
    def is_short(self) -> bool:
        return self.weight < 0


@dataclass(slots=True)
class TargetPortfolio:
    decision_session: date
    entry_session: date
    positions: list[TargetPosition] = field(default_factory=list)
    #: Multiplier the risk overlay applied, 1.0 when untouched, 0.0 when flat.
    risk_scale: float = 1.0
    notes: list[str] = field(default_factory=list)

    @property
    def weights(self) -> dict[str, float]:
        return {p.ticker: p.weight for p in self.positions}

    @property
    def gross(self) -> float:
        return sum(abs(p.weight) for p in self.positions)

    @property
    def net(self) -> float:
        return sum(p.weight for p in self.positions)

    @property
    def n_long(self) -> int:
        return sum(1 for p in self.positions if p.weight > 0)

    @property
    def n_short(self) -> int:
        return sum(1 for p in self.positions if p.weight < 0)


@dataclass(slots=True)
class ExecutionReport:
    adapter: str
    submitted: int
    accepted: int
    artifacts: dict[str, str] = field(default_factory=dict)
    messages: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RunMeta:
    run_id: str
    market: str
    decision_session: date
    entry_session: date
    equity: float
    currency: str
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RawArticle:
    """A news item as ingested, before any analysis.

    ``published_at`` is what the source claims. ``first_seen_at`` is when this pipeline
    actually saw it. Availability uses ``max`` of the two, because a source that
    backdates its timestamps would otherwise hand the backtest free information.
    """

    content_hash: str
    ticker: str
    title: str
    body: str
    url: str
    source: str
    published_at: datetime
    first_seen_at: datetime

    @property
    def available_at(self) -> datetime:
        return max(self.published_at, self.first_seen_at)

    @property
    def text(self) -> str:
        return f"{self.title}\n\n{self.body}".strip()

"""Transaction cost analysis: what the fills actually cost, against what was assumed.

Every backtest in this system charges a modelled cost — 24 bps round trip on US cash
equity, 58 on India — and the report sweeps that assumption at half, double and triple to
show the strategy does not depend on it being exactly right. That is the honest version of
*assuming*. It is not the same as *knowing*.

This module closes the loop. When a fill price is recorded, the difference between it and
the price the order was computed against is measurable, in basis points, per name and per
week. Compared against what the cost model predicted for that same trade, it answers the
question the sweep can only bound: is the modelled cost roughly right for this account, at
this size, at this broker?

Three things this deliberately does not do.

It does not compare against the full round-trip figure. A fill price contains spread and
market impact and nothing else — commission, STT and stamp duty are separate charges that
never appear in the price printed on the ticket. Comparing realised slippage against the
headline round trip would flatter the model by attributing charges to it that it never
claimed to be in the price. So the comparison is against the model's *spread plus impact*
component alone, computed at signal time and stored on the order.

It does not treat the reference price as fair value. The reference is the decision-session
close, which is the price the weights were computed from. The gap between that and the next
session's open is mostly overnight drift, not execution quality — a fact that shows up here
as slippage even when the fill was excellent. Read the *distribution across many weeks*,
not any single number, and expect the mean to sit near zero on drift alone if the broker is
doing a reasonable job.

It does not accept a fill it cannot attribute. A price with no matching order is dropped
and counted, rather than silently averaged into the result.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

BPS = 10_000.0

#: A fill this far from the reference is far more likely to be a typo than a trade. Two
#: thousand basis points is twenty percent; at weekly frequency on a liquid name, a real
#: fill does not land there, and letting one through would move the book-level average
#: enough to make the whole report useless.
IMPLAUSIBLE_SLIPPAGE_BPS = 2_000.0


@dataclass(slots=True)
class FillAnalysis:
    """One order, and how its fill compared with the price it was computed from."""

    client_order_id: str
    ticker: str
    side: str
    instrument: str
    quantity: float
    reference_price: float
    fill_price: float
    notional: float
    #: Positive is always a cost, whichever side the trade was.
    slippage_bps: float
    #: What the cost model predicted for spread plus impact on this trade.
    expected_bps: float | None = None

    @property
    def surprise_bps(self) -> float | None:
        """Realised minus expected. Positive means it cost more than the model said."""
        if self.expected_bps is None:
            return None
        return self.slippage_bps - self.expected_bps

    def to_dict(self) -> dict[str, Any]:
        return {
            "client_order_id": self.client_order_id,
            "ticker": self.ticker,
            "side": self.side,
            "instrument": self.instrument,
            "quantity": self.quantity,
            "reference_price": self.reference_price,
            "fill_price": self.fill_price,
            "notional": self.notional,
            "slippage_bps": self.slippage_bps,
            "expected_bps": self.expected_bps,
            "surprise_bps": self.surprise_bps,
        }


@dataclass(slots=True)
class SlippageReport:
    """Book-level execution quality for one week."""

    run_id: str = ""
    fills: list[FillAnalysis] = field(default_factory=list)
    n_orders: int = 0
    n_unattributed: int = 0
    #: Notional-weighted, because a bad fill on the largest trade matters most.
    weighted_slippage_bps: float = 0.0
    median_slippage_bps: float = 0.0
    weighted_expected_bps: float | None = None
    total_notional: float = 0.0
    #: Currency cost of the slippage, which is the number that shows up in the account.
    slippage_cost: float = 0.0
    dropped: list[str] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        """Fraction of the week's orders that have a usable fill price."""
        return len(self.fills) / self.n_orders if self.n_orders else 0.0

    @property
    def surprise_bps(self) -> float | None:
        if self.weighted_expected_bps is None:
            return None
        return self.weighted_slippage_bps - self.weighted_expected_bps

    def verdict(self) -> str:
        """A sentence a person can act on, or an honest refusal to give one."""
        if not self.fills:
            return (
                "no fill prices recorded — enter what you actually paid to find out "
                "whether the backtest's cost assumption holds for your account"
            )
        if len(self.fills) < 8:
            return (
                f"only {len(self.fills)} fill(s) recorded; too few to say anything about "
                "execution quality. Keep entering prices for a few weeks."
            )

        surprise = self.surprise_bps
        if surprise is None:
            return (
                f"realised slippage {self.weighted_slippage_bps:+.1f} bps, but no modelled "
                "expectation was stored for these orders, so there is nothing to compare it to"
            )
        if surprise > 15.0:
            return (
                f"fills cost {surprise:+.1f} bps more than the model assumed "
                f"({self.weighted_slippage_bps:+.1f} realised against "
                f"{self.weighted_expected_bps:+.1f} expected). If this persists, the "
                "backtest is optimistic — re-run it with --set backtest.cost_scale raised "
                "and see whether the strategy survives."
            )
        if surprise < -15.0:
            return (
                f"fills came in {abs(surprise):.1f} bps better than modelled. Pleasant, but "
                "check it is not just favourable overnight drift before trusting it."
            )
        return (
            f"realised slippage {self.weighted_slippage_bps:+.1f} bps against "
            f"{self.weighted_expected_bps:+.1f} modelled — the cost assumption is holding up"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "n_orders": self.n_orders,
            "n_filled": len(self.fills),
            "n_unattributed": self.n_unattributed,
            "coverage": self.coverage,
            "weighted_slippage_bps": self.weighted_slippage_bps,
            "median_slippage_bps": self.median_slippage_bps,
            "weighted_expected_bps": self.weighted_expected_bps,
            "surprise_bps": self.surprise_bps,
            "total_notional": self.total_notional,
            "slippage_cost": self.slippage_cost,
            "dropped": list(self.dropped),
            "verdict": self.verdict(),
            "fills": [f.to_dict() for f in self.fills],
        }


def slippage_bps(*, side: str, reference_price: float, fill_price: float) -> float:
    """Signed so that positive always means the fill was worse than the reference.

    A buy above the reference and a sell below it are both costs, and both come back
    positive. Without that sign convention a book of mixed sides averages toward zero
    however badly it executed.
    """
    if reference_price <= 0:
        return 0.0
    direction = 1.0 if side.lower() == "buy" else -1.0
    return direction * (fill_price - reference_price) / reference_price * BPS


def analyse_fills(orders: list[dict], fills: dict[str, dict]) -> SlippageReport:
    """Compare recorded fills against the prices their orders were computed from.

    ``orders`` are the rows from ``orders.csv``; ``fills`` maps a client order id to at
    least a ``price``. A fill with no price, no matching order, or an implausible price is
    dropped and reported, never averaged in.
    """
    report = SlippageReport(n_orders=len(orders))
    by_id = {str(o.get("client_order_id")): o for o in orders if o.get("client_order_id")}

    weighted_slip = 0.0
    weighted_expected = 0.0
    expected_notional = 0.0

    for order_id, record in (fills or {}).items():
        price = record.get("price")
        if price in (None, "") or float(price) <= 0:
            continue

        order = by_id.get(str(order_id))
        if order is None:
            report.n_unattributed += 1
            report.dropped.append(f"{order_id}: no matching order in this run")
            continue

        reference = float(order.get("est_price") or 0.0)
        if reference <= 0:
            report.dropped.append(f"{order_id}: the order carries no reference price")
            continue

        fill_price = float(price)
        # A partial fill is analysed at the quantity actually done, because that is the
        # notional the slippage was paid on.
        quantity = float(record.get("quantity") or order.get("quantity") or 0.0)
        if quantity <= 0:
            report.dropped.append(f"{order_id}: fill quantity is zero")
            continue

        side = str(order.get("side", "buy"))
        bps = slippage_bps(side=side, reference_price=reference, fill_price=fill_price)
        if abs(bps) > IMPLAUSIBLE_SLIPPAGE_BPS:
            report.dropped.append(
                f"{order_id}: fill {fill_price:,.2f} is {bps:+,.0f} bps from the reference "
                f"{reference:,.2f} — treated as a typo, not a fill"
            )
            continue

        expected = order.get("expected_slip_bps")
        expected = float(expected) if expected not in (None, "") else None

        notional = fill_price * quantity
        analysis = FillAnalysis(
            client_order_id=str(order_id),
            ticker=str(order.get("ticker", "")),
            side=side,
            instrument=str(order.get("instrument", "equity")),
            quantity=quantity,
            reference_price=reference,
            fill_price=fill_price,
            notional=notional,
            slippage_bps=bps,
            expected_bps=expected,
        )
        report.fills.append(analysis)

        weighted_slip += bps * notional
        report.total_notional += notional
        report.slippage_cost += bps / BPS * notional
        if expected is not None:
            weighted_expected += expected * notional
            expected_notional += notional

    if report.total_notional > 0:
        report.weighted_slippage_bps = weighted_slip / report.total_notional
    if report.fills:
        report.median_slippage_bps = statistics.median(f.slippage_bps for f in report.fills)
    if expected_notional > 0:
        report.weighted_expected_bps = weighted_expected / expected_notional

    report.fills.sort(key=lambda f: -abs(f.slippage_bps))
    return report

"""Notification protocol and the shared weekly-signal message.

The message body is built once and rendered per channel, so Telegram and email always say
the same thing. It deliberately leads with what changed rather than the full book: after
the first few weeks, most positions carry over, and a message that repeats fifteen
unchanged lines every Friday stops being read.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..types import TargetPortfolio


@runtime_checkable
class Notifier(Protocol):
    name: str

    def available(self) -> bool: ...

    def send(self, subject: str, body: str, *, html: str | None = None) -> bool: ...


def format_signal_message(
    portfolio: TargetPortfolio,
    *,
    market: str,
    currency: str,
    previous_weights: dict[str, float] | None = None,
    equity: float | None = None,
    warnings: list[str] | None = None,
) -> tuple[str, str]:
    """Build the plain-text weekly message. Returns ``(subject, body)``."""
    previous = previous_weights or {}
    subject = (
        f"swingbot {market.upper()} — {portfolio.n_long} long / {portfolio.n_short} short "
        f"for {portfolio.entry_session}"
    )

    lines = [
        f"Decision close : {portfolio.decision_session}",
        f"Enter at open  : {portfolio.entry_session}",
        f"Gross / Net    : {portfolio.gross:.0%} / {portfolio.net:+.0%}",
    ]
    if equity:
        lines.append(f"Equity         : {equity:,.0f} {currency}")
    if portfolio.risk_scale < 1.0:
        lines.append(
            f"RISK SCALE     : {portfolio.risk_scale:.0%} (drawdown control active)"
        )

    current = portfolio.weights
    opened = [t for t in current if t not in previous]
    closed = [t for t in previous if t not in current]
    resized = [
        t for t in current
        if t in previous and abs(current[t] - previous[t]) > 0.005
    ]

    if opened or closed or resized:
        lines.append("")
        lines.append("CHANGES")
        for ticker in sorted(opened):
            side = "LONG" if current[ticker] > 0 else "SHORT"
            lines.append(f"  OPEN  {side:<5} {ticker:<12} {current[ticker]:+.1%}")
        for ticker in sorted(closed):
            lines.append(f"  CLOSE       {ticker:<12} (was {previous[ticker]:+.1%})")
        for ticker in sorted(resized):
            lines.append(
                f"  RESIZE      {ticker:<12} {previous[ticker]:+.1%} -> {current[ticker]:+.1%}"
            )
    else:
        lines.append("")
        lines.append("CHANGES: none, hold the existing book.")

    lines.append("")
    lines.append("TARGET BOOK")
    for position in sorted(portfolio.positions, key=lambda p: -p.weight):
        marker = "futures" if position.instrument == "futures" else ""
        lines.append(
            f"  {position.ticker:<12} {position.weight:+.2%}  "
            f"score {position.score:+.3f}  {position.sector:<22} {marker}"
        )

    if portfolio.notes:
        lines.append("")
        lines.append("NOTES")
        lines.extend(f"  - {n}" for n in portfolio.notes)

    if warnings:
        lines.append("")
        lines.append("WARNINGS")
        lines.extend(f"  ! {w}" for w in warnings)

    lines.append("")
    lines.append("Signals only. Not investment advice. Verify before trading.")
    return subject, "\n".join(lines)


class NullNotifier:
    """Does nothing. Keeps the calling code free of None checks."""

    name = "null"

    def available(self) -> bool:
        return False

    def send(self, subject: str, body: str, *, html: str | None = None) -> bool:  # noqa: ARG002
        return False

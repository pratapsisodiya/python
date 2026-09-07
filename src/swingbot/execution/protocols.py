"""The execution seam — the only place a broker could ever attach.

Everything above this module computes target positions. Nothing above it knows what a
broker is, what an order ticket looks like, or which exchange the trade will reach. That
is not a stylistic preference: it is what makes the research honest. A backtest that
imports a broker SDK has, somewhere, a code path where live account state can influence a
historical decision, and finding it after the fact is much harder than preventing it.

``tests/test_architecture.py`` enforces the boundary by scanning imports, and fails the
build if any module outside this package touches a name on the broker denylist.

Adding a broker means writing one class with three methods, in this package, and changing
one config line. Nothing else moves.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from ..types import ExecutionReport, Order, RunMeta


@runtime_checkable
class ExecutionAdapter(Protocol):
    """Translates broker-agnostic orders into whatever a venue wants."""

    name: str

    def current_positions(self) -> dict[str, float]:
        """Ticker to signed share count. Empty when the adapter holds no state."""
        ...

    def account_equity(self) -> float:
        """Account value in the market's currency, used to convert weights to shares."""
        ...

    def submit(self, orders: Sequence[Order], meta: RunMeta) -> ExecutionReport:
        """Act on the orders. For file and paper adapters this writes or books them."""
        ...

    def close(self) -> None: ...

"""File execution adapter: writes order tickets you can act on anywhere.

The default, and the reason this system is broker-independent in practice rather than in
principle. It writes three files:

* ``orders.csv`` — the trades to place, in a shape most brokers' basket-upload screens
  accept, and which is readable by a human placing them by hand.
* ``targets.json`` — the full target book with weights and scores, for record-keeping and
  for the next run to diff against.
* ``positions.json`` — the holdings implied once the orders fill, which is what makes the
  next week's diff correct without any broker connection.

That last file is what turns a stateless signal generator into something that tracks a
portfolio over time. Without it, every week's orders would be computed against an empty
book and the strategy would appear to trade its entire position every week.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path

from ..io.store import read_json, write_json
from ..types import ExecutionReport, Order, RunMeta
from .orders import orders_to_frame, positions_after, summarise_orders

log = logging.getLogger(__name__)


class CSVExecutionAdapter:
    """Writes order files and maintains an assumed-fill position ledger."""

    name = "csv"

    def __init__(
        self,
        output_dir: Path | str,
        *,
        state_path: Path | str | None = None,
        equity: float = 1_000_000.0,
        prices: dict[str, float] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = Path(state_path) if state_path else self.output_dir / "positions.json"
        self._equity = equity
        self._prices = prices or {}

    # ------------------------------------------------------------------------ state

    def current_positions(self) -> dict[str, float]:
        state = read_json(self.state_path, {}) or {}
        return {k: float(v) for k, v in state.get("positions", {}).items()}

    def account_equity(self) -> float:
        state = read_json(self.state_path, {}) or {}
        return float(state.get("equity", self._equity))

    def set_prices(self, prices: dict[str, float]) -> None:
        self._prices = dict(prices)

    # ----------------------------------------------------------------------- submit

    def submit(self, orders: Sequence[Order], meta: RunMeta) -> ExecutionReport:
        orders = list(orders)
        frame = orders_to_frame(orders, self._prices)

        orders_path = self.output_dir / "orders.csv"
        frame.to_csv(orders_path, index=False)

        targets_path = self.output_dir / "targets.json"
        write_json(
            targets_path,
            {
                "run_id": meta.run_id,
                "market": meta.market,
                "decision_session": str(meta.decision_session),
                "entry_session": str(meta.entry_session),
                "currency": meta.currency,
                "equity": meta.equity,
                "orders": json.loads(frame.to_json(orient="records")),
                **meta.extra,
            },
        )

        after = positions_after(self.current_positions(), orders)
        write_json(
            self.state_path,
            {
                "as_of": str(meta.entry_session),
                "equity": meta.equity,
                "positions": after,
                "note": (
                    "Assumed fills at the next session's open. Reconcile against your "
                    "broker statement before the next run; a drift here silently "
                    "corrupts every future order diff."
                ),
            },
        )

        summary = summarise_orders(orders, self._prices)
        log.info(
            "wrote %d order(s) to %s (gross %.0f %s)",
            summary["n_orders"], orders_path, summary["gross_value"], meta.currency,
        )
        return ExecutionReport(
            adapter=self.name,
            submitted=len(orders),
            accepted=len(orders),
            artifacts={
                "orders_csv": str(orders_path),
                "targets_json": str(targets_path),
                "positions_json": str(self.state_path),
            },
            messages=[
                f"{summary['n_orders']} order(s), "
                f"buy {summary['buy_value']:,.0f} / sell {summary['sell_value']:,.0f} {meta.currency}"
            ],
        )

    def close(self) -> None:
        return None


class PaperExecutionAdapter(CSVExecutionAdapter):
    """Same files, plus a running trade blotter.

    Useful for building a track record before risking money: the blotter accumulates
    every assumed fill so realised performance can be compared against what the backtest
    predicted. A live-versus-backtest gap is usually slippage, and it is much better to
    discover that on paper.
    """

    name = "paper"

    def submit(self, orders: Sequence[Order], meta: RunMeta) -> ExecutionReport:
        report = super().submit(orders, meta)

        blotter_path = self.output_dir / "blotter.csv"
        frame = orders_to_frame(list(orders), self._prices)
        if not frame.empty:
            frame.insert(0, "entry_session", str(meta.entry_session))
            frame.insert(0, "run_id", meta.run_id)
            header = not blotter_path.exists()
            frame.to_csv(blotter_path, mode="a", header=header, index=False)
            report.artifacts["blotter_csv"] = str(blotter_path)
        report.adapter = self.name
        return report


def build_adapter(cfg, output_dir: Path | str, *, equity: float | None = None):
    """Construct the adapter named in ``cfg.execution.adapter``."""
    kind = cfg.execution.adapter.strip().lower()
    start_equity = equity if equity is not None else cfg.backtest.initial_equity
    if kind == "csv":
        return CSVExecutionAdapter(output_dir, equity=start_equity)
    if kind == "paper":
        return PaperExecutionAdapter(output_dir, equity=start_equity)
    raise ValueError(
        f"Unknown execution adapter {cfg.execution.adapter!r}. Known: csv, paper. "
        "A broker adapter would be a new class in swingbot.execution implementing "
        "ExecutionAdapter; nothing else in the system would change."
    )

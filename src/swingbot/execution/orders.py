"""Target weights to orders.

Pure arithmetic with no broker vocabulary anywhere, which is what lets the same function
serve a CSV file, a paper book or a real broker adapter.

Two details that matter in practice. Share counts are rounded to whole units (and to lot
sizes where a market uses them), because fractional shares are not universally available
and a backtest that assumes them overstates how precisely a small account can track its
targets. And an order smaller than a configurable minimum is dropped rather than sent: a
two-share trade pays a full round trip to move the portfolio by nothing.
"""

from __future__ import annotations

import math

import pandas as pd

from ..types import Order, OrderType, Side, TargetPortfolio


def build_orders(
    portfolio: TargetPortfolio,
    prices: dict[str, float],
    equity: float,
    current_positions: dict[str, float] | None = None,
    *,
    lot_size: int = 1,
    lot_sizes: dict[str, int] | None = None,
    derivative_lot_size: int = 1,
    min_order_value: float = 0.0,
    tag: str = "",
    notes: list[str] | None = None,
) -> list[Order]:
    """Diff the target book against current holdings and emit the trades.

    ``lot_sizes`` is a per-symbol tradeable increment and takes precedence over the
    market-wide ``lot_size`` (cash equity) and ``derivative_lot_size`` (everything else).
    Per symbol because that is how exchanges actually define it: NSE sets a lot per
    single-stock future and revises it, so one market-wide number would be wrong for most
    names — and rounding a 250-lot symbol to a 500 lot would *double* the intended
    exposure, which is worse than not rounding at all.

    Where a derivative's increment is unknown the quantity is left unrounded and the
    order is tagged ``lot-unknown``, so the ticket says "check this" instead of quietly
    carrying a number no venue will accept.

    ``notes``, when given, is appended to with any position the account is too small to
    express at all — a target below one lot. Passing a list is how the caller opts into
    hearing about it; the default is to stay silent, which keeps this a pure order
    producer for the tests that only care about the tickets.
    """
    current = dict(current_positions or {})
    per_symbol = dict(lot_sizes or {})
    orders: list[Order] = []
    unreachable = notes if notes is not None else []

    targets = {p.ticker: p for p in portfolio.positions}
    instruments = {p.ticker: p.instrument for p in portfolio.positions}

    for ticker in sorted(set(targets) | set(current)):
        price = prices.get(ticker)
        if not price or price <= 0:
            continue

        position = targets.get(ticker)
        target_weight = position.weight if position else 0.0
        instrument = instruments.get(ticker, "equity")
        is_derivative = instrument not in ("equity", "equity_short")

        increment, unknown = _increment_for(
            ticker,
            is_derivative=is_derivative,
            per_symbol=per_symbol,
            equity_lot=lot_size,
            derivative_lot=derivative_lot_size,
        )

        target_shares = _round_lot(target_weight * equity / price, increment)
        held = current.get(ticker, 0.0)
        delta = target_shares - held

        # A position the account is too small to express.
        #
        # Rounding toward zero means a target below one lot becomes no position at all,
        # and it would otherwise disappear without a word. This matters most exactly where
        # lots are largest: a 500-lot future at 166 is 83,000 of notional, which on a
        # 500,000 account is 16 percent — past any sane per-name cap, so the model can
        # never ask for a tradeable size in that name. Better to say "this account cannot
        # hold this position" than to leave someone wondering why the book has a hole.
        if target_shares == 0.0 and abs(target_weight) > 1e-9 and increment > 1:
            one_lot = increment * price
            unreachable.append(
                f"{ticker}: target {abs(target_weight):.2%} is below one lot "
                f"({increment:,} x {price:,.2f} = {one_lot:,.0f}, "
                f"{one_lot / equity:.1%} of equity) — not tradeable at this size"
            )

        if abs(delta) < 1e-9:
            continue
        if abs(delta) * price < min_order_value:
            continue

        tags = [tag] if tag else []
        if unknown:
            tags.append("lot-unknown")

        orders.append(
            Order(
                ticker=ticker,
                side=Side.BUY if delta > 0 else Side.SELL,
                quantity=abs(delta),
                order_type=OrderType.MARKET,
                instrument=instrument,
                client_order_id=f"{portfolio.entry_session:%Y%m%d}-{ticker}",
                tag=" ".join(tags),
            )
        )
    return orders


def _increment_for(
    ticker: str,
    *,
    is_derivative: bool,
    per_symbol: dict[str, int],
    equity_lot: int,
    derivative_lot: int,
) -> tuple[int, bool]:
    """The tradeable increment for one symbol, and whether it had to be guessed.

    Returns ``(increment, unknown)``. ``unknown`` is True only for a derivative with no
    declared increment: cash equity at 1 is genuinely correct, whereas a future at 1 is
    an unplaceable order waiting to be rejected.

    The per-symbol table is a table of **F&O lots**, so it applies only to derivatives.
    Letting it reach cash equity was a real bug and a destructive one: BRITANNIA's futures
    lot is 125, and applying it to a delivery buy turned a perfectly legal 70-share
    purchase into "below one lot, not tradeable". The cash segment trades in single
    shares; only the derivative has a lot.
    """
    if is_derivative:
        declared = per_symbol.get(ticker)
        if declared and declared > 1:
            return int(declared), False
        if derivative_lot > 1:
            return int(derivative_lot), False
        return 1, True
    return max(1, int(equity_lot)), False


def _round_lot(shares: float, lot_size: int) -> float:
    """Round toward zero to a whole lot.

    Toward zero rather than nearest, so rounding never increases exposure beyond the
    target. Erring small is free; erring large breaches a limit that was carefully set.
    """
    if lot_size <= 1:
        return float(math.floor(abs(shares)) * (1 if shares >= 0 else -1))
    lots = math.floor(abs(shares) / lot_size)
    return float(lots * lot_size * (1 if shares >= 0 else -1))


def orders_to_frame(orders: list[Order], prices: dict[str, float] | None = None) -> pd.DataFrame:
    """Order list as a tidy frame, ready for CSV."""
    if not orders:
        return pd.DataFrame(
            columns=["ticker", "side", "quantity", "order_type", "instrument",
                     "limit_price", "est_price", "est_value", "client_order_id", "tag"]
        )
    prices = prices or {}
    return pd.DataFrame(
        [
            {
                "ticker": o.ticker,
                "side": o.side.value,
                "quantity": o.quantity,
                "order_type": o.order_type.value,
                "instrument": o.instrument,
                "limit_price": o.limit_price,
                "est_price": prices.get(o.ticker),
                "est_value": (prices.get(o.ticker) or 0.0) * o.quantity,
                "client_order_id": o.client_order_id,
                # Carried into the file because it is where `lot-unknown` lives, and a
                # flag that never reaches the ticket is not a flag.
                "tag": o.tag,
            }
            for o in orders
        ]
    ).sort_values(["side", "ticker"]).reset_index(drop=True)


def positions_after(
    current: dict[str, float], orders: list[Order]
) -> dict[str, float]:
    """Holdings implied once the orders fill. Used by the paper book."""
    out = dict(current)
    for order in orders:
        out[order.ticker] = out.get(order.ticker, 0.0) + order.signed_quantity
    return {k: v for k, v in out.items() if abs(v) > 1e-9}


def summarise_orders(orders: list[Order], prices: dict[str, float]) -> dict:
    frame = orders_to_frame(orders, prices)
    if frame.empty:
        return {"n_orders": 0, "buy_value": 0.0, "sell_value": 0.0, "gross_value": 0.0}
    buys = frame.loc[frame["side"] == "buy", "est_value"].sum()
    sells = frame.loc[frame["side"] == "sell", "est_value"].sum()
    return {
        "n_orders": len(frame),
        "buy_value": float(buys),
        "sell_value": float(sells),
        "gross_value": float(buys + sells),
    }


# --------------------------------------------------------------------------------------
# Working the list
# --------------------------------------------------------------------------------------

#: Why each step in the sequence comes where it does. Shown in the UI next to the row.
SEQUENCE_REASONS = {
    "close": "closing a position — frees capital and cuts risk immediately",
    "reduce": "trimming a position — reduces risk, no new capital needed",
    "open_short": "opening a short — no cash required, and the borrow may go away",
    "open_long": "spending capital — last, once you know what the sells freed up",
}


def execution_sequence(orders: list[Order], prices: dict[str, float]) -> list[dict]:
    """Order the tickets the way a desk would actually work them.

    ``orders_to_frame`` sorts by side then ticker, which puts every buy first purely
    because "buy" sorts before "sell". That is close to the worst possible order for
    anyone without margin: the buys demand cash that the sells have not yet released, and
    the first rejection arrives on order one.

    The sequence here is the one a human working a rebalance by hand wants:

    1. **Closes** — a position going to zero. Releases the most capital per click and
       takes risk off the book straight away.
    2. **Reductions** — trimming an existing holding. Same direction of travel, less of it.
    3. **New shorts** — no cash needed, and short availability is the thing most likely to
       have vanished by the time you get to it, so find out early.
    4. **New longs** — the only step that spends money, done last when the cash actually
       freed by steps 1 and 2 is known rather than assumed.

    Within each step, largest notional first: a partial execution should be the small
    trades left undone, not the large ones.

    This changes nothing about *what* is traded. It is purely the order in which a person
    works down the list, which is exactly the part a file sorted for tidiness gets wrong.
    """
    if not orders:
        return []

    rows = []
    for order in orders:
        price = prices.get(order.ticker) or 0.0
        notional = price * order.quantity
        rows.append({"order": order, "notional": notional})

    def step(order: Order) -> str:
        # Without a position ledger here, side plus instrument is the available signal:
        # a sell of cash equity reduces or closes a long, a sell of a short instrument
        # opens or adds to a short.
        if order.instrument in ("futures", "equity_short"):
            return "open_short" if order.side is Side.SELL else "close"
        return "reduce" if order.side is Side.SELL else "open_long"

    priority = {"close": 0, "reduce": 1, "open_short": 2, "open_long": 3}
    rows.sort(key=lambda r: (priority[step(r["order"])], -r["notional"]))

    return [
        {
            "position": index + 1,
            "client_order_id": row["order"].client_order_id,
            "ticker": row["order"].ticker,
            "step": step(row["order"]),
            "reason": SEQUENCE_REASONS[step(row["order"])],
            "notional": row["notional"],
            # Signed cash effect: negative spends, positive releases. A short sale
            # releases no cash in a margin account, so it counts as zero rather than as
            # a credit — assuming otherwise is how a buying-power rejection happens.
            "cash_effect": (
                row["notional"]
                if step(row["order"]) in ("close", "reduce")
                else -row["notional"] if step(row["order"]) == "open_long" else 0.0
            ),
        }
        for index, row in enumerate(rows)
    ]

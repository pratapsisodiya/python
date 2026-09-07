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
    min_order_value: float = 0.0,
    tag: str = "",
) -> list[Order]:
    """Diff the target book against current holdings and emit the trades."""
    current = dict(current_positions or {})
    orders: list[Order] = []

    targets = {p.ticker: p for p in portfolio.positions}
    instruments = {p.ticker: p.instrument for p in portfolio.positions}

    for ticker in sorted(set(targets) | set(current)):
        price = prices.get(ticker)
        if not price or price <= 0:
            continue

        position = targets.get(ticker)
        target_weight = position.weight if position else 0.0
        target_shares = _round_lot(target_weight * equity / price, lot_size)
        held = current.get(ticker, 0.0)
        delta = target_shares - held

        if abs(delta) < 1e-9:
            continue
        if abs(delta) * price < min_order_value:
            continue

        orders.append(
            Order(
                ticker=ticker,
                side=Side.BUY if delta > 0 else Side.SELL,
                quantity=abs(delta),
                order_type=OrderType.MARKET,
                instrument=instruments.get(ticker, "equity"),
                client_order_id=f"{portfolio.entry_session:%Y%m%d}-{ticker}",
                tag=tag,
            )
        )
    return orders


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
                     "limit_price", "est_price", "est_value", "client_order_id"]
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

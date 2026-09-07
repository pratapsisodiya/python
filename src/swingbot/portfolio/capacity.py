"""Capacity constraints.

A backtest that trades 20 percent of a name's daily volume is not describing something
anyone could have done. Truncating orders at a participation cap is what turns a
"strategy" into a strategy with a stated capital limit, and reporting the truncation is
what tells you when you have hit it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def participation_capped_weights(
    target: pd.Series,
    previous: pd.Series,
    adv_notional: pd.Series,
    equity: float,
    *,
    max_participation: float = 0.05,
) -> tuple[pd.Series, dict[str, float]]:
    """Truncate weight *changes* that would exceed a share of average daily volume.

    The cap applies to the trade, not the position: holding a large position in a liquid
    name is fine, acquiring it in one session is not.
    """
    if target.empty or equity <= 0:
        return target, {}

    index = target.index.union(previous.index)
    tgt = target.reindex(index).fillna(0.0)
    prev = previous.reindex(index).fillna(0.0)
    adv = adv_notional.reindex(index)

    desired_notional = (tgt - prev).abs() * equity
    max_notional = adv.fillna(np.inf) * max_participation

    truncated: dict[str, float] = {}
    out = tgt.copy()
    breach = desired_notional > max_notional
    for ticker in index[breach.fillna(False)]:
        allowed = float(max_notional.get(ticker, 0.0)) / equity
        direction = np.sign(tgt[ticker] - prev[ticker])
        out[ticker] = prev[ticker] + direction * allowed
        truncated[str(ticker)] = float(
            abs(tgt[ticker] - prev[ticker]) - allowed
        )

    return out[out.abs() > 1e-9], truncated


def liquidity_eligible(
    panel: pd.DataFrame,
    *,
    min_price: float,
    min_adv_notional: float,
    price_column: str = "close",
    adv_column: str = "adv_notional",
) -> pd.Series:
    """Boolean eligibility from trailing price and turnover."""
    if panel.empty:
        return pd.Series(dtype=bool)
    price_ok = panel[price_column] >= min_price if price_column in panel else True
    adv_ok = (
        panel[adv_column].fillna(0.0) >= min_adv_notional if adv_column in panel else True
    )
    return pd.Series(price_ok & adv_ok, index=panel.index)


def hard_to_borrow_filter(
    weights: pd.Series, hard_to_borrow: set[str] | None
) -> tuple[pd.Series, list[str]]:
    """Remove shorts in names that cannot be borrowed.

    A constraint, not a cost. Paying a high borrow rate is a trade; shorting something
    nobody will lend you is not.
    """
    if not hard_to_borrow:
        return weights, []
    blocked = [t for t in weights.index if weights[t] < 0 and str(t) in hard_to_borrow]
    if not blocked:
        return weights, []
    out = weights.drop(index=blocked)
    return out, blocked

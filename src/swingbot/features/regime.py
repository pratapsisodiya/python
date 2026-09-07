"""Market regime features.

Every name on a given date shares these values, so they carry no cross-sectional
information on their own. They earn their place through interactions: a breakout in a
calm, broad-based uptrend is a different proposition from the same breakout during a
volatility spike, and a tree model can only learn that if the regime is in the feature
matrix.

The regime is computed from the universe's own equal-weighted return rather than from an
index series, so it needs no extra data source and stays consistent between India and
the US.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..types import CLOSE, SESSION, TICKER
from .base import BaseTransformer, ensure_sorted, register


@register
class RegimeFeatures(BaseTransformer):
    """Trend, volatility and breadth of the universe itself."""

    name = "regime"
    outputs = (
        "mkt_mom_4w",
        "mkt_mom_12w",
        "mkt_vol_13w",
        "mkt_vol_ratio",
        "breadth_above_sma50",
        "breadth_up_1w",
        "dispersion_1w",
        "mkt_drawdown",
    )
    warmup_sessions = 260

    def transform(self, panel: pd.DataFrame) -> pd.DataFrame:
        p = ensure_sorted(panel)
        if p.empty:
            return self._frame(p, dict.fromkeys(self.outputs, pd.Series(dtype=float)))

        log_close = np.log(p[CLOSE].astype(float).where(lambda s: s > 0))
        p = p.assign(_logc=log_close)
        p = p.assign(_ret=p["_logc"] - p.groupby(TICKER, sort=False)["_logc"].shift(1))

        # Equal-weighted market return per session. Using the mean of available names
        # rather than a fixed index keeps this defined even when coverage changes.
        market = p.groupby(SESSION, sort=True)["_ret"].mean()
        market_index = market.fillna(0.0).cumsum()

        mkt_mom_4w = market_index - market_index.shift(20)
        mkt_mom_12w = market_index - market_index.shift(60)
        mkt_vol_13w = market.rolling(65, min_periods=40).std() * np.sqrt(252.0)
        mkt_vol_26w = market.rolling(130, min_periods=80).std() * np.sqrt(252.0)
        running_max = market_index.cummax()
        mkt_drawdown = market_index - running_max

        # Breadth: how much of the universe participates. A rally carried by three names
        # behaves very differently from a broad one.
        sma50 = p.groupby(TICKER, sort=False)[CLOSE].transform(
            lambda s: s.rolling(50, min_periods=35).mean()
        )
        p = p.assign(_above=(p[CLOSE] > sma50).astype(float).where(sma50.notna()))
        breadth_sma = p.groupby(SESSION, sort=True)["_above"].mean()

        p = p.assign(_up=(p["_ret"] > 0).astype(float).where(p["_ret"].notna()))
        breadth_up = p.groupby(SESSION, sort=True)["_up"].mean().rolling(5, min_periods=3).mean()

        # Cross-sectional dispersion. High dispersion means stock selection has more to
        # work with; low dispersion means everything is moving on one factor.
        dispersion = (
            p.groupby(SESSION, sort=True)["_ret"].std().rolling(5, min_periods=3).mean()
            * np.sqrt(252.0)
        )

        per_session = pd.DataFrame(
            {
                "mkt_mom_4w": mkt_mom_4w,
                "mkt_mom_12w": mkt_mom_12w,
                "mkt_vol_13w": mkt_vol_13w,
                "mkt_vol_ratio": mkt_vol_13w / mkt_vol_26w.replace(0.0, np.nan),
                "breadth_above_sma50": breadth_sma,
                "breadth_up_1w": breadth_up,
                "dispersion_1w": dispersion,
                "mkt_drawdown": mkt_drawdown,
            }
        )
        joined = p[[SESSION]].join(per_session, on=SESSION)
        return self._frame(p, {name: joined[name] for name in self.outputs})

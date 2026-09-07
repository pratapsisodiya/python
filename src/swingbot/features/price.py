"""Price and volatility features.

The workhorse block. These are the effects with the most published evidence at a weekly
horizon, and between them they are most of the 90 percent of the forecast that comes
from price history rather than news.

Every window is expressed in **sessions**, and every computation reads only trailing
rows. Where a feature would naturally use the current bar's own value in a normalising
statistic, it is shifted, because using a bar to normalise itself is a mild but real
form of leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..types import CLOSE, HIGH, LOW, TICKER, VOLUME
from .base import BaseTransformer, ensure_sorted, grouped_rolling, grouped_shift, register


@register
class PriceFeatures(BaseTransformer):
    """Momentum, reversal, volatility and volume features."""

    name = "price"
    outputs = (
        "mom_4w",
        "mom_12w",
        "mom_26w",
        "mom_52w_skip4w",
        "reversal_1w",
        "reversal_2w",
        "vol_13w",
        "vol_26w",
        "vol_ratio",
        "downside_vol_13w",
        "range_pct_4w",
        "dist_52w_high",
        "dist_52w_low",
        "volume_z_13w",
        "dollar_vol_log",
        "gap_1d",
        "close_to_high_1w",
    )
    warmup_sessions = 260

    def transform(self, panel: pd.DataFrame) -> pd.DataFrame:
        p = ensure_sorted(panel)
        close = p[CLOSE].astype(float)
        log_close = np.log(close.where(close > 0))
        p = p.assign(_logc=log_close)

        daily_ret = p["_logc"] - grouped_shift(p, "_logc", 1)
        p = p.assign(_ret=daily_ret)

        def momentum(sessions: int) -> pd.Series:
            return p["_logc"] - grouped_shift(p, "_logc", sessions)

        # Classic 12-1: a year of momentum excluding the most recent month, because the
        # recent part reverses at short horizons and cancels the longer-run effect.
        mom_52 = momentum(252)
        mom_4 = momentum(20)
        mom_52_skip = mom_52 - mom_4

        vol_13 = grouped_rolling(p, "_ret", 65, "std", min_periods=40) * np.sqrt(252.0)
        vol_26 = grouped_rolling(p, "_ret", 130, "std", min_periods=80) * np.sqrt(252.0)

        downside = p["_ret"].where(p["_ret"] < 0.0, 0.0)
        p = p.assign(_down=downside)
        downside_vol = grouped_rolling(p, "_down", 65, "std", min_periods=40) * np.sqrt(252.0)

        roll_max = grouped_rolling(p, CLOSE, 252, "max", min_periods=120)
        roll_min = grouped_rolling(p, CLOSE, 252, "min", min_periods=120)

        high_4w = grouped_rolling(p, HIGH, 20, "max", min_periods=15)
        low_4w = grouped_rolling(p, LOW, 20, "min", min_periods=15)

        # Volume z-score against its own trailing distribution, shifted so the current
        # bar's volume is not part of the mean it is being compared against.
        log_volume = np.log1p(p[VOLUME].astype(float).clip(lower=0.0))
        p = p.assign(_logv=log_volume)
        vol_mean = grouped_shift(p.assign(_m=grouped_rolling(p, "_logv", 65, "mean", min_periods=40)), "_m", 1)
        vol_std = grouped_shift(p.assign(_s=grouped_rolling(p, "_logv", 65, "std", min_periods=40)), "_s", 1)
        volume_z = (p["_logv"] - vol_mean) / vol_std.replace(0.0, np.nan)

        prev_close = grouped_shift(p, CLOSE, 1)
        gap = np.log(p["open"].astype(float) / prev_close.replace(0.0, np.nan))

        week_high = grouped_rolling(p, HIGH, 5, "max", min_periods=3)
        week_low = grouped_rolling(p, LOW, 5, "min", min_periods=3)
        week_span = (week_high - week_low).replace(0.0, np.nan)

        return self._frame(
            p,
            {
                "mom_4w": mom_4,
                "mom_12w": momentum(60),
                "mom_26w": momentum(130),
                "mom_52w_skip4w": mom_52_skip,
                "reversal_1w": -(p["_logc"] - grouped_shift(p, "_logc", 5)),
                "reversal_2w": -(p["_logc"] - grouped_shift(p, "_logc", 10)),
                "vol_13w": vol_13,
                "vol_26w": vol_26,
                "vol_ratio": vol_13 / vol_26.replace(0.0, np.nan),
                "downside_vol_13w": downside_vol,
                "range_pct_4w": (high_4w - low_4w) / p[CLOSE].replace(0.0, np.nan),
                "dist_52w_high": p[CLOSE] / roll_max.replace(0.0, np.nan) - 1.0,
                "dist_52w_low": p[CLOSE] / roll_min.replace(0.0, np.nan) - 1.0,
                "volume_z_13w": volume_z,
                "dollar_vol_log": np.log1p(p[CLOSE].astype(float) * p[VOLUME].astype(float)),
                "gap_1d": gap,
                "close_to_high_1w": (p[CLOSE] - week_low) / week_span,
            },
        )


@register
class ReturnMoments(BaseTransformer):
    """Higher moments of the trailing return distribution.

    Idiosyncratic skewness is a documented cross-sectional effect: lottery-like stocks
    are systematically overpriced. Kurtosis picks out names whose recent history is
    dominated by one jump, where a volatility estimate understates the real risk.
    """

    name = "moments"
    outputs = ("skew_13w", "kurt_13w", "max_ret_1m", "min_ret_1m", "up_day_frac_13w")
    warmup_sessions = 90

    def transform(self, panel: pd.DataFrame) -> pd.DataFrame:
        p = ensure_sorted(panel)
        log_close = np.log(p[CLOSE].astype(float).where(lambda s: s > 0))
        p = p.assign(_logc=log_close)
        p = p.assign(_ret=p["_logc"] - grouped_shift(p, "_logc", 1))

        grouped = p.groupby(TICKER, sort=False)["_ret"]
        skew = grouped.transform(lambda s: s.rolling(65, min_periods=40).skew())
        kurt = grouped.transform(lambda s: s.rolling(65, min_periods=40).kurt())
        max_ret = grouped.transform(lambda s: s.rolling(21, min_periods=15).max())
        min_ret = grouped.transform(lambda s: s.rolling(21, min_periods=15).min())

        p = p.assign(_up=(p["_ret"] > 0).astype(float).where(p["_ret"].notna()))
        up_frac = grouped_rolling(p, "_up", 65, "mean", min_periods=40)

        return self._frame(
            p,
            {
                "skew_13w": skew,
                "kurt_13w": kurt,
                "max_ret_1m": max_ret,
                "min_ret_1m": min_ret,
                "up_day_frac_13w": up_frac,
            },
        )

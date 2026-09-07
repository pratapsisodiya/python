"""Chart structure features.

What a chart reader actually looks at, turned into numbers: is the trend making higher
highs, has price broken out of a base, how far is the nearest resistance, is the range
compressing, did the gap fill.

These are the features most likely to be spurious if built carelessly, because "support"
and "resistance" are easy to define in a way that peeks at the future. A support level
found by looking at the whole series is not a level anyone could have drawn at the time.
Every level here is computed from a strictly trailing window, and every pivot requires
confirmation bars that have already printed, so a pivot is only recognised
``confirm`` sessions after it happened, exactly as a human would see it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..types import CLOSE, HIGH, LOW, OPEN, TICKER, VOLUME
from .base import BaseTransformer, ensure_sorted, grouped_rolling, grouped_shift, register


@register
class PatternFeatures(BaseTransformer):
    """Trend structure, breakouts, consolidation and gap behaviour."""

    name = "patterns"
    outputs = (
        "higher_highs_12w",
        "higher_lows_12w",
        "trend_structure",
        "breakout_20d",
        "breakout_60d",
        "breakdown_20d",
        "dist_resistance_60d",
        "dist_support_60d",
        "squeeze_ratio",
        "consolidation_20d",
        "gap_unfilled",
        "inside_bar_frac_4w",
        "body_ratio_1w",
        "volume_breakout_conf",
        "days_since_52w_high",
        "pullback_from_high_20d",
    )
    warmup_sessions = 140

    def __init__(self, *, confirm: int = 3) -> None:
        #: Sessions a pivot must survive before it is recognised. This is what keeps a
        #: pivot from being visible on the bar it forms, which would be look-ahead.
        self.confirm = confirm

    def transform(self, panel: pd.DataFrame) -> pd.DataFrame:
        p = ensure_sorted(panel)
        close = p[CLOSE].astype(float)
        high = p[HIGH].astype(float)
        low = p[LOW].astype(float)
        open_ = p[OPEN].astype(float)
        safe_close = close.replace(0.0, np.nan)

        # ------------------------------------------------- trend structure of pivots
        # Rolling extremes over the trailing 60 sessions, then compared against the
        # window before that. Higher highs *and* higher lows is an uptrend; the two
        # disagreeing is a broadening or narrowing range, which is worth distinguishing.
        high_60 = grouped_rolling(p, HIGH, 60, "max", min_periods=40)
        low_60 = grouped_rolling(p, LOW, 60, "min", min_periods=40)
        p = p.assign(_h60=high_60, _l60=low_60)
        prior_high_60 = grouped_shift(p, "_h60", 60)
        prior_low_60 = grouped_shift(p, "_l60", 60)

        higher_highs = (high_60 > prior_high_60).astype(float).where(prior_high_60.notna())
        higher_lows = (low_60 > prior_low_60).astype(float).where(prior_low_60.notna())
        trend_structure = higher_highs + higher_lows - 1.0  # +1 up, 0 mixed, -1 down

        # --------------------------------------------------------------- breakouts
        # The prior extreme must exclude the current bar, otherwise every new high is
        # trivially a "breakout" of a window that contains it.
        p = p.assign(_hh20=grouped_rolling(p, HIGH, 20, "max", min_periods=15))
        p = p.assign(_ll20=grouped_rolling(p, LOW, 20, "min", min_periods=15))
        p = p.assign(_hh60=grouped_rolling(p, HIGH, 60, "max", min_periods=40))
        prior_high_20 = grouped_shift(p, "_hh20", 1)
        prior_low_20 = grouped_shift(p, "_ll20", 1)
        prior_high_60_1 = grouped_shift(p, "_hh60", 1)

        breakout_20 = (close / prior_high_20.replace(0.0, np.nan)) - 1.0
        breakout_60 = (close / prior_high_60_1.replace(0.0, np.nan)) - 1.0
        breakdown_20 = (close / prior_low_20.replace(0.0, np.nan)) - 1.0

        # --------------------------------------------------- support and resistance
        # Confirmed pivots only: an extreme is recognised `confirm` sessions after it
        # printed, which is when a chart reader would first have been able to mark it.
        p = p.assign(_res=grouped_shift(p.assign(_r=grouped_rolling(p, HIGH, 60, "max", min_periods=40)), "_r", self.confirm))
        p = p.assign(_sup=grouped_shift(p.assign(_s=grouped_rolling(p, LOW, 60, "min", min_periods=40)), "_s", self.confirm))
        dist_resistance = (p["_res"] - close) / safe_close
        dist_support = (close - p["_sup"]) / safe_close

        # ------------------------------------------------------------- compression
        # A narrowing range before expansion is the classic coiling setup. Comparing a
        # short realised range against a longer one captures it without a pattern name.
        range_20 = grouped_rolling(p, HIGH, 20, "max", min_periods=15) - grouped_rolling(
            p, LOW, 20, "min", min_periods=15
        )
        range_60 = grouped_rolling(p, HIGH, 60, "max", min_periods=40) - grouped_rolling(
            p, LOW, 60, "min", min_periods=40
        )
        squeeze = range_20 / range_60.replace(0.0, np.nan)
        consolidation = range_20 / safe_close

        # -------------------------------------------------------------------- gaps
        prev_close = grouped_shift(p, CLOSE, 1)
        gap = (open_ - prev_close) / prev_close.replace(0.0, np.nan)
        # A gap is unfilled when the session never traded back to the prior close.
        gap_up_open = gap > 0.005
        gap_down_open = gap < -0.005
        unfilled = (
            (gap_up_open & (low > prev_close)).astype(float)
            - (gap_down_open & (high < prev_close)).astype(float)
        ).where(prev_close.notna())

        # ----------------------------------------------------------- candle shapes
        prior_high_1 = grouped_shift(p, HIGH, 1)
        prior_low_1 = grouped_shift(p, LOW, 1)
        inside = ((high <= prior_high_1) & (low >= prior_low_1)).astype(float)
        p = p.assign(_inside=inside.where(prior_high_1.notna()))
        inside_frac = grouped_rolling(p, "_inside", 20, "mean", min_periods=15)

        span = (high - low).replace(0.0, np.nan)
        p = p.assign(_body=(close - open_).abs() / span)
        body_ratio = grouped_rolling(p, "_body", 5, "mean", min_periods=3)

        # ------------------------------------------------- breakout confirmation
        # A breakout on below-average volume is the one most likely to fail, so the
        # interaction is given to the model explicitly rather than hoped for.
        vol_mean_20 = grouped_rolling(p, VOLUME, 20, "mean", min_periods=15)
        p = p.assign(_vm=vol_mean_20)
        vol_ratio = p[VOLUME].astype(float) / grouped_shift(p, "_vm", 1).replace(0.0, np.nan)
        volume_conf = np.sign(breakout_20.fillna(0.0)) * np.log1p(vol_ratio.clip(lower=0.0))

        # ----------------------------------------------------- distance from highs
        high_252 = grouped_rolling(p, CLOSE, 252, "max", min_periods=120)
        at_high = (close >= high_252 * 0.999).astype(float)
        p = p.assign(_athigh=at_high)
        days_since_high = p.groupby(TICKER, sort=False)["_athigh"].transform(
            _sessions_since_last_true
        )
        pullback = (close / grouped_rolling(p, CLOSE, 20, "max", min_periods=15).replace(0.0, np.nan)) - 1.0

        return self._frame(
            p,
            {
                "higher_highs_12w": higher_highs,
                "higher_lows_12w": higher_lows,
                "trend_structure": trend_structure,
                "breakout_20d": breakout_20,
                "breakout_60d": breakout_60,
                "breakdown_20d": breakdown_20,
                "dist_resistance_60d": dist_resistance,
                "dist_support_60d": dist_support,
                "squeeze_ratio": squeeze,
                "consolidation_20d": consolidation,
                "gap_unfilled": unfilled,
                "inside_bar_frac_4w": inside_frac,
                "body_ratio_1w": body_ratio,
                "volume_breakout_conf": volume_conf,
                "days_since_52w_high": days_since_high,
                "pullback_from_high_20d": pullback,
            },
        )


def _sessions_since_last_true(flags: pd.Series) -> pd.Series:
    """Sessions elapsed since the series last held 1.0, counting only past bars.

    Implemented as a cumulative-max of the index where the flag fired, which reads only
    backwards. Capped so an unbounded counter does not dominate a tree split.
    """
    values = flags.to_numpy(dtype=float)
    positions = np.arange(len(values), dtype=float)
    hit_positions = np.where(values > 0.5, positions, np.nan)
    last_hit = pd.Series(hit_positions).ffill().to_numpy()
    since = positions - last_hit
    since = np.where(np.isnan(last_hit), np.nan, since)
    return pd.Series(np.clip(since, 0.0, 252.0), index=flags.index)

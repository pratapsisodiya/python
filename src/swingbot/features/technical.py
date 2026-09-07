"""Classical technical indicators.

These are the chart tools a discretionary swing trader reads, expressed as numbers a
model can rank cross-sectionally. Two deliberate choices:

Indicators are emitted in **normalised** form. A raw MACD line is proportional to price
level, so ranking it across a universe would mostly rank price. Dividing by price or by
ATR makes it comparable between a 50-rupee stock and a 5000-rupee one, which is the only
form in which a cross-sectional model can use it.

Nothing here uses the current bar to normalise itself where that would be circular, and
every window is a trailing one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..types import CLOSE, HIGH, LOW, TICKER, VOLUME
from .base import (
    BaseTransformer,
    ensure_sorted,
    grouped_ewm,
    grouped_rolling,
    grouped_shift,
    register,
)


def _true_range(panel: pd.DataFrame) -> pd.Series:
    prev_close = grouped_shift(panel, CLOSE, 1)
    high, low = panel[HIGH].astype(float), panel[LOW].astype(float)
    return pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)


def _wilder_average(panel: pd.DataFrame, column: str, window: int) -> pd.Series:
    """Wilder's smoothing, which is an EWM with alpha = 1/window."""
    grouped = panel.groupby(TICKER, sort=False)[column]
    return grouped.transform(
        lambda s: s.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    )


@register
class TechnicalFeatures(BaseTransformer):
    """Moving averages, MACD, RSI, Bollinger, ATR, ADX and OBV, all normalised."""

    name = "technical"
    outputs = (
        "sma_ratio_10_50",
        "sma_ratio_20_200",
        "px_over_sma50",
        "px_over_sma200",
        "ema_slope_20",
        "macd_norm",
        "macd_hist_norm",
        "rsi_14",
        "rsi_14_centered",
        "stoch_k_14",
        "bb_position_20",
        "bb_width_20",
        "atr_pct_14",
        "adx_14",
        "di_spread_14",
        "obv_slope_20",
        "money_flow_14",
    )
    warmup_sessions = 220

    def transform(self, panel: pd.DataFrame) -> pd.DataFrame:
        p = ensure_sorted(panel)
        close = p[CLOSE].astype(float)
        high = p[HIGH].astype(float)
        low = p[LOW].astype(float)
        volume = p[VOLUME].astype(float)
        safe_close = close.replace(0.0, np.nan)

        # ---------------------------------------------------------- moving averages
        sma10 = grouped_rolling(p, CLOSE, 10, "mean", min_periods=8)
        sma20 = grouped_rolling(p, CLOSE, 20, "mean", min_periods=15)
        sma50 = grouped_rolling(p, CLOSE, 50, "mean", min_periods=35)
        sma200 = grouped_rolling(p, CLOSE, 200, "mean", min_periods=140)

        ema20 = grouped_ewm(p, CLOSE, 20)
        p = p.assign(_ema20=ema20)
        ema_slope = (ema20 - grouped_shift(p, "_ema20", 10)) / safe_close

        # ------------------------------------------------------------------- MACD
        ema12 = grouped_ewm(p, CLOSE, 12)
        ema26 = grouped_ewm(p, CLOSE, 26)
        macd = ema12 - ema26
        p = p.assign(_macd=macd)
        signal = grouped_ewm(p, "_macd", 9)

        # ------------------------------------------------------------------- RSI
        p = p.assign(_diff=close - grouped_shift(p, CLOSE, 1))
        gain = p["_diff"].clip(lower=0.0)
        loss = (-p["_diff"]).clip(lower=0.0)
        p = p.assign(_gain=gain, _loss=loss)
        avg_gain = _wilder_average(p, "_gain", 14)
        avg_loss = _wilder_average(p, "_loss", 14)
        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        rsi = 100.0 - 100.0 / (1.0 + rs)
        # A flat series has zero average loss, which is maximally overbought, not NaN.
        rsi = rsi.where(avg_loss.notna() & (avg_loss > 0), other=100.0 * (avg_gain > 0))

        # ------------------------------------------------------------ stochastic
        low14 = grouped_rolling(p, LOW, 14, "min", min_periods=10)
        high14 = grouped_rolling(p, HIGH, 14, "max", min_periods=10)
        stoch_span = (high14 - low14).replace(0.0, np.nan)
        stoch_k = 100.0 * (close - low14) / stoch_span

        # -------------------------------------------------------------- Bollinger
        std20 = grouped_rolling(p, CLOSE, 20, "std", min_periods=15)
        band = (2.0 * std20).replace(0.0, np.nan)
        bb_position = (close - sma20) / band
        bb_width = (4.0 * std20) / sma20.replace(0.0, np.nan)

        # -------------------------------------------------------------------- ATR
        p = p.assign(_tr=_true_range(p))
        atr = _wilder_average(p, "_tr", 14)
        atr_pct = atr / safe_close

        # -------------------------------------------------------------------- ADX
        up_move = high - grouped_shift(p, HIGH, 1)
        down_move = grouped_shift(p, LOW, 1) - low
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
        p = p.assign(_pdm=plus_dm, _mdm=minus_dm)
        atr_safe = atr.replace(0.0, np.nan)
        plus_di = 100.0 * _wilder_average(p, "_pdm", 14) / atr_safe
        minus_di = 100.0 * _wilder_average(p, "_mdm", 14) / atr_safe
        di_sum = (plus_di + minus_di).replace(0.0, np.nan)
        dx = 100.0 * (plus_di - minus_di).abs() / di_sum
        p = p.assign(_dx=dx)
        adx = _wilder_average(p, "_dx", 14)

        # -------------------------------------------------------------------- OBV
        direction = np.sign(p["_diff"].fillna(0.0))
        p = p.assign(_obv=(direction * volume).groupby(p[TICKER], sort=False).cumsum())
        obv_prior = grouped_shift(p, "_obv", 20)
        obv_scale = grouped_rolling(p, VOLUME, 20, "mean", min_periods=15).replace(0.0, np.nan)
        obv_slope = (p["_obv"] - obv_prior) / (obv_scale * 20.0)

        # ------------------------------------------------------------- money flow
        typical = (high + low + close) / 3.0
        raw_flow = typical * volume
        p = p.assign(_tp=typical, _flow=raw_flow)
        tp_diff = typical - grouped_shift(p, "_tp", 1)
        p = p.assign(
            _pos_flow=raw_flow.where(tp_diff > 0, 0.0),
            _neg_flow=raw_flow.where(tp_diff < 0, 0.0),
        )
        pos = grouped_rolling(p, "_pos_flow", 14, "sum", min_periods=10)
        neg = grouped_rolling(p, "_neg_flow", 14, "sum", min_periods=10)
        money_ratio = pos / neg.replace(0.0, np.nan)
        money_flow = 100.0 - 100.0 / (1.0 + money_ratio)

        return self._frame(
            p,
            {
                "sma_ratio_10_50": sma10 / sma50.replace(0.0, np.nan) - 1.0,
                "sma_ratio_20_200": sma20 / sma200.replace(0.0, np.nan) - 1.0,
                "px_over_sma50": close / sma50.replace(0.0, np.nan) - 1.0,
                "px_over_sma200": close / sma200.replace(0.0, np.nan) - 1.0,
                "ema_slope_20": ema_slope,
                "macd_norm": macd / safe_close,
                "macd_hist_norm": (macd - signal) / safe_close,
                "rsi_14": rsi,
                "rsi_14_centered": (rsi - 50.0) / 50.0,
                "stoch_k_14": stoch_k,
                "bb_position_20": bb_position,
                "bb_width_20": bb_width,
                "atr_pct_14": atr_pct,
                "adx_14": adx,
                "di_spread_14": (plus_di - minus_di) / 100.0,
                "obv_slope_20": obv_slope,
                "money_flow_14": money_flow,
            },
        )

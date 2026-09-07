"""Transaction costs, per market.

Costs are where most weekly strategies die, and they die quietly: a backtest that ignores
them shows a respectable Sharpe, and the same strategy nets to zero. At 60 percent weekly
turnover a 25 basis point round trip costs about 7.8 percent a year, which is more than
most weekly signals produce gross.

So the model is explicit and per-market:

**India (cash-segment delivery, the long sleeve).** Brokerage, STT at 0.1 percent on each
side, exchange transaction charges, stamp duty on the buy, and GST on the brokerage and
exchange fees. STT is the dominant term and it is charged on both sides, which is why
Indian weekly strategies need a much larger gross edge than US ones.

**India (single-stock futures, the short sleeve).** NSE delivery cannot be held short
overnight, so a weekly short is a futures position. Lower statutory charges but a roll
cost each month and a carry on the position.

**US.** Commission, regulatory fees on sells, and stock borrow on shorts.

**Both** additionally pay a half-spread, estimated from the stock's own high-low range
when no quote data is available, and a square-root impact term scaled by participation in
average daily volume. The square root is the standard empirical form: doubling order size
raises impact by about 40 percent, not 100.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import Config, CostConfig
from ..types import ShortInstrument

BPS = 1e-4


@dataclass(slots=True)
class CostBreakdown:
    """Every cost component, so a report can show where the money went."""

    commission: float = 0.0
    taxes: float = 0.0
    spread: float = 0.0
    impact: float = 0.0
    borrow: float = 0.0
    roll: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.commission + self.taxes + self.spread + self.impact + self.borrow + self.roll
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "commission": self.commission,
            "taxes": self.taxes,
            "spread": self.spread,
            "impact": self.impact,
            "borrow": self.borrow,
            "roll": self.roll,
            "total": self.total,
        }

    def __add__(self, other: CostBreakdown) -> CostBreakdown:
        return CostBreakdown(
            commission=self.commission + other.commission,
            taxes=self.taxes + other.taxes,
            spread=self.spread + other.spread,
            impact=self.impact + other.impact,
            borrow=self.borrow + other.borrow,
            roll=self.roll + other.roll,
        )


class CostModel:
    """Per-market transaction and holding costs."""

    def __init__(
        self,
        costs: CostConfig,
        *,
        short_instrument: ShortInstrument = ShortInstrument.CASH_EQUITY,
        scale: float = 1.0,
    ) -> None:
        self.c = costs
        self.short_instrument = short_instrument
        self.scale = scale

    @classmethod
    def from_config(cls, cfg: Config, *, scale: float | None = None) -> CostModel:
        return cls(
            cfg.costs,
            short_instrument=cfg.market_profile.short_instrument,
            scale=cfg.backtest.cost_scale if scale is None else scale,
        )

    # ------------------------------------------------------------------ trade cost

    def trade_cost(
        self,
        *,
        notional: float,
        is_buy: bool,
        is_short_leg: bool,
        volatility_annual: float = 0.30,
        adv_notional: float | None = None,
        high_low_range: float | None = None,
    ) -> CostBreakdown:
        """Cost of a single trade, in currency units."""
        if notional <= 0:
            return CostBreakdown()

        futures = is_short_leg and self.short_instrument is ShortInstrument.FUTURES

        if futures:
            commission_bps = self.c.futures_commission_bps
            exchange_bps = self.c.futures_exchange_bps
            stt_bps = self.c.futures_stt_sell_bps if not is_buy else 0.0
            stamp_bps = self.c.futures_stamp_duty_bps if is_buy else 0.0
        else:
            commission_bps = self.c.commission_bps
            exchange_bps = self.c.exchange_bps
            stt_bps = self.c.stt_buy_bps if is_buy else self.c.stt_sell_bps
            stamp_bps = self.c.stamp_duty_bps if is_buy else 0.0
            if not is_buy:
                stt_bps += self.c.regulatory_sell_bps

        commission = max(notional * commission_bps * BPS, self.c.min_commission)
        exchange = notional * exchange_bps * BPS
        # GST applies to brokerage and exchange charges, not to the statutory taxes.
        gst = (commission + exchange) * self.c.gst_rate
        taxes = notional * (stt_bps + stamp_bps) * BPS + exchange + gst

        spread = notional * self._half_spread_bps(high_low_range) * BPS
        impact = notional * self._impact_bps(notional, volatility_annual, adv_notional) * BPS

        return CostBreakdown(
            commission=commission * self.scale,
            taxes=taxes * self.scale,
            spread=spread * self.scale,
            impact=impact * self.scale,
        )

    def _half_spread_bps(self, high_low_range: float | None) -> float:
        """Half-spread, estimated from the stock's own trailing high-low range.

        A wide daily range implies a wide spread, and the relationship holds both
        across names and across time, which a flat per-market assumption does not:
        spreads widen exactly when a strategy most wants to trade.

        The coefficient is calibrated against observed effective spreads rather than
        picked for tidiness. A liquid US large cap runs a daily range near 200 basis
        points against an effective half-spread of one to two basis points, so the
        half-spread is on the order of one percent of the range, not a quarter of it.
        Getting this wrong is expensive in the flattering direction if too low and
        strategy-killing if too high: at 60 percent weekly turnover, every basis point
        of half-spread error moves annual return by about 1.2 percent.

        The floor still applies, so a preternaturally calm name is not assumed to trade
        for free.
        """
        if high_low_range is None or not np.isfinite(high_low_range) or high_low_range <= 0:
            return self.c.min_half_spread_bps
        estimated = high_low_range * 10_000.0 * self.c.spread_from_range_factor
        return float(max(self.c.min_half_spread_bps, min(estimated, 50.0)))

    def _impact_bps(
        self, notional: float, volatility_annual: float, adv_notional: float | None
    ) -> float:
        """Square-root market impact scaled by participation."""
        if not adv_notional or adv_notional <= 0:
            return 0.0
        participation = min(notional / adv_notional, 1.0)
        daily_vol_bps = (volatility_annual / np.sqrt(252.0)) * 10_000.0
        return float(self.c.impact_k * daily_vol_bps * np.sqrt(participation))

    # ---------------------------------------------------------------- holding cost

    def holding_cost(
        self, *, short_notional: float, sessions_held: int, rolls: int = 0
    ) -> CostBreakdown:
        """Borrow or futures carry on the short sleeve, accrued per session held."""
        if short_notional <= 0 or sessions_held <= 0:
            return CostBreakdown()
        daily = self.c.borrow_bps_annual * BPS / 252.0
        borrow = short_notional * daily * sessions_held
        roll = (
            short_notional * self.c.futures_roll_bps * BPS * rolls
            if self.short_instrument is ShortInstrument.FUTURES
            else 0.0
        )
        return CostBreakdown(borrow=borrow * self.scale, roll=roll * self.scale)

    # ------------------------------------------------------------------- portfolio

    def rebalance_cost(
        self,
        target: pd.Series,
        previous: pd.Series,
        equity: float,
        *,
        volatility: pd.Series | None = None,
        adv_notional: pd.Series | None = None,
        high_low_range: pd.Series | None = None,
        sessions_held: int = 5,
    ) -> tuple[CostBreakdown, float]:
        """Total cost of moving from ``previous`` to ``target``, and the turnover."""
        index = target.index.union(previous.index)
        tgt = target.reindex(index).fillna(0.0)
        prev = previous.reindex(index).fillna(0.0)
        delta = tgt - prev

        total = CostBreakdown()
        for ticker in index:
            change = float(delta.get(ticker, 0.0))
            if abs(change) < 1e-9:
                continue
            notional = abs(change) * equity
            # A short leg is one where the position is short on either side of the trade.
            is_short_leg = float(tgt.get(ticker, 0.0)) < 0 or float(prev.get(ticker, 0.0)) < 0
            total = total + self.trade_cost(
                notional=notional,
                is_buy=change > 0,
                is_short_leg=is_short_leg,
                volatility_annual=_lookup(volatility, ticker, 0.30),
                adv_notional=_lookup(adv_notional, ticker, None),
                high_low_range=_lookup(high_low_range, ticker, None),
            )

        short_notional = float(tgt[tgt < 0].abs().sum()) * equity
        if short_notional > 0:
            rolls = 1 if sessions_held >= 21 else 0
            total = total + self.holding_cost(
                short_notional=short_notional, sessions_held=sessions_held, rolls=rolls
            )

        turnover = float(delta.abs().sum() / 2.0)
        return total, turnover


def _lookup(series: pd.Series | None, key, default):
    if series is None or key not in series.index:
        return default
    value = series.get(key)
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return default
    return float(value)


def round_trip_bps(cfg: Config, *, is_short: bool = False) -> float:
    """Headline round-trip cost in basis points, for the report and for sanity checks."""
    model = CostModel.from_config(cfg, scale=1.0)
    notional = 1_000_000.0
    buy = model.trade_cost(
        notional=notional, is_buy=True, is_short_leg=is_short, adv_notional=1e8
    )
    sell = model.trade_cost(
        notional=notional, is_buy=False, is_short_leg=is_short, adv_notional=1e8
    )
    return float((buy.total + sell.total) / notional * 10_000.0)

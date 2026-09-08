"""Turn forecasts into a target book.

The long-short construction the user asked for: top-ranked names long, bottom-ranked
short, sized by inverse volatility, scaled to a volatility target, then passed through
the hard limits.

Two details that decide whether the strategy survives costs.

**The no-trade band.** Rebalancing a position from 4.1 percent to 4.2 percent pays a full
round trip to change nothing. Skipping changes below a threshold is the single
highest-return line of code in this module: at weekly frequency it typically removes a
third of turnover while barely touching the realised signal.

**Short instrument.** On NSE a weekly short cannot be cash equity, so each short position
records the instrument it assumes. The cost model prices it accordingly and the report
states it, which stops a backtest from quietly assuming a trade the market does not
allow.
"""

from __future__ import annotations

import logging

import pandas as pd

from ..config import Config
from ..types import (
    SECTOR,
    TICKER,
    ShortInstrument,
    TargetPortfolio,
    TargetPosition,
)
from .capacity import participation_capped_weights
from .risk import RiskLimits, apply_limits
from .sizing import (
    equal_weights,
    inverse_vol_weights,
    kelly_cap,
    scale_to_target_vol,
    shrunk_covariance,
)

log = logging.getLogger(__name__)


class PortfolioConstructor:
    """Scores in, target weights out."""

    def __init__(
        self,
        *,
        n_long: int = 8,
        n_short: int = 7,
        gross_leverage: float = 1.0,
        max_net_exposure: float = 0.30,
        target_vol_annual: float = 0.12,
        sizing: str = "inverse_vol",
        kelly_fraction: float = 0.25,
        max_weight: float = 0.12,
        max_sector_weight: float = 0.35,
        no_trade_band: float = 0.005,
        max_participation_adv: float = 0.05,
        allow_short: bool = True,
        short_instrument: ShortInstrument = ShortInstrument.CASH_EQUITY,
    ) -> None:
        self.n_long = n_long
        self.n_short = n_short if allow_short else 0
        self.target_vol_annual = target_vol_annual
        self.sizing = sizing
        self.kelly_fraction = kelly_fraction
        self.no_trade_band = no_trade_band
        self.max_participation_adv = max_participation_adv
        self.allow_short = allow_short
        self.short_instrument = short_instrument
        self.limits = RiskLimits(
            max_weight=max_weight,
            max_sector_weight=max_sector_weight,
            gross_leverage=gross_leverage,
            max_net_exposure=max_net_exposure,
        )

    # ---------------------------------------------------------------- construction

    @classmethod
    def from_config(cls, cfg: Config) -> PortfolioConstructor:
        profile = cfg.market_profile
        return cls(
            n_long=cfg.portfolio.n_long,
            n_short=cfg.portfolio.n_short,
            gross_leverage=cfg.portfolio.gross_leverage,
            max_net_exposure=cfg.portfolio.max_net_exposure,
            target_vol_annual=cfg.portfolio.target_vol_annual,
            sizing=cfg.portfolio.sizing,
            kelly_fraction=cfg.portfolio.kelly_fraction,
            max_weight=cfg.portfolio.max_weight,
            max_sector_weight=cfg.portfolio.max_sector_weight,
            no_trade_band=cfg.portfolio.no_trade_band,
            max_participation_adv=cfg.portfolio.max_participation_adv,
            allow_short=cfg.shorts_allowed,
            short_instrument=profile.short_instrument,
        )

    # ---------------------------------------------------------------------- public

    def build(
        self,
        scores: pd.Series,
        *,
        decision_session,
        entry_session,
        volatility: pd.Series | None = None,
        sectors: pd.Series | None = None,
        returns_history: pd.DataFrame | None = None,
        previous_weights: dict[str, float] | None = None,
        risk_scale: float = 1.0,
        eligible: pd.Series | None = None,
        adv_notional: pd.Series | None = None,
        equity: float | None = None,
        expected_returns: pd.Series | None = None,
    ) -> TargetPortfolio:
        """Build this week's target book."""
        notes: list[str] = []
        portfolio = TargetPortfolio(
            decision_session=decision_session,
            entry_session=entry_session,
            risk_scale=risk_scale,
        )

        clean = scores.dropna()
        if eligible is not None:
            keep = eligible.reindex(clean.index).fillna(False)
            dropped = int((~keep).sum())
            if dropped:
                notes.append(f"{dropped} name(s) excluded as untradeable")
            clean = clean.loc[keep.astype(bool)]

        if clean.empty:
            portfolio.notes = notes + ["no eligible names"]
            return portfolio

        if risk_scale <= 1e-9:
            portfolio.notes = notes + ["kill-switch armed: flat"]
            return portfolio

        selected = self._select(clean)
        if selected.empty:
            portfolio.notes = notes + ["no names selected"]
            return portfolio

        weights, sizing_notes = self._size(
            selected, volatility, returns_history, expected_returns
        )
        notes.extend(sizing_notes)
        weights = weights * risk_scale

        weights, limit_notes = apply_limits(
            weights, limits=self.limits, sectors=sectors
        )
        notes.extend(limit_notes)

        if previous_weights:
            weights, band_note = self._apply_no_trade_band(weights, previous_weights)
            if band_note:
                notes.append(band_note)

            # The limits are re-applied after the band, and they are the final authority.
            #
            # The band substitutes last week's weight wherever the change was too small
            # to be worth paying for, and last week's weight was sized against last
            # week's volatility and last week's book. Substituting it can therefore push
            # gross or a sector back over its cap — observed at 100.96% gross against a
            # 100% limit before this second pass existed.
            #
            # Re-capping costs a little of what the band saves, because trimming a held
            # position is itself a trade. That is the right trade to make: turnover is an
            # expense, but a breached concentration limit is the thing the limit exists
            # to prevent. For a book already inside its limits this pass is a no-op and
            # the band keeps its full benefit.
            weights, recheck_notes = apply_limits(
                weights, limits=self.limits, sectors=sectors
            )
            notes.extend(n for n in recheck_notes if n not in limit_notes)

        # Capacity: truncate any weight CHANGE that would demand more of a name's daily
        # volume than the participation cap allows.
        #
        # Applied to the change, not the position: holding a large position in a liquid
        # name is fine, acquiring it in a single session is not. A backtest that trades
        # 20 percent of a name's daily volume is not describing something anyone could
        # have executed, and the cap is what converts "this strategy works" into "this
        # strategy works up to this much capital".
        #
        # Ordering repeats the lesson from the no-trade band: truncating a change leaves
        # the book off-target, so the hard limits are re-applied afterwards and stay the
        # final authority.
        if self.max_participation_adv > 0 and adv_notional is not None and equity:
            previous = pd.Series(previous_weights or {}, dtype=float)
            capped, truncated = participation_capped_weights(
                weights,
                previous,
                adv_notional,
                float(equity),
                max_participation=self.max_participation_adv,
            )
            if truncated:
                worst = sorted(truncated.items(), key=lambda kv: -kv[1])[:3]
                shown = ", ".join(f"{t} by {v:.2%}" for t, v in worst)
                notes.append(
                    f"{len(truncated)} order(s) truncated by the "
                    f"{self.max_participation_adv:.1%} participation cap ({shown})"
                )
                weights, capacity_notes = apply_limits(
                    capped, limits=self.limits, sectors=sectors
                )
                notes.extend(n for n in capacity_notes if n not in notes)
            else:
                weights = capped

        weights = weights[weights.abs() > 1e-9]

        sector_map = (
            sectors.reindex(weights.index).fillna("Unknown")
            if sectors is not None
            else pd.Series("Unknown", index=weights.index)
        )
        instrument = self._instrument_for
        portfolio.positions = [
            TargetPosition(
                ticker=str(ticker),
                weight=float(weight),
                score=float(clean.get(ticker, 0.0)),
                instrument=instrument(weight),
                sector=str(sector_map.get(ticker, "Unknown")),
            )
            for ticker, weight in weights.items()
        ]
        portfolio.notes = notes
        return portfolio

    # --------------------------------------------------------------------- private

    def _instrument_for(self, weight: float) -> str:
        if weight >= 0:
            return "equity"
        return (
            "futures" if self.short_instrument is ShortInstrument.FUTURES else "equity_short"
        )

    def _select(self, scores: pd.Series) -> pd.Series:
        """Top names long, bottom names short.

        Requires enough names to make the ranking meaningful. Picking the top eight of
        twelve is not a cross-sectional signal, it is nearly the whole universe.
        """
        ranked = scores.sort_values(ascending=False)
        min_needed = (self.n_long + self.n_short) * 2
        if len(ranked) < min_needed:
            available = max(1, len(ranked) // 4)
            n_long = min(self.n_long, available)
            n_short = min(self.n_short, available)
        else:
            n_long, n_short = self.n_long, self.n_short

        longs = ranked.head(n_long) if n_long > 0 else pd.Series(dtype=float)
        shorts = ranked.tail(n_short) if n_short > 0 else pd.Series(dtype=float)

        # Only take a position where the score actually points that way. A "short" whose
        # score is positive is just the least attractive long, and shorting it is a bet
        # the model never made.
        longs = longs[longs > 0]
        shorts = shorts[shorts < 0]

        signed = pd.concat([longs, -shorts.abs()])
        return signed[~signed.index.duplicated(keep="first")]

    def _size(
        self,
        selected: pd.Series,
        volatility: pd.Series | None,
        returns_history: pd.DataFrame | None,
        expected_returns: pd.Series | None = None,
    ) -> tuple[pd.Series, list[str]]:
        """Signed selections to weights, with the Kelly ceiling applied last.

        Order matters, and it used to be wrong. The Kelly cap ran *before* the gross
        normalisation and the volatility scaling, both of which multiply the whole vector.
        Whatever the cap took off was handed straight back on the next line: the book
        still changed shape, so the knob looked alive, but the resulting weights sat above
        the ceiling that had supposedly just been applied. Same failure as the volatility
        target being erased by the gross renormalisation in ``apply_limits``, and the
        reason ``test_kelly_ceiling_is_never_exceeded`` asserts the bound directly rather
        than merely asserting that the setting does something.

        A ceiling has to be the last thing that touches the weights. So the sizing rule
        runs, the book is scaled to its gross and to its volatility target, and only then
        does Kelly trim whatever still exceeds its limit. Trimming only ever reduces, so
        nothing downstream can undo it.
        """
        notes: list[str] = []

        if self.sizing == "equal" or volatility is None or volatility.empty:
            weights = equal_weights(selected)
        else:
            weights = inverse_vol_weights(selected, volatility)

        weights = weights * self.limits.gross_leverage / max(weights.abs().sum(), 1e-12)

        if self.target_vol_annual > 0:
            covariance = pd.DataFrame()
            if returns_history is not None and not returns_history.empty:
                covariance = shrunk_covariance(
                    returns_history.reindex(columns=weights.index)
                )
            weights, scale = scale_to_target_vol(
                weights,
                covariance,
                target_vol=self.target_vol_annual,
                fallback_vol=volatility,
            )

        if self.sizing == "kelly":
            weights, kelly_note = self._apply_kelly(weights, volatility, expected_returns)
            if kelly_note:
                notes.append(kelly_note)

        return weights, notes

    def _apply_kelly(
        self,
        weights: pd.Series,
        volatility: pd.Series | None,
        expected_returns: pd.Series | None,
    ) -> tuple[pd.Series, str]:
        """Apply the fractional Kelly ceiling, or refuse loudly and fall back.

        ``kelly_cap`` needs an expected return over the hold window. What it used to be
        handed was ``selected`` — a cross-sectional rank in roughly [-0.5, 0.5] — which
        against 35 percent volatility implies a ceiling near 0.6 against weights near
        0.07. The ``min()`` could not bind. A risk control that cannot fire is worse than
        no risk control, because the config advertises it.

        The real quantity comes from :mod:`swingbot.model.calibrate`, and it is not always
        available: the first walk-forward fold has no earlier fold to calibrate on. When
        it is missing, this falls back to plain inverse-vol sizing and *says so* in the
        portfolio notes. Silently sizing as if a ceiling had been applied is exactly the
        failure this whole pass exists to remove.
        """
        if weights.empty or self.kelly_fraction <= 0:
            return weights, ""

        if volatility is None or volatility.empty:
            return weights, (
                "sizing=kelly requested but no volatility estimate was available; "
                "no Kelly ceiling applied"
            )

        mu = (
            expected_returns.reindex(weights.index).astype(float)
            if expected_returns is not None
            else pd.Series(float("nan"), index=weights.index)
        )
        n_missing = int(mu.isna().sum())
        if n_missing == len(mu):
            log.warning(
                "sizing=kelly but no calibrated expected return is available for any of "
                "the %d selected name(s); falling back to inverse-vol with no ceiling",
                len(mu),
            )
            return weights, (
                "sizing=kelly requested but no calibrated expected return was available; "
                "fell back to inverse-vol sizing with NO Kelly ceiling"
            )

        capped = kelly_cap(weights, mu, volatility, fraction=self.kelly_fraction)
        bound = int((capped.abs() < weights.abs() - 1e-12).sum())
        parts = []
        if bound:
            parts.append(f"{bound} position(s) trimmed by the Kelly ceiling")
        if n_missing:
            parts.append(f"{n_missing} name(s) had no calibrated mu and were left uncapped")
        return capped, "; ".join(parts)

    def _apply_no_trade_band(
        self, target: pd.Series, previous: dict[str, float]
    ) -> tuple[pd.Series, str]:
        """Keep the existing weight where the change is too small to be worth paying for."""
        if self.no_trade_band <= 0:
            return target, ""

        prev = pd.Series(previous, dtype=float)
        combined = target.reindex(target.index.union(prev.index)).fillna(0.0)
        prev = prev.reindex(combined.index).fillna(0.0)

        change = (combined - prev).abs()
        # Only hold a position that is already on; the band must never open a new one.
        hold = (change < self.no_trade_band) & (prev.abs() > 1e-9)
        if not bool(hold.any()):
            return combined[combined.abs() > 1e-9], ""

        adjusted = combined.copy()
        adjusted.loc[hold] = prev.loc[hold]
        adjusted = adjusted[adjusted.abs() > 1e-9]
        return adjusted, f"{int(hold.sum())} position(s) held inside the no-trade band"


def weights_to_frame(portfolio: TargetPortfolio) -> pd.DataFrame:
    """Target book as a tidy frame, for reports and order files."""
    if not portfolio.positions:
        return pd.DataFrame(
            columns=[TICKER, "weight", "score", "instrument", SECTOR, "side"]
        )
    return pd.DataFrame(
        [
            {
                TICKER: p.ticker,
                "weight": p.weight,
                "score": p.score,
                "instrument": p.instrument,
                SECTOR: p.sector,
                "side": "short" if p.weight < 0 else "long",
            }
            for p in portfolio.positions
        ]
    ).sort_values("weight", ascending=False).reset_index(drop=True)

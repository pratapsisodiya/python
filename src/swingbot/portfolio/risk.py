"""Hard risk limits and the drawdown kill-switch.

Limits are applied in a fixed order, and the order matters: capping per-name weight can
push the book over its sector limit, and capping sectors changes gross, so the sequence
ends by re-normalising gross and net. Each step reports what it did, so a book that hits
a constraint says so rather than quietly deforming.

The kill-switch is deliberately implemented as part of the portfolio path rather than as
a post-hoc filter on the equity curve. Applying it after the fact would let a backtest
claim protection it never paid for: in reality, de-risking into a drawdown means you are
also small during the recovery, and that cost has to appear in the returns.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..types import SECTOR, TICKER


@dataclass(slots=True)
class RiskLimits:
    max_weight: float = 0.12
    max_sector_weight: float = 0.35
    gross_leverage: float = 1.0
    max_net_exposure: float = 0.30
    min_position_weight: float = 0.001


@dataclass(slots=True)
class RiskState:
    """Kill-switch state, carried across rebalances by the backtest."""

    peak_equity: float = 1.0
    scale: float = 1.0
    cooldown_remaining: int = 0
    triggered_at: str = ""
    events: list[str] = field(default_factory=list)

    @property
    def is_flat(self) -> bool:
        return self.scale <= 1e-9


def apply_limits(
    weights: pd.Series,
    *,
    limits: RiskLimits,
    sectors: pd.Series | None = None,
    max_iterations: int = 8,
) -> tuple[pd.Series, list[str]]:
    """Enforce every hard limit, returning the adjusted book and what was binding.

    The limits interact, which is why this iterates rather than applying each once.
    Capping per-name weight lowers gross; scaling gross back up can re-breach the name
    cap; capping a sector lowers gross again. A single pass in any order leaves some
    limit violated.

    The rule that resolves it: **gross is only ever scaled down, never up past a binding
    cap.** If the caps make the target gross unreachable — three sectors cannot fill a
    100 percent book under a 35 percent sector cap — the book stays under-leveraged and
    says so. Running a smaller book than intended is safe; breaching a concentration
    limit to hit a leverage number is exactly the trade that concentration limits exist
    to prevent.
    """
    notes: list[str] = []
    if weights.empty:
        return weights, notes

    out = weights.astype(float).copy()
    sector_map = (
        sectors.reindex(out.index).fillna("Unknown") if sectors is not None and not sectors.empty
        else None
    )

    # Start at the target gross so the caps below act on realistic sizes.
    gross = out.abs().sum()
    if gross > 1e-12:
        out = out * (limits.gross_leverage / gross)

    capped_names = 0
    scaled_sectors: dict[str, float] = {}

    for _ in range(max_iterations):
        before = out.copy()

        over = out.abs() > limits.max_weight + 1e-12
        if bool(over.any()):
            capped_names = max(capped_names, int(over.sum()))
            out = np.sign(out) * out.abs().clip(upper=limits.max_weight)

        if sector_map is not None:
            gross_by_sector = out.abs().groupby(sector_map).sum()
            for sector, sector_gross in gross_by_sector.items():
                if sector_gross > limits.max_sector_weight + 1e-12:
                    members = sector_map[sector_map == sector].index
                    out.loc[members] = out.loc[members] * (
                        limits.max_sector_weight / sector_gross
                    )
                    scaled_sectors[str(sector)] = float(sector_gross)

        # Only ever scale down. Scaling up is what re-breached the caps.
        gross = out.abs().sum()
        if gross > limits.gross_leverage + 1e-12:
            out = out * (limits.gross_leverage / gross)

        if np.allclose(out.to_numpy(), before.to_numpy(), atol=1e-12):
            break

    if capped_names:
        notes.append(f"{capped_names} name(s) capped at {limits.max_weight:.1%}")
    for sector, original in scaled_sectors.items():
        notes.append(
            f"sector {sector} scaled from {original:.1%} to {limits.max_sector_weight:.1%}"
        )

    # Drop dust. A position of a few basis points pays a full round trip and moves nothing.
    dust = (out.abs() < limits.min_position_weight) & (out != 0.0)
    if bool(dust.any()):
        out.loc[dust] = 0.0

    # Net exposure, trimming the larger side so tightening a limit never adds risk.
    net = out.sum()
    if abs(net) > limits.max_net_exposure + 1e-12:
        excess = abs(net) - limits.max_net_exposure
        side = out > 0 if net > 0 else out < 0
        side_gross = out.loc[side].abs().sum()
        if side_gross > 1e-9:
            out.loc[side] = out.loc[side] * max(0.0, 1.0 - excess / side_gross)
            notes.append(f"net exposure trimmed from {net:+.1%} to {out.sum():+.1%}")

    final_gross = out.abs().sum()
    if final_gross < limits.gross_leverage - 0.01:
        notes.append(
            f"gross {final_gross:.1%} below target {limits.gross_leverage:.1%}: "
            "concentration limits bind, book intentionally under-leveraged"
        )

    return out, notes


def update_kill_switch(
    state: RiskState,
    equity: float,
    *,
    scale_at: float = 0.08,
    flat_at: float = 0.15,
    cooldown_weeks: int = 4,
    enabled: bool = True,
) -> RiskState:
    """Advance the drawdown kill-switch by one rebalance.

    Two thresholds rather than one, because a single flat-at trigger turns a normal
    drawdown into a binary event. Halving gross first is a proportionate response to a
    drawdown that may well be noise, and it costs less when it turns out to be.
    """
    if not enabled:
        state.scale = 1.0
        return state

    state.peak_equity = max(state.peak_equity, equity)
    drawdown = equity / state.peak_equity - 1.0 if state.peak_equity > 0 else 0.0

    if state.cooldown_remaining > 0:
        state.cooldown_remaining -= 1
        if state.cooldown_remaining == 0:
            state.scale = 0.5
            state.events.append("cooldown ended, re-entering at half gross")
        return state

    if drawdown <= -flat_at:
        if state.scale > 0.0:
            state.events.append(f"flat at drawdown {drawdown:.1%}")
        state.scale = 0.0
        state.cooldown_remaining = cooldown_weeks
    elif drawdown <= -scale_at:
        if state.scale > 0.5:
            state.events.append(f"gross halved at drawdown {drawdown:.1%}")
        state.scale = 0.5
    else:
        if state.scale < 1.0 and drawdown > -scale_at / 2.0:
            state.events.append(f"risk restored at drawdown {drawdown:.1%}")
            state.scale = 1.0

    return state


def check_limits(
    weights: pd.Series,
    *,
    limits: RiskLimits,
    sectors: pd.Series | None = None,
    tolerance: float = 1e-6,
) -> list[str]:
    """Return every violated limit. Empty means the book is compliant.

    Used by the property test, which generates random forecast vectors and asserts the
    constructed book never breaches a limit.
    """
    problems: list[str] = []
    if weights.empty:
        return problems

    max_abs = float(weights.abs().max())
    if max_abs > limits.max_weight + tolerance:
        problems.append(f"max weight {max_abs:.4f} exceeds {limits.max_weight:.4f}")

    gross = float(weights.abs().sum())
    if gross > limits.gross_leverage + tolerance:
        problems.append(f"gross {gross:.4f} exceeds {limits.gross_leverage:.4f}")

    net = float(weights.sum())
    if abs(net) > limits.max_net_exposure + tolerance:
        problems.append(f"net {net:+.4f} exceeds {limits.max_net_exposure:.4f}")

    if sectors is not None and not sectors.empty:
        sector_map = sectors.reindex(weights.index).fillna("Unknown")
        by_sector = weights.abs().groupby(sector_map).sum()
        for sector, value in by_sector.items():
            if value > limits.max_sector_weight + tolerance:
                problems.append(
                    f"sector {sector} gross {value:.4f} exceeds {limits.max_sector_weight:.4f}"
                )

    return problems


def sector_series(panel: pd.DataFrame) -> pd.Series:
    """Ticker to sector, for whatever slice of the panel is being sized."""
    if SECTOR not in panel.columns:
        return pd.Series(dtype=str)
    return panel.set_index(TICKER)[SECTOR]

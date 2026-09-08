"""Seeded synthetic market generator.

Two jobs, both important.

First, it lets the entire test suite and the demo run offline, which matters because the
free price sources are rate-limited, geofenced or behind bot challenges depending on
where you run from.

Second, and more usefully, it generates a market with a **known injected signal**. That
turns otherwise-unfalsifiable claims into assertions: the golden backtest must recover
the injected alpha, and the leak canaries must show a shuffled-label model finding
nothing. A harness that cannot recover a signal it planted is broken, and a harness that
finds signal in shuffled labels is leaking. Both are silent failures on real data.

The generator produces a factor structure rather than independent random walks, because
independent names would make cross-sectional ranking trivially well-behaved and hide the
correlation problems that make weekly Sharpe estimates unreliable in practice.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta

import numpy as np
import pandas as pd

from ..pit import UTC
from ..types import (
    AVAILABLE_AT,
    CLOSE,
    DIV_CASH,
    HIGH,
    IS_DELISTED,
    LOW,
    OPEN,
    SESSION,
    SPLIT_FACTOR,
    TICKER,
    VOLUME,
)


class SyntheticProvider:
    """Generates a coherent equity panel from a seed.

    Parameters
    ----------
    alpha_strength:
        How strongly last week's own signal predicts next week's return. Zero produces a
        pure noise market, which is what the shuffled-label canary should also look like.
    reversal_strength:
        Short-horizon mean reversion, the best-documented weekly equity effect. Gives the
        momentum baseline something real to find.
    """

    name = "synthetic"

    def __init__(
        self,
        *,
        seed: int = 7,
        n_factors: int = 3,
        alpha_strength: float = 0.015,
        reversal_strength: float = 0.020,
        annual_vol: float = 0.32,
        close_hour_utc: int = 21,
        sector_map: dict[str, str] | None = None,
    ) -> None:
        self.seed = seed
        self.n_factors = n_factors
        self.alpha_strength = alpha_strength
        self.reversal_strength = reversal_strength
        self.annual_vol = annual_vol
        self.close_hour_utc = close_hour_utc
        self.sector_map = sector_map or {}

    def available(self) -> bool:
        return True

    # ------------------------------------------------------------------------ public

    def daily_bars(
        self, tickers: Sequence[str], start: date, end: date
    ) -> pd.DataFrame:
        tickers = list(dict.fromkeys(tickers))
        if not tickers:
            return pd.DataFrame()

        sessions = _business_sessions(start, end)
        if len(sessions) < 2:
            return pd.DataFrame()

        n_t, n_s = len(tickers), len(sessions)
        rng = np.random.default_rng(self.seed)

        daily_vol = self.annual_vol / np.sqrt(252.0)

        # Factor structure: a market factor plus a few style factors, with per-name
        # loadings. This is what makes names correlated, which in turn is what makes
        # the effective sample size much smaller than rows-times-names.
        loadings = rng.normal(0.0, 1.0, size=(n_t, self.n_factors))
        loadings[:, 0] = np.abs(loadings[:, 0]) * 0.6 + 0.5  # market beta, positive
        factor_returns = rng.normal(0.0, daily_vol * 0.6, size=(n_s, self.n_factors))

        idio = rng.normal(0.0, daily_vol * 0.8, size=(n_s, n_t))

        # The injected signal. Each name carries a slowly-varying latent score; next
        # week's return loads on it. Recovering this is the golden backtest's job.
        latent = _ou_process(rng, n_s, n_t, half_life=25.0)

        returns = np.zeros((n_s, n_t), dtype=float)
        drift = daily_vol * 0.02

        for t in range(n_s):
            systematic = factor_returns[t] @ loadings.T
            step = systematic + idio[t] + drift

            if t >= 5:
                # Signal known as of t-1 predicts the return at t.
                step += self.alpha_strength / 5.0 * latent[t - 1]
                # Short-horizon reversal against the prior week's move.
                prior_week = returns[t - 5 : t].sum(axis=0)
                step -= self.reversal_strength / 5.0 * _zscore(prior_week)

            returns[t] = step

        # Build OHLC from the close path with a plausible intraday range.
        base_price = rng.uniform(40.0, 400.0, size=n_t)
        closes = base_price * np.exp(np.cumsum(returns, axis=0))

        prev_closes = np.vstack([base_price, closes[:-1]])
        gap = rng.normal(0.0, daily_vol * 0.35, size=(n_s, n_t))
        opens = prev_closes * np.exp(gap)

        span = np.abs(rng.normal(0.0, daily_vol * 0.9, size=(n_s, n_t))) + 1e-4
        highs = np.maximum(opens, closes) * (1.0 + span * 0.5)
        lows = np.minimum(opens, closes) * (1.0 - span * 0.5)

        # Traded notional around 20 million in the market currency, which is the right
        # order of magnitude for a liquid large or mid cap and keeps the liquidity
        # filter from screening out the entire synthetic universe.
        log_adv = rng.normal(17.0, 1.1, size=n_t)
        base_volume = np.exp(log_adv) / np.maximum(closes[0], 1.0)
        volume_noise = np.exp(rng.normal(0.0, 0.45, size=(n_s, n_t)))
        volumes = np.maximum(base_volume[None, :] * volume_noise, 100.0)

        frame = pd.DataFrame(
            {
                TICKER: np.repeat(np.array(tickers, dtype=object), n_s),
                SESSION: np.tile(np.array(sessions, dtype=object), n_t),
                OPEN: opens.T.reshape(-1),
                HIGH: highs.T.reshape(-1),
                LOW: lows.T.reshape(-1),
                CLOSE: closes.T.reshape(-1),
                VOLUME: volumes.T.reshape(-1).round(0),
            }
        )
        frame[SPLIT_FACTOR] = 1.0
        frame[DIV_CASH] = 0.0
        frame[IS_DELISTED] = False
        frame[AVAILABLE_AT] = pd.to_datetime(frame[SESSION], utc=True) + pd.Timedelta(
            hours=self.close_hour_utc
        )
        return frame.sort_values([TICKER, SESSION]).reset_index(drop=True)

    def latent_signal(
        self, tickers: Sequence[str], start: date, end: date
    ) -> pd.DataFrame:
        """The injected signal itself, for tests that need ground truth.

        Never used by the pipeline. Only by tests that assert the harness can recover a
        planted effect.
        """
        tickers = list(dict.fromkeys(tickers))
        sessions = _business_sessions(start, end)
        rng = np.random.default_rng(self.seed)
        # Advance the generator identically to daily_bars so the latent path matches.
        rng.normal(0.0, 1.0, size=(len(tickers), self.n_factors))
        rng.normal(0.0, 1.0, size=(len(sessions), self.n_factors))
        rng.normal(0.0, 1.0, size=(len(sessions), len(tickers)))
        latent = _ou_process(rng, len(sessions), len(tickers), half_life=25.0)
        return pd.DataFrame(latent, index=pd.Index(sessions, name=SESSION), columns=tickers)


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _business_sessions(start: date, end: date) -> list[date]:
    """Weekdays between two dates. Close enough to a trading calendar for synthetic use."""
    out: list[date] = []
    cursor = start
    while cursor <= end:
        if cursor.weekday() < 5:
            out.append(cursor)
        cursor += timedelta(days=1)
    return out


def _ou_process(
    rng: np.random.Generator, n_steps: int, n_series: int, *, half_life: float
) -> np.ndarray:
    """Mean-reverting latent scores, standardised cross-sectionally each step."""
    phi = 0.5 ** (1.0 / max(half_life, 1e-6))
    shocks = rng.normal(0.0, 1.0, size=(n_steps, n_series))
    out = np.zeros((n_steps, n_series), dtype=float)
    out[0] = shocks[0]
    for t in range(1, n_steps):
        out[t] = phi * out[t - 1] + np.sqrt(max(1.0 - phi**2, 1e-9)) * shocks[t]
    return np.apply_along_axis(_zscore, 1, out)


def _zscore(values: np.ndarray) -> np.ndarray:
    mean = values.mean()
    std = values.std()
    if std < 1e-12:
        return np.zeros_like(values)
    return (values - mean) / std

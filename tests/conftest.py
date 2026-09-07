"""Shared fixtures. Everything here runs offline on generated data.

The whole suite must pass with no network, because the free price sources are
rate-limited, geofenced or behind bot challenges depending on where you run from, and a
test suite that only passes on a good day is not a test suite.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from swingbot.calendars import TradingCalendar
from swingbot.config import load_config
from swingbot.data import SyntheticProvider, screen_bars
from swingbot.data.universe import StaticUniverse
from swingbot.features import (
    FeaturePipeline,
    build_labels,
    combine_weights,
    recency_weights,
    uniqueness_weights,
)

#: Small enough to keep the suite fast, large enough for cross-sectional ranking to mean
#: something. Below about 20 names a "decile" is one stock.
N_NAMES = 40
START = dt.date(2017, 1, 2)
END = dt.date(2023, 12, 29)


@pytest.fixture(scope="session")
def cfg_us():
    return load_config("us", set_values=["features.min_names_per_week=10"])


@pytest.fixture(scope="session")
def cfg_india():
    return load_config("india", set_values=["features.min_names_per_week=10"])


@pytest.fixture(scope="session")
def tickers() -> list[str]:
    return [f"T{i:02d}" for i in range(N_NAMES)]


@pytest.fixture(scope="session")
def alpha_bars(tickers):
    """A market with a known injected signal. The golden backtest must recover it."""
    provider = SyntheticProvider(
        seed=13, close_hour_utc=21, alpha_strength=0.020, reversal_strength=0.020
    )
    return screen_bars(provider.daily_bars(tickers, START, END))


@pytest.fixture(scope="session")
def null_bars(tickers):
    """A market with no signal at all. Nothing may find anything here."""
    provider = SyntheticProvider(
        seed=29, close_hour_utc=21, alpha_strength=0.0, reversal_strength=0.0
    )
    return screen_bars(provider.daily_bars(tickers, START, END))


@pytest.fixture(scope="session")
def calendar(alpha_bars, cfg_us):
    return TradingCalendar.from_bars(alpha_bars, cfg_us)


@pytest.fixture(scope="session")
def grid(calendar, cfg_us):
    return calendar.weekly_grid(cfg_us.calendar.hold_sessions)


def build_panel(bars, cfg, *, tickers=None, min_names=10):
    """Assemble a modelling panel the same way the pipeline does."""
    calendar = TradingCalendar.from_bars(bars, cfg)
    grid = calendar.weekly_grid(cfg.calendar.hold_sessions)
    sessions = [g.decision_session for g in grid]

    labels = build_labels(bars, calendar, hold_sessions=cfg.calendar.hold_sessions, grid=grid)
    if not labels.empty:
        labels["sample_weight"] = combine_weights(
            uniqueness_weights(labels), recency_weights(labels["decision_session"])
        )
        labels = labels[
            ["ticker", "decision_session", "label", "entry_session", "exit_session",
             "label_t0", "label_t1", "sample_weight"]
        ]

    names = tickers or sorted(bars["ticker"].unique())
    universe = StaticUniverse(list(names), {t: f"S{i % 5}" for i, t in enumerate(names)})
    from swingbot.data import membership_panel

    pipeline = FeaturePipeline.from_config(cfg)
    pipeline.min_names_per_week = min_names
    panel = pipeline.build(
        bars,
        decision_sessions=sessions,
        membership=membership_panel(universe, sessions, tickers=list(names)),
        labels=labels if not labels.empty else None,
    )
    return panel, calendar, grid


@pytest.fixture(scope="session")
def alpha_panel(alpha_bars, cfg_us):
    panel, calendar, grid = build_panel(alpha_bars, cfg_us)
    return panel.dropna(subset=["label"]), calendar, grid


@pytest.fixture(scope="session")
def null_panel(null_bars, cfg_us):
    panel, calendar, grid = build_panel(null_bars, cfg_us)
    return panel.dropna(subset=["label"]), calendar, grid


def make_articles(tickers, calendar, grid, *, per_week=1.0, seed=5, signal=0.0, realised=None):
    """A small news corpus. ``signal`` controls how predictive the tone is."""
    import numpy as np

    from swingbot.news.dedupe import article_hash
    from swingbot.types import RawArticle

    rng = np.random.default_rng(seed)
    positive = ["{t} profit beats estimates, revenue surges",
                "{t} raises guidance after strong quarter",
                "{t} wins large order, shares rally"]
    negative = ["{t} profit misses estimates as margins slip",
                "{t} cuts guidance and warns on demand",
                "{t} shares fall after regulatory probe"]
    neutral = ["{t} in focus ahead of results",
               "Market wrap: {t} among traded counters"]

    out = []
    for decision in grid:
        outcomes = (realised or {}).get(decision.decision_session)
        for ticker in tickers:
            if rng.random() > per_week / 5.0:
                continue
            score = rng.normal()
            if outcomes is not None and signal:
                value = outcomes.get(ticker)
                if value is not None and np.isfinite(value):
                    score = np.tanh(value * 12.0) * signal + rng.normal(0, 1 - min(signal, 0.99))
            pool = positive if score > 0.45 else negative if score < -0.45 else neutral
            title = pool[rng.integers(0, len(pool))].format(t=ticker)
            stamp = (decision.decision_ts - pd.Timedelta(hours=float(rng.uniform(6, 90)))).floor("s")
            stub = RawArticle(
                content_hash="", ticker=ticker, title=title, body=f"{title}. Detail follows.",
                url=f"https://x.invalid/{ticker}/{stamp:%Y%m%d%H%M}", source="test",
                published_at=stamp.to_pydatetime(), first_seen_at=stamp.to_pydatetime(),
            )
            out.append(
                RawArticle(
                    content_hash=article_hash(stub), ticker=ticker, title=title,
                    body=stub.body, url=stub.url, source="test",
                    published_at=stub.published_at, first_seen_at=stub.first_seen_at,
                )
            )
    return out

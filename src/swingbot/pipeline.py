"""End-to-end orchestration shared by the CLI commands.

Every command is a thin wrapper over the functions here, so the demo, the backtest and
the live weekly signal all traverse exactly the same code path. That is not tidiness for
its own sake: if the live path built features differently from the backtest path, the
backtest would be measuring a strategy nobody is running, and the difference would be
invisible until it cost money.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd

from .calendars import TradingCalendar, WeeklyDecision
from .config import Config
from .data import (
    ProviderChain,
    apply_liquidity_filter,
    load_universe,
    membership_panel,
    screen_bars,
)
from .features import (
    FeaturePipeline,
    build_labels,
    build_news_features,
    combine_weights,
    news_coverage_report,
    recency_weights,
    uniqueness_weights,
)
from .io.store import ParquetStore
from .types import DECISION_SESSION, SESSION, TICKER

log = logging.getLogger(__name__)


@dataclass(slots=True)
class MarketData:
    """Everything the modelling layer needs, assembled once."""

    cfg: Config
    bars: pd.DataFrame
    calendar: TradingCalendar
    grid: list[WeeklyDecision]
    universe: object
    panel: pd.DataFrame = field(default_factory=pd.DataFrame)
    eligibility: pd.DataFrame = field(default_factory=pd.DataFrame)
    news_features: pd.DataFrame = field(default_factory=pd.DataFrame)
    caveats: list[str] = field(default_factory=list)

    @property
    def decision_sessions(self) -> list[date]:
        return [g.decision_session for g in self.grid]


def load_bars(cfg: Config, *, refresh: bool = False) -> tuple[pd.DataFrame, object]:
    """Fetch or read cached bars for the configured universe."""
    universe = load_universe(cfg)
    tickers = universe.all_tickers()

    store = ParquetStore(cfg.market_dir)
    chain = ProviderChain.from_config(cfg, store=store)

    start = cfg.data.start or date(2015, 1, 1)
    end = cfg.data.end or date.today()

    bars = chain.daily_bars(tickers, start, end, refresh=refresh)
    if bars.empty:
        raise RuntimeError(
            f"No price data for {cfg.market_profile.display_name}. "
            "Check config/markets/<market>.yaml providers, or drop CSVs into "
            f"{cfg.market_dir / 'csv'}, or run `swingbot demo` for synthetic data."
        )
    return screen_bars(bars), universe


def build_market_data(
    cfg: Config,
    *,
    bars: pd.DataFrame | None = None,
    universe=None,
    news_features: pd.DataFrame | None = None,
    with_labels: bool = True,
    refresh: bool = False,
) -> MarketData:
    """Assemble bars, calendar, grid, features and labels into a modelling panel."""
    if bars is None or universe is None:
        bars, universe = load_bars(cfg, refresh=refresh)

    calendar = TradingCalendar.from_bars(bars, cfg)
    grid = calendar.weekly_grid(cfg.calendar.hold_sessions)
    if not grid:
        raise RuntimeError(
            f"Not enough history to form a weekly grid: {calendar.describe()}"
        )

    caveats: list[str] = []
    warning = getattr(universe, "bias_warning", lambda: None)()
    if warning:
        caveats.append(warning)

    data = MarketData(
        cfg=cfg, bars=bars, calendar=calendar, grid=grid, universe=universe, caveats=caveats
    )

    # Liquidity eligibility, computed on trailing information only.
    screened = apply_liquidity_filter(
        bars, min_price=cfg.data.min_price, min_adv_notional=cfg.data.min_adv_notional
    )
    data.eligibility = screened[[SESSION, TICKER, "liquid", "adv_notional"]]

    labels = pd.DataFrame()
    if with_labels:
        labels = build_labels(
            bars,
            calendar,
            hold_sessions=cfg.calendar.hold_sessions,
            grid=grid,
            vol_scale=cfg.labels.vol_scale,
        )
        if not labels.empty:
            labels["sample_weight"] = combine_weights(
                uniqueness_weights(labels),
                recency_weights(
                    labels[DECISION_SESSION],
                    half_life_weeks=cfg.model.recency_half_life_weeks,
                ),
            )
            labels = labels[
                [
                    TICKER, DECISION_SESSION, "label", "entry_session", "exit_session",
                    "label_t0", "label_t1", "sample_weight",
                ]
            ]

    pipeline = FeaturePipeline.from_config(cfg)
    membership = membership_panel(
        universe, data.decision_sessions, tickers=sorted(bars[TICKER].unique())
    )

    if news_features is None and cfg.features.news.enabled:
        news_features = load_news_features(cfg, calendar, grid=grid)
    data.news_features = news_features if news_features is not None else pd.DataFrame()

    panel = pipeline.build(
        bars,
        decision_sessions=data.decision_sessions,
        membership=membership,
        labels=labels if not labels.empty else None,
        news_features=data.news_features if not data.news_features.empty else None,
    )

    if not panel.empty and not data.news_features.empty:
        coverage = news_coverage_report(data.news_features, panel)
        log.info(
            "news coverage: %.1f%% of name-weeks, median %.0f article(s)",
            coverage["coverage"] * 100, coverage["median_articles"],
        )
        if coverage["coverage"] < 0.20:
            caveats.append(
                f"News covers only {coverage['coverage']:.1%} of name-weeks, so any news "
                "effect is a claim about that fraction of the book, not the whole of it."
            )

    data.panel = panel
    return data


def load_news_features(
    cfg: Config, calendar: TradingCalendar, *, grid: list[WeeklyDecision] | None = None
) -> pd.DataFrame:
    """Read stored news analyses and aggregate them to ticker-weeks."""
    store = ParquetStore(cfg.market_dir)
    analyses = store.read("news", "analyses.parquet")
    if analyses.empty:
        return pd.DataFrame()

    return build_news_features(
        analyses,
        calendar,
        grid=grid,
        hold_sessions=cfg.calendar.hold_sessions,
        lookback_days=cfg.features.news.lookback_days,
        half_life_hours=cfg.features.news.half_life_hours,
        peer_weight=cfg.features.news.peer_weight,
        speculative_discount=cfg.features.news.speculative_discount,
    )


def fetch_and_analyze_news(
    cfg: Config, *, lookback_days: int = 30, tickers: list[str] | None = None
) -> pd.DataFrame:
    """Fetch articles, analyse them, and persist the results.

    Appends rather than replaces, and deduplicates on content hash, so running this
    weekly accumulates a corpus over time. That accumulation is the only way to build
    news history from RSS, which serves only recent items.
    """
    from .news import NewsAnalysisPipeline, build_provider, records_to_frame

    universe = load_universe(cfg)
    names = tickers or universe.all_tickers()
    company_names = _company_names(cfg)

    provider = build_provider(cfg, company_names=company_names)
    if not provider.available():
        log.warning("news provider %s is not available", provider.name)
        return pd.DataFrame()

    end = datetime.now(UTC)
    start = end - timedelta(days=lookback_days)
    articles = list(provider.fetch(names, start, end))
    log.info("fetched %d article(s) from %s", len(articles), provider.name)
    if not articles:
        return pd.DataFrame()

    pipeline = NewsAnalysisPipeline.from_config(cfg)
    try:
        records = pipeline.run(articles)
        stats = pipeline.stats()
        log.info(
            "analysed %d record(s) via %s; cache holds %d entr(ies), %.2f USD spent",
            len(records), pipeline.analyzer.backend, stats["entries"], stats["total_spend_usd"],
        )
    finally:
        pipeline.close()

    frame = records_to_frame(records)
    if frame.empty:
        return frame

    store = ParquetStore(cfg.market_dir)
    store.append(frame, "news", "analyses.parquet", dedupe_on=["content_hash"])
    return frame


def _company_names(cfg: Config) -> dict[str, str]:
    """Ticker to company name, so news search finds coverage rather than nothing."""
    path = cfg.universe.file
    if path is None or not Path(path).exists():
        return {}
    raw = pd.read_csv(path)
    raw.columns = [c.strip().lower() for c in raw.columns]
    if "name" not in raw.columns or "ticker" not in raw.columns:
        return {}
    return dict(zip(raw["ticker"].astype(str), raw["name"].astype(str), strict=True))


def generate_demo_data(cfg: Config, *, years: int = 8, n_names: int = 0) -> Path:
    """Write a synthetic market to the CSV provider directory.

    Gives the pipeline something real to chew on with no network, and — because the
    generator plants a known signal — makes it possible to check that the harness can
    recover an effect it was told about.
    """
    from .data.prices_csv import write_csv_fixture
    from .data.synthetic import SyntheticProvider

    universe = load_universe(cfg)
    all_tickers = universe.all_tickers()
    # Zero means the whole configured universe. Covering only part of it would let the
    # provider chain quietly fill the rest from a separate synthetic draw, so the demo
    # would be running on two different markets stitched together.
    tickers = all_tickers if n_names <= 0 else all_tickers[:n_names]
    end = date.today()
    start = date(end.year - years, 1, 1)

    close_hour = 10 if cfg.market_profile.name == "india" else 21
    provider = SyntheticProvider(seed=cfg.run.seed, close_hour_utc=close_hour)
    bars = provider.daily_bars(tickers, start, end)

    directory = cfg.market_dir / "csv"
    write_csv_fixture(bars, directory)
    log.info(
        "wrote %d synthetic bar(s) for %d ticker(s) to %s",
        len(bars), len(tickers), directory,
    )
    return directory


def generate_demo_news(cfg: Config, *, n_names: int = 0, per_week: float = 1.5) -> Path:
    """Write a synthetic news corpus whose tone is genuinely predictive.

    The tone is generated with a deliberate, known correlation to the following week's
    return. That makes the news path testable: the ablation should detect an effect here,
    and the leak canaries should still find nothing when the labels are shuffled. A
    corpus of pure noise could not distinguish "the news block works" from "the news
    block is wired up wrong".
    """
    import numpy as np

    from .news.dedupe import article_hash
    from .news.providers import write_jsonl
    from .types import RawArticle

    universe = load_universe(cfg)
    all_tickers = universe.all_tickers()
    tickers = all_tickers if n_names <= 0 else all_tickers[:n_names]
    bars, _ = load_bars(cfg)
    calendar = TradingCalendar.from_bars(bars, cfg)
    grid = calendar.weekly_grid(cfg.calendar.hold_sessions)

    from .features.labels import forward_return_matrix

    realised = forward_return_matrix(bars, grid)
    rng = np.random.default_rng(cfg.run.seed + 1)

    headlines_up = [
        "{name} Q{q} profit beats estimates, revenue surges",
        "{name} raises guidance after strong quarter",
        "{name} wins large order from international client",
        "{name} shares rally as analysts upgrade the stock",
        "{name} announces buyback, board approves payout",
    ]
    headlines_down = [
        "{name} Q{q} profit misses estimates as margins slip",
        "{name} cuts guidance, warns on weak demand",
        "{name} shares fall after regulatory probe reported",
        "{name} downgraded by brokerage on valuation concerns",
        "{name} announces equity raise, stock drops",
    ]
    headlines_flat = [
        "{name} in focus ahead of quarterly results",
        "Market wrap: {name} among actively traded counters",
        "{name} to hold annual general meeting next month",
    ]

    names = _company_names(cfg)
    articles = []
    for decision in grid:
        outcomes = realised.get(decision.decision_session)
        if outcomes is None:
            continue
        for ticker in tickers:
            if rng.random() > per_week / 5.0:
                continue
            outcome = outcomes.get(ticker)
            if outcome is None or not np.isfinite(outcome):
                continue

            # Tone correlates with the forward return but is far from deterministic:
            # a signal-to-noise ratio a real news feed would never exceed.
            signal = np.tanh(outcome * 12.0)
            noisy = signal * 0.45 + rng.normal(0.0, 0.85)

            if noisy > 0.45:
                pool = headlines_up
            elif noisy < -0.45:
                pool = headlines_down
            else:
                pool = headlines_flat

            template = pool[rng.integers(0, len(pool))]
            title = template.format(
                name=names.get(ticker, ticker), q=int(rng.integers(1, 5))
            )
            # Published a day or two before the decision close, never after it.
            hours_before = float(rng.uniform(6.0, 96.0))
            # Floored to whole seconds: sub-second precision is meaningless for a news
            # timestamp and pandas warns when discarding it on conversion.
            stamp = (decision.decision_ts - pd.Timedelta(hours=hours_before)).floor("s")

            stub = RawArticle(
                content_hash="", ticker=ticker, title=title,
                body=f"{title}. Analysts commented on the development during the session.",
                url=f"https://example.invalid/{ticker}/{stamp:%Y%m%d%H%M}",
                source="demo", published_at=stamp.to_pydatetime(),
                first_seen_at=stamp.to_pydatetime(),
            )
            articles.append(
                RawArticle(
                    content_hash=article_hash(stub), ticker=ticker, title=title,
                    body=stub.body, url=stub.url, source="demo",
                    published_at=stub.published_at, first_seen_at=stub.first_seen_at,
                )
            )

    path = cfg.market_dir / "news" / "articles.jsonl"
    write_jsonl(articles, path)
    log.info("wrote %d synthetic article(s) to %s", len(articles), path)
    return path


def analyze_demo_news(cfg: Config) -> pd.DataFrame:
    """Analyse the demo corpus with the configured backend and persist it."""
    from .news import JSONLNewsProvider, NewsAnalysisPipeline, records_to_frame

    path = cfg.news.jsonl_path or (cfg.market_dir / "news" / "articles.jsonl")
    provider = JSONLNewsProvider(path)
    if not provider.available():
        log.warning("no news corpus at %s", path)
        return pd.DataFrame()

    articles = list(
        provider.fetch([], datetime(1990, 1, 1, tzinfo=UTC), datetime.now(UTC))
    )
    if not articles:
        return pd.DataFrame()

    pipeline = NewsAnalysisPipeline.from_config(cfg)
    pipeline.max_articles = max(pipeline.max_articles, len(articles))
    try:
        records = pipeline.run(articles, dedupe=True)
    finally:
        pipeline.close()

    frame = records_to_frame(records)
    if not frame.empty:
        ParquetStore(cfg.market_dir).write(frame, "news", "analyses.parquet")
    log.info("analysed %d demo article(s)", len(frame))
    return frame

"""Command line interface.

Every command is a thin wrapper over :mod:`swingbot.pipeline`, so the demo, the backtest
and the live weekly run all traverse the same code. ``signal`` is the only one meant for
production; the rest are research tools.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime
from typing import Annotated

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .config import Config, available_markets, load_config

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Broker-independent weekly swing trading: research, signals, reports.",
)
news_app = typer.Typer(no_args_is_help=True, help="News ingest and analysis.")
app.add_typer(news_app, name="news")

console = Console()

MarketOpt = Annotated[str, typer.Option("--market", "-m", help="us | india (aliases: in, nse)")]
ProfileOpt = Annotated[str | None, typer.Option("--profile", "-p", help="config/profiles/<name>.yaml")]
SetOpt = Annotated[list[str] | None, typer.Option("--set", help="Override, e.g. portfolio.n_long=10")]


def _setup(market: str, profile: str | None, set_values: list[str] | None) -> Config:
    cfg = load_config(market, profile=profile, set_values=list(set_values or []))
    logging.basicConfig(
        level=getattr(logging, cfg.run.log_level.upper(), logging.INFO),
        format="%(levelname)-7s %(name)-28s %(message)s",
    )
    return cfg


def _frame(title: str, frame: pd.DataFrame, *, max_rows: int = 40) -> None:
    if frame.empty:
        console.print(f"[dim]{title}: nothing to show[/dim]")
        return
    table = Table(title=title, title_style="bold", header_style="dim", box=None, pad_edge=False)
    for column in frame.columns:
        table.add_column(str(column), justify="right" if frame[column].dtype.kind in "ifc" else "left")
    for _, row in frame.head(max_rows).iterrows():
        table.add_row(*[_fmt(v) for v in row])
    console.print(table)
    if len(frame) > max_rows:
        console.print(f"[dim]... {len(frame) - max_rows} more row(s)[/dim]")


def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:,.4f}" if abs(value) < 1000 else f"{value:,.1f}"
    return str(value)


# --------------------------------------------------------------------------------------
# doctor
# --------------------------------------------------------------------------------------


@app.command()
def doctor(
    market: MarketOpt = "us",
    profile: ProfileOpt = None,
    set_values: SetOpt = None,
) -> None:
    """Check the environment, data coverage, calendar and caches."""
    cfg = _setup(market, profile, set_values)
    console.print(f"[bold]swingbot {__version__}[/bold] — {cfg.market_profile.display_name}")
    console.print(f"config hash [cyan]{cfg.config_hash()}[/cyan]  markets: {', '.join(available_markets())}")
    console.print()

    rows = []
    for name, module in [
        ("pandas", "pandas"), ("numpy", "numpy"), ("scikit-learn", "sklearn"),
        ("lightgbm", "lightgbm"), ("duckdb", "duckdb"), ("anthropic", "anthropic"),
        ("transformers", "transformers"),
    ]:
        try:
            mod = __import__(module)
            rows.append({"package": name, "status": "ok",
                         "detail": getattr(mod, "__version__", "")})
        except ImportError:
            optional = name in ("lightgbm", "duckdb", "anthropic", "transformers")
            rows.append({"package": name, "status": "optional" if optional else "MISSING",
                         "detail": "not installed"})
    _frame("packages", pd.DataFrame(rows))

    from .backtest.costs import round_trip_bps

    console.print()
    console.print(
        f"[bold]costs[/bold] round trip: long {round_trip_bps(cfg):.1f} bps, "
        f"short {round_trip_bps(cfg, is_short=True):.1f} bps "
        f"(shorts as {cfg.market_profile.short_instrument.value})"
    )
    console.print(
        f"[bold]blend[/bold] price {cfg.model.blend.normalised[0]:.0%} / "
        f"news {cfg.model.blend.normalised[1]:.0%}   "
        f"[bold]nlp backend[/bold] {cfg.nlp.backend}"
    )

    try:
        from .pipeline import load_bars

        bars, universe = load_bars(cfg)
        from .calendars import TradingCalendar

        calendar = TradingCalendar.from_bars(bars, cfg)
        grid = calendar.weekly_grid(cfg.calendar.hold_sessions)
        console.print()
        console.print(f"[bold]data[/bold] {len(bars):,} bars, {bars['ticker'].nunique()} tickers")
        console.print(f"[bold]calendar[/bold] {calendar.describe()}")
        console.print(f"[bold]grid[/bold] {len(grid)} decision weeks")
        if grid:
            latest = grid[-1]
            console.print(
                f"   latest: decide {latest.decision_session} -> "
                f"enter {latest.entry_session} -> exit {latest.exit_session}"
            )
        warning = getattr(universe, "bias_warning", lambda: None)()
        if warning:
            console.print(f"[yellow]![/yellow] {warning}")
    except Exception as exc:
        console.print(f"\n[yellow]data not ready:[/yellow] {exc}")

    cache_path = cfg.market_dir / "news_cache.sqlite"
    if cache_path.exists():
        from .io.cache import ContentCache

        with ContentCache(cache_path) as cache:
            stats = cache.stats()
        console.print(
            f"\n[bold]news cache[/bold] {stats['entries']} entr(ies), "
            f"{stats['failed']} failed, {stats['total_spend_usd']:.2f} USD spent"
        )

    trials_path = cfg.run.data_dir / "trials.sqlite"
    if trials_path.exists():
        from .validation.trials import TrialLedger

        with TrialLedger(trials_path) as ledger:
            summary = ledger.summary(cfg.market_profile.name)
        console.print(
            f"[bold]trial ledger[/bold] {summary['n_configs']} distinct config(s) tried "
            f"— this is the n in the deflated Sharpe"
        )


# --------------------------------------------------------------------------------------
# demo / fetch
# --------------------------------------------------------------------------------------


@app.command()
def demo(
    market: MarketOpt = "us",
    years: Annotated[int, typer.Option(help="Years of synthetic history")] = 8,
    names: Annotated[int, typer.Option(help="Number of tickers, 0 for the whole universe")] = 0,
    with_news: Annotated[bool, typer.Option("--with-news/--no-news")] = True,
    profile: ProfileOpt = None,
    set_values: SetOpt = None,
) -> None:
    """Generate a synthetic market (and news corpus) so everything runs offline."""
    cfg = _setup(market, profile, set_values)
    from .pipeline import analyze_demo_news, generate_demo_data, generate_demo_news

    directory = generate_demo_data(cfg, years=years, n_names=names)
    console.print(f"[green]wrote synthetic prices[/green] {directory}")

    if with_news:
        path = generate_demo_news(cfg, n_names=names)
        console.print(f"[green]wrote synthetic news[/green] {path}")
        frame = analyze_demo_news(cfg)
        console.print(f"[green]analysed[/green] {len(frame)} article(s) with {cfg.nlp.backend}")

    console.print("\nnext: [cyan]swingbot backtest --market " + market + " --ablation[/cyan]")


@app.command("fetch")
def fetch_prices(
    market: MarketOpt = "us",
    start: Annotated[str | None, typer.Option(help="YYYY-MM-DD")] = None,
    refresh: Annotated[bool, typer.Option("--refresh", help="Ignore the cache")] = False,
    profile: ProfileOpt = None,
    set_values: SetOpt = None,
) -> None:
    """Download price history for the configured universe."""
    overrides = list(set_values or [])
    if start:
        overrides.append(f"data.start={start}")
    cfg = _setup(market, profile, overrides)

    from .pipeline import load_bars

    bars, _ = load_bars(cfg, refresh=refresh)
    console.print(
        f"[green]{len(bars):,} bars[/green] for {bars['ticker'].nunique()} ticker(s), "
        f"{bars['session'].min()} to {bars['session'].max()}"
    )


@news_app.command("fetch")
def news_fetch(
    market: MarketOpt = "us",
    lookback: Annotated[int, typer.Option(help="Days of history to request")] = 30,
    profile: ProfileOpt = None,
    set_values: SetOpt = None,
) -> None:
    """Fetch articles and analyse them with the configured backend."""
    cfg = _setup(market, profile, set_values)
    from .pipeline import fetch_and_analyze_news

    console.print(
        f"provider [cyan]{cfg.news.provider}[/cyan], "
        f"analyzer [cyan]{cfg.nlp.backend}[/cyan]"
    )
    frame = fetch_and_analyze_news(cfg, lookback_days=lookback)
    if frame.empty:
        console.print("[yellow]no articles analysed[/yellow]")
        return

    console.print(f"[green]{len(frame)} analysis record(s)[/green]")
    summary = (
        frame.groupby("event_type")
        .agg(n=("sentiment", "size"), mean_sentiment=("sentiment", "mean"),
             mean_score=("signed_score", "mean"))
        .sort_values("n", ascending=False)
        .reset_index()
    )
    _frame("by event type", summary)


@news_app.command("analyze")
def news_analyze(
    market: MarketOpt = "us",
    backend: Annotated[str | None, typer.Option(help="lexicon | openai_compat | anthropic | finbert")] = None,
    profile: ProfileOpt = None,
    set_values: SetOpt = None,
) -> None:
    """Re-analyse the stored corpus, optionally with a different backend.

    Results are cached per backend and model, so switching backends does not overwrite
    earlier analyses and a backtest pinned to one of them keeps its numbers.
    """
    overrides = list(set_values or [])
    if backend:
        overrides.append(f"nlp.backend={backend}")
    cfg = _setup(market, profile, overrides)
    from .pipeline import analyze_demo_news

    frame = analyze_demo_news(cfg)
    if frame.empty:
        console.print("[yellow]no corpus to analyse[/yellow]")
        return
    console.print(f"[green]{len(frame)} record(s)[/green] via {cfg.nlp.backend}")


@news_app.command("stats")
def news_stats(
    market: MarketOpt = "us",
    profile: ProfileOpt = None,
    set_values: SetOpt = None,
) -> None:
    """Coverage and cache statistics for the stored news."""
    cfg = _setup(market, profile, set_values)
    from .io.cache import ContentCache
    from .io.store import ParquetStore

    analyses = ParquetStore(cfg.market_dir).read("news", "analyses.parquet")
    if analyses.empty:
        console.print("[dim]no analyses stored[/dim]")
    else:
        console.print(
            f"[bold]{len(analyses):,}[/bold] analysis record(s), "
            f"{analyses['ticker'].nunique()} ticker(s), "
            f"{analyses['available_at'].min()} to {analyses['available_at'].max()}"
        )
        _frame(
            "by backend",
            analyses.groupby("backend").agg(n=("sentiment", "size")).reset_index(),
        )

    cache_path = cfg.market_dir / "news_cache.sqlite"
    if cache_path.exists():
        with ContentCache(cache_path) as cache:
            stats = cache.stats()
        console.print(
            f"cache: {stats['entries']} entr(ies), {stats['failed']} failed, "
            f"{stats['total_spend_usd']:.2f} USD spent"
        )


# --------------------------------------------------------------------------------------
# backtest
# --------------------------------------------------------------------------------------


@app.command()
def backtest(
    market: MarketOpt = "us",
    ablation: Annotated[bool, typer.Option("--ablation", help="Run every null model and baseline")] = False,
    sensitivity: Annotated[bool, typer.Option("--sensitivity", help="Sweep cost assumptions")] = False,
    report: Annotated[bool, typer.Option("--report/--no-report")] = True,
    open_report: Annotated[bool, typer.Option("--open", help="Open the tearsheet")] = False,
    profile: ProfileOpt = None,
    set_values: SetOpt = None,
) -> None:
    """Walk-forward backtest with purged cross-validation."""
    cfg = _setup(market, profile, set_values)
    from .backtest import BacktestEngine, cost_sensitivity, run_ablation, walk_forward_predict
    from .features.pipeline import news_feature_columns, price_feature_columns
    from .io.runs import RunContext
    from .model import BlendedModel, build_model
    from .pipeline import build_market_data
    from .validation.deflated import deflated_sharpe
    from .validation.splits import PurgedWalkForward
    from .validation.trials import Trial, TrialLedger

    data = build_market_data(cfg)
    if data.panel.empty:
        console.print("[red]no modelling panel could be built[/red]")
        raise typer.Exit(1)

    panel = data.panel.dropna(subset=["label"])
    console.print(
        f"panel {panel.shape[0]:,} rows x {panel.shape[1]} cols, "
        f"{panel['decision_session'].nunique()} weeks"
    )

    run = RunContext.create(
        cfg.run.runs_dir, command="backtest", market=cfg.market_profile.name,
        config_hash=cfg.config_hash(), config_yaml=cfg.to_yaml(),
    )

    price_cols = price_feature_columns(panel)
    news_cols = news_feature_columns(panel)
    console.print(f"features: {len(price_cols)} price, {len(news_cols)} news")

    ablation_report = None
    if ablation:
        ablation_report = run_ablation(
            panel, data.bars, data.calendar, cfg, grid=data.grid,
            eligibility=data.eligibility, caveats=data.caveats,
            include_news=bool(news_cols),
        )
        _frame("ablation", ablation_report.table())
        console.print()
        for verdict in ablation_report.verdicts:
            style = "red" if "LEAK WARNING" in verdict else "yellow" if "does not" in verdict else "green"
            console.print(f"[{style}]•[/{style}] {verdict}")
        result = (
            ablation_report.results.get("price_news")
            or ablation_report.results.get("price_only")
        )
        if result is None:
            console.print("[red]no primary variant completed[/red]")
            raise typer.Exit(1)
    else:
        splitter = PurgedWalkForward(
            train_weeks=cfg.cv.train_weeks, test_weeks=cfg.cv.test_weeks,
            embargo_weeks=cfg.cv.embargo_weeks, expanding=cfg.cv.expanding,
            min_train_weeks=cfg.cv.min_train_weeks,
        )
        pw, nw = cfg.model.blend.normalised
        if news_cols:
            factory = lambda: BlendedModel(  # noqa: E731
                build_model(cfg.model.price_model, cfg, seed=cfg.run.seed),
                build_model(cfg.model.news_model, cfg, seed=cfg.run.seed),
                price_columns=price_cols, news_columns=news_cols,
                price_weight=pw, news_weight=nw,
            )
            name = "price_news"
        else:
            factory = lambda: build_model(cfg.model.price_model, cfg, seed=cfg.run.seed)  # noqa: E731
            name = "price_only"

        predictions = walk_forward_predict(
            panel, factory, feature_columns=price_cols + news_cols, splitter=splitter
        )
        if predictions.empty:
            console.print(
                "[red]no out-of-sample predictions[/red] — not enough history for the "
                f"configured folds (train {cfg.cv.train_weeks}w + test {cfg.cv.test_weeks}w)"
            )
            raise typer.Exit(1)
        result = BacktestEngine(cfg).run(
            predictions, data.bars, data.calendar, name=name, grid=data.grid,
            eligibility=data.eligibility, caveats=data.caveats,
        )
        _frame("result", pd.DataFrame([result.summary_row()]))

    # Trial ledger, then deflation against its real count.
    ledger_path = cfg.run.data_dir / "trials.sqlite"
    with TrialLedger(ledger_path) as ledger:
        ledger.record(
            Trial(
                market=cfg.market_profile.name, config_hash=cfg.config_hash(),
                run_id=run.run_id, git_revision=run.meta.get("git_revision", ""),
                sharpe=result.metrics.sharpe, ann_return=result.metrics.ann_return,
                max_drawdown=result.metrics.max_drawdown,
                turnover=result.metrics.turnover, n_periods=result.metrics.n_periods,
            )
        )
        n_trials = ledger.count(cfg.market_profile.name)
        variance = ledger.sharpe_variance(cfg.market_profile.name)

    deflated = deflated_sharpe(
        result.returns, n_trials=n_trials, variance_of_trials=variance
    )
    console.print(
        f"\n[bold]deflated Sharpe[/bold] {n_trials} distinct config(s) tried, "
        f"P(true Sharpe > 0) = {deflated.probability:.2f} — {deflated.verdict()}"
    )

    cost_rows = None
    if sensitivity:
        cost_rows = cost_sensitivity(
            panel, data.bars, data.calendar, cfg, grid=data.grid,
            eligibility=data.eligibility,
        )
        _frame("cost sensitivity", cost_rows)

    run.write_frame(result.to_frame(), "weekly.csv", index=True)
    run.write_json(result.metrics.to_dict(), "metrics.json")
    if ablation_report is not None:
        run.write_frame(ablation_report.table(), "ablation.csv")

    if report:
        from .report import open_in_browser, render_tearsheet

        path = render_tearsheet(
            result, cfg, output_path=run.path("tearsheet.html"),
            ablation=ablation_report, cost_rows=cost_rows, deflated=deflated,
            run_id=run.run_id, git_revision=run.meta.get("git_revision", ""),
        )
        console.print(f"[green]tearsheet[/green] {path}")
        if open_report:
            open_in_browser(path)

    run.finish(result.metrics.to_dict())
    console.print(f"[dim]run artefacts: {run.directory}[/dim]")


# --------------------------------------------------------------------------------------
# signal
# --------------------------------------------------------------------------------------


@app.command()
def signal(
    market: MarketOpt = "us",
    asof: Annotated[str | None, typer.Option(help="YYYY-MM-DD, defaults to the latest week")] = None,
    equity: Annotated[float | None, typer.Option(help="Account equity for share sizing")] = None,
    notify: Annotated[bool, typer.Option("--notify/--no-notify")] = True,
    profile: ProfileOpt = None,
    set_values: SetOpt = None,
) -> None:
    """Produce this week's target book and write order files. The production command."""
    cfg = _setup(market, profile, set_values)
    from .execution import build_adapter, build_orders, orders_to_frame
    from .features.pipeline import news_feature_columns, price_feature_columns
    from .io.runs import RunContext
    from .model import BlendedModel, build_model
    from .notify import build_notifiers, format_signal_message
    from .pipeline import build_market_data
    from .portfolio import PortfolioConstructor
    from .types import RunMeta

    data = build_market_data(cfg)
    panel = data.panel
    if panel.empty:
        console.print("[red]no panel[/red]")
        raise typer.Exit(1)

    asof_date = date.fromisoformat(asof) if asof else data.decision_sessions[-1]
    decision = data.calendar.decision_for(asof_date, cfg.calendar.hold_sessions)
    if decision is None:
        console.print(f"[red]no decision date at or before {asof_date}[/red]")
        raise typer.Exit(1)

    price_cols = price_feature_columns(panel)
    news_cols = news_feature_columns(panel)
    pw, nw = cfg.model.blend.normalised

    # Train on everything with a complete label and score the target week. The label is
    # NaN for the most recent weeks precisely because their outcome has not happened.
    labelled = panel.dropna(subset=["label"])
    target_rows = panel.loc[panel["decision_session"] == decision.decision_session]
    if target_rows.empty:
        console.print(f"[red]no panel rows for {decision.decision_session}[/red]")
        raise typer.Exit(1)

    if news_cols:
        model = BlendedModel(
            build_model(cfg.model.price_model, cfg, seed=cfg.run.seed),
            build_model(cfg.model.news_model, cfg, seed=cfg.run.seed),
            price_columns=price_cols, news_columns=news_cols,
            price_weight=pw, news_weight=nw,
        )
    else:
        model = build_model(cfg.model.price_model, cfg, seed=cfg.run.seed)

    train = labelled.loc[labelled["label_t1"] < decision.decision_session]
    if len(train) < 500:
        console.print(
            f"[yellow]only {len(train)} training rows with labels that closed before "
            f"{decision.decision_session}[/yellow]"
        )
    model.fit(
        train[price_cols + news_cols], train["label"],
        sample_weight=train.get("sample_weight"), week_index=train["decision_session"],
    )
    scores = pd.Series(
        model.predict(target_rows[price_cols + news_cols]).to_numpy(),
        index=target_rows["ticker"].to_numpy(),
    )

    # Sizing inputs, all trailing.
    from .backtest.engine import _return_history, _rolling_stats, _weekly_return_panel

    stats = _rolling_stats(data.bars).get(decision.decision_session, {})
    history = _return_history(
        _weekly_return_panel(data.bars, data.calendar),
        decision.decision_session, scores.index,
    )
    eligible = None
    if not data.eligibility.empty:
        block = data.eligibility.loc[data.eligibility["session"] == decision.decision_session]
        if not block.empty:
            eligible = block.set_index("ticker")["liquid"].reindex(scores.index).fillna(False)

    run = RunContext.create(
        cfg.run.runs_dir, command="signal", market=cfg.market_profile.name,
        config_hash=cfg.config_hash(), config_yaml=cfg.to_yaml(),
        extra={"decision_session": str(decision.decision_session)},
    )
    adapter = build_adapter(cfg, run.directory, equity=equity)
    account_equity = equity if equity is not None else adapter.account_equity()

    portfolio = PortfolioConstructor.from_config(cfg).build(
        scores,
        decision_session=decision.decision_session,
        entry_session=decision.entry_session,
        volatility=stats.get("volatility"),
        sectors=target_rows.set_index("ticker")["sector"],
        returns_history=history,
        previous_weights=adapter.current_positions() and None,
        eligible=eligible,
    )

    last_close = (
        data.bars.loc[data.bars["session"] == decision.decision_session]
        .set_index("ticker")["close"].to_dict()
    )
    adapter.set_prices(last_close)
    orders = build_orders(
        portfolio, last_close, account_equity, adapter.current_positions(),
        tag=f"{cfg.market_profile.name}-weekly",
    )
    meta = RunMeta(
        run_id=run.run_id, market=cfg.market_profile.name,
        decision_session=decision.decision_session, entry_session=decision.entry_session,
        equity=account_equity, currency=cfg.market_profile.currency,
        extra={"gross": portfolio.gross, "net": portfolio.net,
               "risk_scale": portfolio.risk_scale},
    )
    report = adapter.submit(orders, meta)

    from .portfolio.construct import weights_to_frame

    _frame(f"target book — enter at the open on {decision.entry_session}",
           weights_to_frame(portfolio))
    _frame("orders", orders_to_frame(orders, last_close))
    for message in report.messages:
        console.print(f"[green]{message}[/green]")

    subject, body = format_signal_message(
        portfolio, market=cfg.market_profile.name,
        currency=cfg.market_profile.currency, equity=account_equity,
        warnings=data.caveats,
    )
    from .report import render_signal_page

    page = render_signal_page(
        portfolio, cfg, output_path=run.path("signal.html"),
        orders=orders_to_frame(orders, last_close), warnings=data.caveats,
    )
    console.print(f"[green]signal page[/green] {page}")

    if notify:
        for notifier in build_notifiers(cfg):
            ok = notifier.send(subject, body, html=page.read_text())
            console.print(f"{'[green]sent[/green]' if ok else '[yellow]failed[/yellow]'} via {notifier.name}")

    run.finish({"n_positions": len(portfolio.positions), "n_orders": len(orders)})
    console.print(f"[dim]run artefacts: {run.directory}[/dim]")


# --------------------------------------------------------------------------------------
# run-weekly / trials
# --------------------------------------------------------------------------------------


@app.command("run-weekly")
def run_weekly(
    market: MarketOpt = "us",
    lookback: Annotated[int, typer.Option(help="Days of news to fetch")] = 14,
    profile: ProfileOpt = "live",
    set_values: SetOpt = None,
) -> None:
    """The scheduled command: refresh data, analyse news, emit the signal."""
    cfg = _setup(market, profile, set_values)
    console.print(f"[bold]weekly run[/bold] {cfg.market_profile.display_name} "
                  f"{datetime.now(UTC):%Y-%m-%d %H:%M UTC}")

    from .pipeline import fetch_and_analyze_news, load_bars

    try:
        bars, _ = load_bars(cfg, refresh=True)
        console.print(f"prices: {len(bars):,} bars to {bars['session'].max()}")
    except Exception as exc:
        console.print(f"[yellow]price refresh failed, using cache:[/yellow] {exc}")

    if cfg.features.news.enabled:
        try:
            frame = fetch_and_analyze_news(cfg, lookback_days=lookback)
            console.print(f"news: {len(frame)} new analysis record(s)")
        except Exception as exc:
            console.print(f"[yellow]news step failed, continuing without it:[/yellow] {exc}")

    signal(
        market=market, asof=None, equity=None, notify=True,
        profile=profile, set_values=set_values,
    )


@app.command()
def trials(
    market: MarketOpt = "us",
    limit: Annotated[int, typer.Option()] = 25,
    profile: ProfileOpt = None,
    set_values: SetOpt = None,
) -> None:
    """List the trial ledger that feeds the deflated Sharpe."""
    cfg = _setup(market, profile, set_values)
    from .validation.trials import TrialLedger

    path = cfg.run.data_dir / "trials.sqlite"
    if not path.exists():
        console.print("[dim]no trials recorded yet[/dim]")
        return
    with TrialLedger(path) as ledger:
        frame = ledger.list(cfg.market_profile.name, limit=limit)
        summary = ledger.summary(cfg.market_profile.name)
    if not frame.empty:
        _frame("trials", frame[["id", "variant", "config_hash", "sharpe", "max_drawdown",
                                "n_periods", "created_at"]])
    console.print(
        f"{summary['n_configs']} distinct config(s), best Sharpe "
        f"{summary.get('best_sharpe', 0):.2f}, median {summary.get('median_sharpe', 0):.2f}"
    )


@app.command()
def version() -> None:
    """Print the version."""
    console.print(f"swingbot {__version__}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()

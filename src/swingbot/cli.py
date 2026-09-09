"""Command line interface.

Every command here is a printer. The work happens in :mod:`swingbot.service`, which the
web dashboard calls too, so a book shown on a screen and a book written to a run directory
came from the same function rather than from two implementations that agree today.

What belongs in this file: argument parsing, tables, colour, and turning a
:class:`~swingbot.service.ServiceError` into a red line and a non-zero exit. What does not:
anything that decides a position or computes a number.

``signal`` is the only command meant for production; the rest are research tools.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .config import Config, load_config

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


def _fail(exc: Exception) -> typer.Exit:
    """Render a service-level refusal and stop.

    ``ServiceError`` means the request itself cannot be satisfied and the message is
    already written for a person, so it is printed as-is rather than wrapped in a
    traceback the user would have to read past.
    """
    console.print(f"[red]{exc}[/red]")
    return typer.Exit(1)


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
    from .service import doctor_report

    r = doctor_report(cfg)

    console.print(f"[bold]swingbot {r.version}[/bold] — {r.display_name}")
    console.print(
        f"config hash [cyan]{r.config_hash}[/cyan]  markets: {', '.join(r.markets)}"
    )
    console.print()
    _frame("packages", pd.DataFrame(r.packages))

    console.print()
    console.print(
        f"[bold]costs[/bold] round trip: long {r.round_trip_long_bps:.1f} bps, "
        f"short {r.round_trip_short_bps:.1f} bps (shorts as {r.short_instrument})"
    )
    console.print(
        f"[bold]blend[/bold] price {r.price_weight:.0%} / news {r.news_weight:.0%}   "
        f"[bold]nlp backend[/bold] {r.nlp_backend}"
    )

    if r.data_ready:
        console.print()
        console.print(f"[bold]data[/bold] {r.n_bars:,} bars, {r.n_tickers} tickers")
        console.print(f"[bold]calendar[/bold] {r.calendar}")
        console.print(f"[bold]grid[/bold] {r.n_weeks} decision weeks")
        if r.latest_decision:
            console.print(
                f"   latest: decide {r.latest_decision} -> enter {r.latest_entry} "
                f"-> exit {r.latest_exit}"
            )
        if r.bias_warning:
            console.print(f"[yellow]![/yellow] {r.bias_warning}")

        # Where those bars came from. "128 tickers, 272,000 bars" is a statement about
        # volume, not about truth, and a chain that ends in a generator can answer for a
        # renamed symbol without anything looking wrong.
        prov = r.provenance
        if prov:
            sources = ", ".join(f"{k} {v}" for k, v in prov["sources"].items()) or "unknown"
            style = "green" if prov["clean"] else "yellow"
            console.print(f"[bold]prices[/bold] from {sources}")
            console.print(f"   [{style}]{prov['verdict']}[/{style}]")
    else:
        console.print(f"\n[yellow]data not ready:[/yellow] {r.data_error}")

    if r.shorting:
        console.print()
        if not r.shorting.get("available"):
            console.print(f"[yellow]shorting[/yellow] {r.shorting.get('reason', '')}")
        else:
            n_ok = r.shorting.get("n_tradeable", 0)
            style = "red" if n_ok == 0 else "yellow" if n_ok < 7 else "green"
            console.print(f"[bold]shorting[/bold] [{style}]{r.shorting['verdict']}[/{style}]")
            if r.shorting.get("n_no_futures"):
                console.print(
                    f"   {r.shorting['n_no_futures']} universe name(s) have no futures "
                    "contract at all and are long-only regardless of the model"
                )
            blocked = r.shorting.get("blocked") or []
            if blocked:
                _frame(
                    "cheapest lots — the equity each one needs",
                    pd.DataFrame(blocked[:8]),
                )

    if r.news_cache:
        console.print(
            f"\n[bold]news cache[/bold] {r.news_cache['entries']} entr(ies), "
            f"{r.news_cache['failed']} failed, "
            f"{r.news_cache['total_spend_usd']:.2f} USD spent"
        )
    if r.n_trial_configs:
        console.print(
            f"[bold]trial ledger[/bold] {r.n_trial_configs} distinct config(s) tried "
            "— this is the n in the deflated Sharpe"
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
    from .service import ServiceError, run_demo

    try:
        outcome = run_demo(cfg, years=years, n_names=names, with_news=with_news)
    except ServiceError as exc:
        raise _fail(exc) from None

    console.print(f"[green]wrote synthetic prices[/green] {outcome.prices_path}")
    if outcome.news_path is not None:
        console.print(f"[green]wrote synthetic news[/green] {outcome.news_path}")
        console.print(
            f"[green]analysed[/green] {outcome.n_analysed} article(s) with {outcome.backend}"
        )
    console.print(f"\nnext: [cyan]swingbot backtest --market {market} --ablation[/cyan]")


@app.command("instruments")
def instruments(
    market: MarketOpt = "india",
    profile: ProfileOpt = None,
    set_values: SetOpt = None,
) -> None:
    """Refresh the NSE instrument snapshot: ISINs, F&O lot sizes, freeze limits.

    Only meaningful for India. A single-stock future trades in an exchange-defined lot
    that differs per underlying and is revised periodically, so the snapshot is dated and
    refreshed on purpose rather than silently picked up — a backtest has to be able to
    reproduce against the lots that were in force.
    """
    cfg = _setup(market, profile, set_values)
    if cfg.market_profile.name != "india":
        console.print(
            f"[yellow]instrument snapshots are an NSE concept; {cfg.market_profile.name} "
            "trades cash equity in single shares[/yellow]"
        )
        raise typer.Exit(1)

    from .data import load_universe, refresh_snapshot

    universe = load_universe(cfg)
    tickers = universe.all_tickers()
    try:
        path, n, with_lots = refresh_snapshot(tickers=tickers)
    except Exception as exc:
        raise _fail(exc) from None

    console.print(f"[green]wrote {n} instrument(s)[/green] {path}")
    console.print(
        f"{with_lots} carry an F&O lot size and can therefore be shorted weekly; "
        f"{n - with_lots} have no futures contract and are long-only whatever the model thinks"
    )
    console.print(
        "[dim]next: copy the lot_size column into your universe file, or re-run "
        "`swingbot doctor --market india` to see the capital needed to short[/dim]"
    )


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
    pbo: Annotated[
        bool,
        typer.Option(
            "--pbo",
            help="Probability of backtest overfitting across combinatorial purged folds. "
            "Expensive — many model fits — so it is off by default.",
        ),
    ] = False,
    report: Annotated[bool, typer.Option("--report/--no-report")] = True,
    open_report: Annotated[bool, typer.Option("--open", help="Open the tearsheet")] = False,
    profile: ProfileOpt = None,
    set_values: SetOpt = None,
) -> None:
    """Walk-forward backtest with purged cross-validation."""
    cfg = _setup(market, profile, set_values)
    from .service import ServiceError, run_backtest

    try:
        out = run_backtest(
            cfg, ablation=ablation, sensitivity=sensitivity, pbo=pbo, report=report
        )
    except ServiceError as exc:
        raise _fail(exc) from None

    console.print(
        f"panel {out.panel_rows:,} rows, {out.panel_weeks} weeks; "
        f"features: {out.n_price_features} price, {out.n_news_features} news"
    )

    if out.ablation is not None:
        _frame("ablation", out.ablation.table())
        console.print()
        for verdict in out.verdicts:
            style = (
                "red" if "LEAK WARNING" in verdict
                else "yellow" if "does not" in verdict
                else "green"
            )
            console.print(f"[{style}]•[/{style}] {verdict}")
    else:
        _frame("result", pd.DataFrame([out.result.summary_row()]))

    console.print(
        f"\n[bold]deflated Sharpe[/bold] {out.n_trials} distinct config(s) tried, "
        f"P(true Sharpe > 0) = {out.deflated.probability:.2f} — {out.deflated.verdict()}"
    )

    if out.pbo is not None:
        if out.pbo.is_valid:
            _frame("PBO paths", out.pbo.table())
            style = (
                "red" if out.pbo.pbo > 0.5 else "yellow" if out.pbo.pbo > 0.25 else "green"
            )
            console.print(
                f"[bold]PBO[/bold] {out.pbo.pbo:.2f} over {out.pbo.n_paths} path(s), "
                f"{len(out.pbo.candidates)} candidate(s) — "
                f"[{style}]{out.pbo.verdict()}[/{style}]"
            )
        else:
            console.print(f"[yellow]PBO skipped: {out.pbo.verdict()}[/yellow]")

    if out.cost_rows is not None:
        _frame("cost sensitivity", out.cost_rows)

    for caveat in out.caveats:
        console.print(f"[yellow]![/yellow] {caveat}")

    if out.tearsheet is not None:
        console.print(f"[green]tearsheet[/green] {out.tearsheet}")
        if open_report:
            from .report import open_in_browser

            open_in_browser(out.tearsheet)

    console.print(f"[dim]run artefacts: {out.directory}[/dim]")


# --------------------------------------------------------------------------------------
# signal
# --------------------------------------------------------------------------------------


@app.command()
def signal(
    market: MarketOpt = "us",
    asof: Annotated[str | None, typer.Option(help="YYYY-MM-DD, defaults to the latest week")] = None,
    equity: Annotated[float | None, typer.Option(help="Account equity for share sizing")] = None,
    notify: Annotated[bool, typer.Option("--notify/--no-notify")] = True,
    use_model: Annotated[
        str | None,
        typer.Option(
            "--use-model",
            help="Reuse the model saved by an earlier run (run id or unique prefix) "
            "instead of refitting. This is what makes a past trade explainable.",
        ),
    ] = None,
    profile: ProfileOpt = None,
    set_values: SetOpt = None,
) -> None:
    """Produce this week's target book and write order files. The production command."""
    cfg = _setup(market, profile, set_values)
    from .service import ServiceError, run_signal

    try:
        out = run_signal(cfg, asof=asof, equity=equity, notify=notify, use_model=use_model)
    except ServiceError as exc:
        raise _fail(exc) from None

    if out.pinned_model:
        trained = out.model_card.train_end if out.model_card else "unknown"
        console.print(
            f"[cyan]using pinned model[/cyan] from {out.pinned_model} (trained through {trained})"
        )
    if out.calibration:
        console.print(f"[dim]{out.calibration}[/dim]")

    _frame(f"target book — enter at the open on {out.entry_session}", out.book_frame)
    _frame("orders", out.orders_frame)

    # The portfolio's notes are where it explains itself: a limit that bound, a capacity
    # truncation, a Kelly ceiling that could not be applied. Printed after the tables,
    # where someone reading the book will already have a question they answer.
    for note in out.notes:
        console.print(f"[yellow]note:[/yellow] {note}")
    for caveat in out.caveats:
        console.print(f"[yellow]![/yellow] {caveat}")
    for message in out.execution.messages:
        console.print(f"[green]{message}[/green]")

    if out.page is not None:
        console.print(f"[green]signal page[/green] {out.page}")
    for line in out.notified:
        console.print(f"[dim]notify[/dim] {line}")
    console.print(f"[dim]run artefacts: {out.directory}[/dim]")


# --------------------------------------------------------------------------------------
# run-weekly / trials
# --------------------------------------------------------------------------------------


@app.command("run-weekly")
def run_weekly(
    market: MarketOpt = "us",
    lookback: Annotated[int, typer.Option(help="Days of news to fetch")] = 14,
    equity: Annotated[float | None, typer.Option(help="Account equity for share sizing")] = None,
    profile: ProfileOpt = "live",
    set_values: SetOpt = None,
) -> None:
    """The scheduled command: refresh data, analyse news, emit the signal."""
    cfg = _setup(market, profile, set_values)
    from .service import ServiceError

    console.print(f"[bold]weekly run[/bold] {cfg.market_profile.display_name} "
                  f"{datetime.now(UTC):%Y-%m-%d %H:%M UTC}")

    from .service import run_weekly as _run_weekly

    try:
        out, messages = _run_weekly(cfg, lookback=lookback, equity=equity, notify=True)
    except ServiceError as exc:
        raise _fail(exc) from None

    # A refresh step that failed is reported, not swallowed: a book built on a stale cache
    # is still worth having, but only if you know that is what you are looking at.
    for message in messages:
        style = "yellow" if "failed" in message else "dim"
        console.print(f"[{style}]{message}[/{style}]")

    _frame(f"target book — enter at the open on {out.entry_session}", out.book_frame)
    _frame("orders", out.orders_frame)
    for note in out.notes:
        console.print(f"[yellow]note:[/yellow] {note}")
    for line in out.notified:
        console.print(f"[dim]notify[/dim] {line}")
    console.print(f"[dim]run artefacts: {out.directory}[/dim]")


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
def runs(
    market: MarketOpt = "us",
    limit: Annotated[int, typer.Option()] = 20,
    profile: ProfileOpt = None,
    set_values: SetOpt = None,
) -> None:
    """List recent runs. The run id is what `signal --use-model` takes."""
    cfg = _setup(market, profile, set_values)
    from .io.runs import list_runs

    entries = list_runs(cfg.run.runs_dir, limit=limit)
    if not entries:
        console.print(f"[dim]no runs under {cfg.run.runs_dir}[/dim]")
        return

    frame = pd.DataFrame(
        [
            {
                "run_id": e.get("run_id", ""),
                "command": e.get("command", ""),
                "market": e.get("market", ""),
                "git": e.get("git_revision", ""),
                "model": "yes" if (Path(cfg.run.runs_dir) / e.get("run_id", "") / "model" / "model.pkl").exists() else "",
                "started_at": e.get("started_at", "")[:19],
            }
            for e in entries
        ]
    )
    _frame(f"runs under {cfg.run.runs_dir}", frame, max_rows=limit)


@app.command()
def serve(
    market: MarketOpt = "us",
    host: Annotated[str, typer.Option(help="Bind address. Keep it on loopback.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port to listen on")] = 8765,
    open_browser: Annotated[bool, typer.Option("--open", help="Open the dashboard")] = False,
    profile: ProfileOpt = None,
    set_values: SetOpt = None,
) -> None:
    """Run the local web dashboard. Needs pip install 'swingbot\\[web]'."""
    cfg = _setup(market, profile, set_values)
    from .web import is_loopback
    from .web import serve as serve_app

    # No authentication, and the page has a button that runs code on this machine. On
    # loopback that is fine — it is your own machine. On anything else it is a decision
    # someone should make on purpose, so it is stated rather than discovered.
    if not is_loopback(host):
        console.print(
            f"[red]![/red] binding to [bold]{host}[/bold], not loopback. The dashboard has "
            "no login and can start jobs on this machine, so anyone who can reach this "
            "address can run them. Use 127.0.0.1 unless you have a reason not to."
        )

    url = f"http://{host}:{port}"
    console.print(f"[green]swingbot dashboard[/green] {url}   ({cfg.market_profile.display_name})")
    console.print("[dim]read-only with respect to your broker — no credentials, no orders sent[/dim]")
    console.print("[dim]ctrl-c to stop[/dim]")

    if open_browser:
        from .report import open_in_browser

        open_in_browser(url)

    try:
        # The profile and overrides go through too, so a job started from the page
        # resolves its config through the same layers this command did.
        serve_app(cfg, host=host, port=port, profile=profile, set_values=list(set_values or []))
    except RuntimeError as exc:
        # The one expected RuntimeError here is the missing-optional-dependency message,
        # which is already written for a person.
        raise _fail(exc) from None
    except KeyboardInterrupt:
        console.print("\n[dim]stopped[/dim]")


@app.command()
def version() -> None:
    """Print the version."""
    console.print(f"swingbot {__version__}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()

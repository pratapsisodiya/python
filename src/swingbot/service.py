"""The orchestration layer: one code path, several front ends.

Everything a user can *ask the system to do* — generate a demo market, run a backtest,
produce this week's book, check the environment — lives here as a plain function that
takes a :class:`~swingbot.config.Config` and returns a result object.

This module exists because the logic used to live inside the Typer command bodies. That
was fine while a terminal was the only front end. It stopped being fine the moment a web
dashboard needed to do the same things: a second caller would have meant a second copy of
the trading logic, free to drift from the first, and the drift would show up as a book on
the screen that did not match the book in the run directory.

So the split is:

* **service** decides and computes, and returns objects. It raises
  :class:`ServiceError` for anything a user did wrong, and knows nothing about terminals,
  HTTP status codes, or how a table should look.
* **cli** prints those objects. **web** serialises them. Neither one computes.

Every function here writes a run directory, because that is what makes a result
reproducible after the fact, and the run id is the handle both front ends use to refer to
what happened.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from .config import Config
from .types import Order, RunMeta, TargetPortfolio

log = logging.getLogger(__name__)

__all__ = [
    "BacktestOutcome",
    "DemoOutcome",
    "DoctorReport",
    "ServiceError",
    "SignalOutcome",
    "doctor_report",
    "run_backtest",
    "run_demo",
    "run_signal",
    "run_weekly",
]


class ServiceError(RuntimeError):
    """A request that cannot be satisfied, with a message meant for a human.

    Distinct from the exceptions the computation layers raise. This one says "what you
    asked for is not possible, here is why" — not enough history for the configured folds,
    no such run id, a pinned model whose schema no longer matches. The CLI turns it into a
    red line and a non-zero exit; the web layer turns it into a 400 with the same text.
    Neither has to guess which failures are the user's business.
    """


# --------------------------------------------------------------------------------------
# Result objects
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class DemoOutcome:
    prices_path: Path
    news_path: Path | None = None
    n_analysed: int = 0
    backend: str = ""


@dataclass(slots=True)
class BacktestOutcome:
    """Everything a backtest produced, plus where it was written."""

    run_id: str
    directory: Path
    result: Any                     # BacktestResult
    variant: str
    n_trials: int
    deflated: Any                   # DeflatedSharpeResult
    ablation: Any = None            # AblationReport | None
    cost_rows: pd.DataFrame | None = None
    pbo: Any = None                 # PBOResult | None
    tearsheet: Path | None = None
    panel_rows: int = 0
    panel_weeks: int = 0
    n_price_features: int = 0
    n_news_features: int = 0
    caveats: list[str] = field(default_factory=list)

    @property
    def verdicts(self) -> list[str]:
        return list(self.ablation.verdicts) if self.ablation is not None else []


@dataclass(slots=True)
class SignalOutcome:
    """This week's book, the orders that get there, and the audit trail."""

    run_id: str
    directory: Path
    portfolio: TargetPortfolio
    orders: list[Order]
    orders_frame: pd.DataFrame
    book_frame: pd.DataFrame
    prices: dict[str, float]
    equity: float
    currency: str
    decision_session: date
    entry_session: date
    execution: Any                  # ExecutionReport
    page: Path | None = None
    model_card: Any = None          # ModelCard | None
    pinned_model: str = ""
    calibration: str = ""
    subject: str = ""
    body: str = ""
    notified: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)

    @property
    def notes(self) -> list[str]:
        return list(self.portfolio.notes)


@dataclass(slots=True)
class DoctorReport:
    market: str
    display_name: str
    config_hash: str
    version: str
    markets: list[str]
    packages: list[dict[str, str]]
    round_trip_long_bps: float
    round_trip_short_bps: float
    short_instrument: str
    price_weight: float
    news_weight: float
    nlp_backend: str
    data_ready: bool = False
    data_error: str = ""
    n_bars: int = 0
    n_tickers: int = 0
    calendar: str = ""
    n_weeks: int = 0
    latest_decision: str = ""
    latest_entry: str = ""
    latest_exit: str = ""
    bias_warning: str = ""
    news_cache: dict[str, Any] = field(default_factory=dict)
    n_trial_configs: int = 0
    n_runs: int = 0
    #: India only: what shorting via single-stock futures actually requires.
    shorting: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------------------
# demo
# --------------------------------------------------------------------------------------


def run_demo(
    cfg: Config, *, years: int = 8, n_names: int = 0, with_news: bool = True
) -> DemoOutcome:
    """Generate a synthetic market, and optionally a news corpus, then analyse it."""
    from .pipeline import analyze_demo_news, generate_demo_data, generate_demo_news

    prices = generate_demo_data(cfg, years=years, n_names=n_names)
    outcome = DemoOutcome(prices_path=prices, backend=cfg.nlp.backend)

    if with_news:
        outcome.news_path = generate_demo_news(cfg, n_names=n_names)
        outcome.n_analysed = len(analyze_demo_news(cfg))
    return outcome


# --------------------------------------------------------------------------------------
# backtest
# --------------------------------------------------------------------------------------


def run_backtest(
    cfg: Config,
    *,
    ablation: bool = False,
    sensitivity: bool = False,
    pbo: bool = False,
    report: bool = True,
) -> BacktestOutcome:
    """Walk-forward backtest with purged cross-validation, and its honesty checks."""
    from .backtest import (
        BacktestEngine,
        cost_sensitivity,
        run_ablation,
        run_pbo,
        walk_forward_predict,
    )
    from .features.pipeline import news_feature_columns, price_feature_columns
    from .io.runs import RunContext
    from .model import BlendedModel, build_model
    from .pipeline import build_market_data
    from .validation.deflated import deflated_sharpe
    from .validation.splits import PurgedWalkForward
    from .validation.trials import Trial, TrialLedger

    data = build_market_data(cfg)
    if data.panel.empty:
        raise ServiceError(
            "No modelling panel could be built. Check that price data exists for this "
            "market, or run `swingbot demo` for a synthetic one."
        )

    panel = data.panel.dropna(subset=["label"])
    price_cols = price_feature_columns(panel)
    news_cols = news_feature_columns(panel)

    run = RunContext.create(
        cfg.run.runs_dir,
        command="backtest",
        market=cfg.market_profile.name,
        config_hash=cfg.config_hash(),
        config_yaml=cfg.to_yaml(),
    )

    ablation_report = None
    if ablation:
        ablation_report = run_ablation(
            panel, data.bars, data.calendar, cfg, grid=data.grid,
            eligibility=data.eligibility, caveats=data.caveats,
            include_news=bool(news_cols),
        )
        result = (
            ablation_report.results.get("price_news")
            or ablation_report.results.get("price_only")
        )
        if result is None:
            raise ServiceError(
                "No primary variant completed. The ablation ran but neither price_news "
                "nor price_only produced out-of-sample predictions."
            )
        variant = result.name
    else:
        splitter = PurgedWalkForward(
            train_weeks=cfg.cv.train_weeks, test_weeks=cfg.cv.test_weeks,
            embargo_weeks=cfg.cv.embargo_weeks, expanding=cfg.cv.expanding,
            min_train_weeks=cfg.cv.min_train_weeks,
        )
        pw, nw = cfg.model.blend.normalised
        if news_cols:
            def factory():
                return BlendedModel(
                    build_model(cfg.model.price_model, cfg, seed=cfg.run.seed),
                    build_model(cfg.model.news_model, cfg, seed=cfg.run.seed),
                    price_columns=price_cols, news_columns=news_cols,
                    price_weight=pw, news_weight=nw,
                )
            variant = "price_news"
        else:
            def factory():
                return build_model(cfg.model.price_model, cfg, seed=cfg.run.seed)
            variant = "price_only"

        predictions = walk_forward_predict(
            panel, factory, feature_columns=price_cols + news_cols, splitter=splitter
        )
        if predictions.empty:
            raise ServiceError(
                "No out-of-sample predictions — not enough history for the configured "
                f"folds (train {cfg.cv.train_weeks}w + test {cfg.cv.test_weeks}w against "
                f"{panel['decision_session'].nunique()} weeks of data). Shorten cv.train_weeks "
                "or generate more history."
            )
        result = BacktestEngine(cfg).run(
            predictions, data.bars, data.calendar, name=variant, grid=data.grid,
            eligibility=data.eligibility, caveats=data.caveats,
        )

    # The ledger is written before deflation, because the trial being deflated is itself
    # one of the trials. Recording it afterwards would deflate against a count one short.
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

    deflated = deflated_sharpe(result.returns, n_trials=n_trials, variance_of_trials=variance)

    pbo_result = None
    if pbo:
        pbo_result = run_pbo(panel, cfg, feature_columns=price_cols + news_cols)

    cost_rows = None
    if sensitivity:
        cost_rows = cost_sensitivity(
            panel, data.bars, data.calendar, cfg, grid=data.grid,
            eligibility=data.eligibility,
        )

    run.write_frame(result.to_frame(), "weekly.csv", index=True)
    run.write_json(result.metrics.to_dict(), "metrics.json")
    if ablation_report is not None:
        run.write_frame(ablation_report.table(), "ablation.csv")
        run.write_json({"verdicts": list(ablation_report.verdicts)}, "verdicts.json")
    if cost_rows is not None and not cost_rows.empty:
        run.write_frame(cost_rows, "cost_sensitivity.csv")
    run.write_json(deflated.to_dict() | {"verdict": deflated.verdict()}, "deflated.json")
    if pbo_result is not None and pbo_result.is_valid:
        run.write_json(pbo_result.to_dict(), "pbo.json")
        run.write_frame(pbo_result.table(), "pbo_paths.csv")

    tearsheet = None
    if report:
        from .report import render_tearsheet

        tearsheet = render_tearsheet(
            result, cfg, output_path=run.path("tearsheet.html"),
            ablation=ablation_report, cost_rows=cost_rows, deflated=deflated,
            pbo=pbo_result,
            run_id=run.run_id, git_revision=run.meta.get("git_revision", ""),
        )

    run.finish(result.metrics.to_dict())

    return BacktestOutcome(
        run_id=run.run_id,
        directory=run.directory,
        result=result,
        variant=variant,
        n_trials=n_trials,
        deflated=deflated,
        ablation=ablation_report,
        cost_rows=cost_rows,
        pbo=pbo_result,
        tearsheet=tearsheet,
        panel_rows=len(panel),
        panel_weeks=int(panel["decision_session"].nunique()),
        n_price_features=len(price_cols),
        n_news_features=len(news_cols),
        caveats=list(data.caveats),
    )


# --------------------------------------------------------------------------------------
# signal — the production command
# --------------------------------------------------------------------------------------


def run_signal(
    cfg: Config,
    *,
    asof: str | date | None = None,
    equity: float | None = None,
    notify: bool = True,
    use_model: str | None = None,
) -> SignalOutcome:
    """Produce this week's target book, diff it against holdings, write the order files."""
    from .backtest.engine import _return_history, _rolling_stats, _weekly_return_panel
    from .execution import build_adapter, build_orders, execution_sequence, orders_to_frame
    from .features.pipeline import news_feature_columns, price_feature_columns
    from .io.runs import RunContext
    from .model import BlendedModel, build_model
    from .model.registry import save_model
    from .notify import build_notifiers, format_signal_message
    from .pipeline import build_market_data
    from .portfolio import PortfolioConstructor
    from .portfolio.construct import weights_to_frame
    from .report import render_signal_page

    data = build_market_data(cfg)
    panel = data.panel
    if panel.empty:
        raise ServiceError("No modelling panel could be built for this market.")

    asof_date = _as_date(asof) or data.decision_sessions[-1]
    decision = data.calendar.decision_for(asof_date, cfg.calendar.hold_sessions)
    if decision is None:
        raise ServiceError(
            f"No decision date at or before {asof_date}. The weekly grid runs "
            f"{data.decision_sessions[0]} to {data.decision_sessions[-1]}."
        )

    price_cols = price_feature_columns(panel)
    news_cols = news_feature_columns(panel)
    feature_columns = price_cols + news_cols
    pw, nw = cfg.model.blend.normalised

    # Train on everything with a complete label and score the target week. The label is
    # NaN for the most recent weeks precisely because their outcome has not happened.
    labelled = panel.dropna(subset=["label"])
    target_rows = panel.loc[panel["decision_session"] == decision.decision_session]
    if target_rows.empty:
        raise ServiceError(f"No panel rows for {decision.decision_session}.")

    def make_model():
        if news_cols:
            return BlendedModel(
                build_model(cfg.model.price_model, cfg, seed=cfg.run.seed),
                build_model(cfg.model.news_model, cfg, seed=cfg.run.seed),
                price_columns=price_cols, news_columns=news_cols,
                price_weight=pw, news_weight=nw,
            )
        return build_model(cfg.model.price_model, cfg, seed=cfg.run.seed)

    train = labelled.loc[labelled["label_t1"] < decision.decision_session]

    # Load a pinned model, or fit one. Both paths converge on `model`; only the fitting
    # path produces something worth saving.
    #
    # This command used to refit from scratch every week and save nothing, which meant the
    # model that produced last week's orders no longer existed. A trade could be described
    # but never explained, `SchemaMismatchError` could never fire, and "reproduce that
    # decision" had no answer. Saving the fit and being able to pin it is what turns a run
    # directory into an audit trail.
    model_card = None
    if use_model:
        model, model_card = load_pinned_model(cfg, use_model, feature_columns, decision)
    else:
        model = make_model()
        if len(train) < 500:
            log.warning(
                "only %d training rows with labels that closed before %s",
                len(train), decision.decision_session,
            )
        model.fit(
            train[feature_columns], train["label"],
            sample_weight=train.get("sample_weight"), week_index=train["decision_session"],
        )

    scores = pd.Series(
        model.predict(target_rows[feature_columns]).to_numpy(),
        index=target_rows["ticker"].to_numpy(),
    )

    expected_returns, calibration = (None, "")
    if cfg.portfolio.sizing == "kelly":
        expected_returns, calibration = calibrated_expected_returns(
            cfg, train, scores, make_model, feature_columns
        )

    # Sizing inputs, all trailing.
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

    last_close = (
        data.bars.loc[data.bars["session"] == decision.decision_session]
        .set_index("ticker")["close"].to_dict()
    )

    run = RunContext.create(
        cfg.run.runs_dir, command="signal", market=cfg.market_profile.name,
        config_hash=cfg.config_hash(), config_yaml=cfg.to_yaml(),
        extra={"decision_session": str(decision.decision_session)},
    )
    if use_model:
        run.meta["pinned_model"] = use_model
        if model_card is not None:
            run.write_json(model_card.to_dict(), "model/model_card.json")
    else:
        save_model(
            model,
            run.path("model"),
            market=cfg.market_profile.name,
            config_hash=cfg.config_hash(),
            train_start=train["decision_session"].min() if not train.empty else None,
            train_end=train["decision_session"].max() if not train.empty else None,
            n_train_rows=len(train),
            price_columns=price_cols,
            news_columns=news_cols,
            extra={"decision_session": str(decision.decision_session)},
        )

    adapter = build_adapter(cfg, run.directory, equity=equity)
    account_equity = equity if equity is not None else adapter.account_equity()
    held_shares = adapter.current_positions()

    portfolio = PortfolioConstructor.from_config(cfg).build(
        scores,
        decision_session=decision.decision_session,
        entry_session=decision.entry_session,
        volatility=stats.get("volatility"),
        sectors=target_rows.set_index("ticker")["sector"],
        returns_history=history,
        previous_weights=_held_weights(held_shares, last_close, account_equity),
        risk_scale=1.0,
        eligible=eligible,
        adv_notional=stats.get("adv_notional"),
        equity=account_equity,
        expected_returns=expected_returns,
    )

    adapter.set_prices(last_close)

    # Tradeable increments and the minimum order worth sending.
    #
    # Both of these were dead: `build_orders` accepted `lot_size` and `min_order_value`,
    # implemented them correctly, and no caller ever passed either — while
    # `portfolio.min_position_notional` sat in the config read by nothing. The visible
    # consequence was an India order file asking to sell 173 of a single-stock future, a
    # quantity no exchange will accept, because futures trade in lots.
    profile = cfg.market_profile
    lot_sizes = getattr(data.universe, "lot_sizes", lambda: {})()
    sizing_notes: list[str] = []
    orders = build_orders(
        portfolio, last_close, account_equity, held_shares,
        lot_size=profile.equity_lot_size,
        lot_sizes=lot_sizes,
        derivative_lot_size=profile.derivative_lot_size,
        min_order_value=cfg.portfolio.min_position_notional,
        tag=f"{profile.name}-weekly",
        notes=sizing_notes,
    )
    # A position the account cannot express belongs on the book's own note list, next to
    # the risk limits and the capacity truncations — it is the same kind of fact.
    portfolio.notes.extend(sizing_notes)
    # Can this account short at all?
    #
    # An indivisible lot and a per-name weight cap together set a hard floor on account
    # size, and on NSE that floor is high: the cheapest F&O lot in the Nifty 200 is around
    # 4.3 lakh of notional, so a 12 percent cap needs roughly 36 lakh of equity before a
    # single short fits. Below that the model still ranks names and still asks for shorts,
    # and every one of them is unplaceable. Saying so here is the difference between a
    # surprising order file and a rejected order.
    adequacy = _capital_adequacy_note(cfg, portfolio, last_close, account_equity)
    if adequacy:
        portfolio.notes.append(adequacy)

    if any("lot-unknown" in order.tag for order in orders):
        portfolio.notes.append(
            f"{sum('lot-unknown' in o.tag for o in orders)} order(s) are for an "
            "instrument that trades in exchange-defined lots and no lot size is "
            "declared, so the quantity is NOT rounded to a placeable size. Confirm the "
            "lot with your broker and round down, or fill in the lot_size column in "
            f"{cfg.universe.file}."
        )
    meta = RunMeta(
        run_id=run.run_id, market=cfg.market_profile.name,
        decision_session=decision.decision_session, entry_session=decision.entry_session,
        equity=account_equity, currency=cfg.market_profile.currency,
        extra={"gross": portfolio.gross, "net": portfolio.net,
               "risk_scale": portfolio.risk_scale},
    )
    execution = adapter.submit(orders, meta)
    orders_frame = orders_to_frame(orders, last_close)
    orders_frame = _attach_expected_slippage(
        orders_frame, cfg, adv_notional=stats.get("adv_notional"),
        volatility=stats.get("volatility"), high_low_range=stats.get("hl_range"),
    )

    # The target book, written alongside the orders.
    #
    # `targets.json` is the execution adapter's file and holds the orders — the trades to
    # place. It does not hold the *book*, nor the portfolio's notes, and those are exactly
    # what someone needs when reading a past run: which names were held at what weight,
    # and why the book came out the shape it did. A capacity truncation or a Kelly
    # fallback used to exist only as a line printed to a terminal, which meant last week's
    # explanation was gone by Monday.
    #
    # Written by the service rather than by the adapter, because the adapter's contract is
    # deliberately about orders and nothing else — that is the seam a broker would attach
    # to, and widening it to carry research metadata would be the wrong trade.
    # Rewrite orders.csv with the expectation column the adapter's own frame lacks.
    # The adapter's contract is orders and nothing else — widening it to carry research
    # metadata would be the wrong trade — so the service adds the column afterwards.
    orders_path = Path(execution.artifacts.get("orders_csv", "")) if execution.artifacts else None
    if orders_path and orders_path.exists():
        orders_frame.to_csv(orders_path, index=False)

    book = weights_to_frame(portfolio)
    run.write_json(
        {
            "run_id": run.run_id,
            "market": cfg.market_profile.name,
            "decision_session": str(decision.decision_session),
            "entry_session": str(decision.entry_session),
            "currency": cfg.market_profile.currency,
            "equity": account_equity,
            "gross": portfolio.gross,
            "net": portfolio.net,
            "n_long": portfolio.n_long,
            "n_short": portfolio.n_short,
            "risk_scale": portfolio.risk_scale,
            "notes": list(portfolio.notes),
            "caveats": list(data.caveats),
            "calibration": calibration,
            "pinned_model": use_model or "",
            "positions": book.to_dict("records"),
            # The order to work the tickets in, decided here rather than in a front end.
            # The web layer is not allowed to import the execution package — that is what
            # keeps the dashboard unable to place a trade — so anything derived from the
            # order set has to be computed at signal time and written down.
            "sequence": execution_sequence(orders, last_close),
        },
        "book.json",
    )

    subject, body = format_signal_message(
        portfolio, market=cfg.market_profile.name,
        currency=cfg.market_profile.currency, equity=account_equity,
        warnings=data.caveats,
    )
    page = render_signal_page(
        portfolio, cfg, output_path=run.path("signal.html"),
        orders=orders_frame, warnings=data.caveats,
    )

    notified: list[str] = []
    if notify:
        html = page.read_text()
        for notifier in build_notifiers(cfg):
            ok = notifier.send(subject, body, html=html)
            notified.append(f"{notifier.name}: {'sent' if ok else 'failed'}")

    run.finish({"n_positions": len(portfolio.positions), "n_orders": len(orders)})

    return SignalOutcome(
        run_id=run.run_id,
        directory=run.directory,
        portfolio=portfolio,
        orders=orders,
        orders_frame=orders_frame,
        book_frame=book,
        prices=last_close,
        equity=account_equity,
        currency=cfg.market_profile.currency,
        decision_session=decision.decision_session,
        entry_session=decision.entry_session,
        execution=execution,
        page=page,
        model_card=model_card,
        pinned_model=use_model or "",
        calibration=calibration,
        subject=subject,
        body=body,
        notified=notified,
        caveats=list(data.caveats),
    )


def run_weekly(
    cfg: Config, *, lookback: int = 14, equity: float | None = None, notify: bool = True
) -> tuple[SignalOutcome, list[str]]:
    """The scheduled command: refresh prices, analyse new news, emit the signal.

    Returns the signal plus a log of what the refresh steps did. Both refresh steps are
    allowed to fail without taking the run down: a stale price cache still produces a book
    worth looking at, whereas no book at all on a Friday evening is useless. What is not
    allowed is failing *silently*, so each failure comes back as a line the caller shows.
    """
    from .pipeline import fetch_and_analyze_news, load_bars

    messages: list[str] = []

    try:
        bars, _ = load_bars(cfg, refresh=True)
        messages.append(f"prices: {len(bars):,} bars to {bars['session'].max()}")
    except Exception as exc:
        messages.append(f"price refresh failed, using cache: {exc}")

    if cfg.features.news.enabled:
        try:
            frame = fetch_and_analyze_news(cfg, lookback_days=lookback)
            messages.append(f"news: {len(frame)} new analysis record(s)")
        except Exception as exc:
            messages.append(f"news step failed, continuing without it: {exc}")

    return run_signal(cfg, equity=equity, notify=notify), messages


# --------------------------------------------------------------------------------------
# Pieces of `signal` that both the model-pinning tests and the web layer need by name
# --------------------------------------------------------------------------------------


def calibrated_expected_returns(
    cfg: Config,
    train: pd.DataFrame,
    scores: pd.Series,
    model_factory,
    feature_columns: list[str],
) -> tuple[pd.Series | None, str]:
    """Map this week's scores to expected returns, or return None and say why.

    Everything here happens on rows whose labels had already closed before the decision
    date, so the map is built entirely from outcomes that had happened. It returns None
    rather than a passthrough when it cannot fit: the constructor treats None as "no Kelly
    ceiling available" and records that in the portfolio notes, which is the honest
    outcome. A passthrough of raw scores would look like a calibrated return and reinstate
    the exact bug this replaced.

    In a backtest ``walk_forward_predict`` supplies the map per fold. Here there are no
    folds, so the folds are built: a purged walk-forward over the rows whose labels have
    already closed, then one isotonic fit on all of it. It costs a second pass over
    history and only runs when the config actually asks for Kelly — but without it
    ``sizing: kelly`` would work in the backtest and quietly do nothing in production,
    which is the worse half of the bug this replaced.
    """
    from .backtest import walk_forward_predict
    from .model.calibrate import ScoreCalibrator
    from .validation.splits import PurgedWalkForward

    if train.empty:
        return None, "sizing=kelly: no closed labels to calibrate on"

    splitter = PurgedWalkForward(
        train_weeks=cfg.cv.train_weeks,
        test_weeks=cfg.cv.test_weeks,
        embargo_weeks=cfg.cv.embargo_weeks,
        expanding=cfg.cv.expanding,
        min_train_weeks=cfg.cv.min_train_weeks,
    )
    oos = walk_forward_predict(
        train, model_factory, feature_columns=feature_columns, splitter=splitter,
        calibrate=False,
    )
    if oos.empty:
        return None, (
            "sizing=kelly: not enough history for a single walk-forward fold, so no "
            "expected return could be calibrated"
        )

    calibrator = ScoreCalibrator()
    calibrator.fitted_through = oos["decision_session"].max()
    calibrator.fit(oos["prediction"], oos["label"])
    if not calibrator.is_fitted:
        return None, f"sizing=kelly: {calibrator.describe()}"

    return calibrator.transform(scores), calibrator.describe()


def load_pinned_model(cfg: Config, run_id: str, feature_columns: list[str], decision):
    """Load a saved model, refusing a schema mismatch and refusing a stale fit.

    Two refusals, and they guard different failures.

    A **schema mismatch** is silent corruption: if the feature set changed since the model
    was fitted, the columns misalign and the estimator keeps producing confident numbers
    that mean nothing. ``load_model`` raises rather than aligning by position.

    A **stale fit** is a judgement call, which is why it is a config knob. Pinning exists
    so a past decision can be reproduced exactly and a past trade explained. Using a fit
    from a year ago to trade *this* week is a different act, and it should have to be asked
    for explicitly rather than happening because a run id was convenient.
    """
    from .io.runs import find_run
    from .model.registry import SchemaMismatchError, load_model

    directory = find_run(cfg.run.runs_dir, run_id)
    if directory is None:
        raise ServiceError(
            f"No run matching {run_id!r} under {cfg.run.runs_dir}. "
            "`swingbot runs` lists what is available."
        )

    try:
        model, card = load_model(directory / "model", expected_features=feature_columns)
    except FileNotFoundError:
        raise ServiceError(
            f"Run {directory.name} saved no model — only runs created after model "
            "pinning was added carry one."
        ) from None
    except SchemaMismatchError as exc:
        raise ServiceError(str(exc)) from None

    if card and card.train_end:
        trained_through = _as_date(card.train_end)
        if trained_through is not None:
            age_weeks = (decision.decision_session - trained_through).days / 7.0
            if age_weeks > cfg.model.max_model_age_weeks:
                raise ServiceError(
                    f"Pinned model is {age_weeks:.0f} weeks stale: trained through "
                    f"{trained_through}, decision is {decision.decision_session}, limit is "
                    f"{cfg.model.max_model_age_weeks} weeks. Refit, or raise "
                    "model.max_model_age_weeks if you meant to reproduce an old decision."
                )

    if card and card.config_hash != cfg.config_hash():
        log.warning(
            "the pinned model was fitted under config hash %s, current is %s — the model "
            "reproduces, but sizing and cost settings may since have changed",
            card.config_hash[:8], cfg.config_hash()[:8],
        )
    return model, card


# --------------------------------------------------------------------------------------
# doctor
# --------------------------------------------------------------------------------------

#: Import checks for `doctor`. Anything not in this set is a hard requirement.
_OPTIONAL_PACKAGES = frozenset({"lightgbm", "duckdb", "anthropic", "transformers", "fastapi"})

_CHECKED_PACKAGES = (
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("scikit-learn", "sklearn"),
    ("lightgbm", "lightgbm"),
    ("duckdb", "duckdb"),
    ("anthropic", "anthropic"),
    ("transformers", "transformers"),
    ("fastapi", "fastapi"),
)


def doctor_report(cfg: Config) -> DoctorReport:
    """Environment, data coverage, calendar and caches. Never raises."""
    from . import __version__
    from .backtest.costs import round_trip_bps
    from .config import available_markets
    from .io.runs import list_runs

    packages = []
    for name, module in _CHECKED_PACKAGES:
        try:
            mod = __import__(module)
            packages.append(
                {"package": name, "status": "ok", "detail": getattr(mod, "__version__", "")}
            )
        except ImportError:
            packages.append({
                "package": name,
                "status": "optional" if name in _OPTIONAL_PACKAGES else "MISSING",
                "detail": "not installed",
            })

    price_weight, news_weight = cfg.model.blend.normalised
    report = DoctorReport(
        market=cfg.market_profile.name,
        display_name=cfg.market_profile.display_name,
        config_hash=cfg.config_hash(),
        version=__version__,
        markets=list(available_markets()),
        packages=packages,
        round_trip_long_bps=round_trip_bps(cfg),
        round_trip_short_bps=round_trip_bps(cfg, is_short=True),
        short_instrument=cfg.market_profile.short_instrument.value,
        price_weight=price_weight,
        news_weight=news_weight,
        nlp_backend=cfg.nlp.backend,
        n_runs=len(list_runs(cfg.run.runs_dir, limit=1000)),
    )

    # Data coverage is the one part that can legitimately fail — on a fresh clone there is
    # no data yet, and `doctor` exists precisely to say so calmly.
    try:
        from .calendars import TradingCalendar
        from .pipeline import load_bars

        bars, universe = load_bars(cfg)
        calendar = TradingCalendar.from_bars(bars, cfg)
        grid = calendar.weekly_grid(cfg.calendar.hold_sessions)

        report.data_ready = True
        report.n_bars = len(bars)
        report.n_tickers = int(bars["ticker"].nunique())
        report.calendar = calendar.describe()
        report.n_weeks = len(grid)
        if grid:
            report.latest_decision = str(grid[-1].decision_session)
            report.latest_entry = str(grid[-1].entry_session)
            report.latest_exit = str(grid[-1].exit_session)
        report.bias_warning = getattr(universe, "bias_warning", lambda: None)() or ""
    except Exception as exc:
        report.data_error = str(exc)

    # What shorting costs in capital, not in basis points.
    #
    # This belongs in `doctor` because it is a fact about the account, not about a
    # particular week: on NSE a weekly short must be a single-stock future, futures trade
    # in indivisible exchange-set lots, and one lot of the cheapest Nifty 200 name is
    # several lakh. Under a per-name weight cap that sets a floor on equity below which
    # the short sleeve cannot exist. Better to learn it from `doctor` than from a rejected
    # order on a Monday morning.
    if cfg.market_profile.short_instrument.value == "futures" and report.data_ready:
        report.shorting = _shorting_requirements(cfg)

    cache_path = cfg.market_dir / "news_cache.sqlite"
    if cache_path.exists():
        from .io.cache import ContentCache

        with ContentCache(cache_path) as cache:
            report.news_cache = cache.stats()

    trials_path = cfg.run.data_dir / "trials.sqlite"
    if trials_path.exists():
        from .validation.trials import TrialLedger

        with TrialLedger(trials_path) as ledger:
            report.n_trial_configs = int(ledger.summary(cfg.market_profile.name)["n_configs"])

    return report


# --------------------------------------------------------------------------------------
# Small shared helpers
# --------------------------------------------------------------------------------------


def _shorting_requirements(cfg: Config) -> dict[str, Any]:
    """Minimum equity at which the futures short sleeve becomes reachable."""
    from .data import NSEInstruments, capital_adequacy
    from .pipeline import load_bars

    instruments = NSEInstruments.load_if_present()
    if instruments is None:
        return {
            "available": False,
            "reason": (
                "no NSE instrument snapshot; run `swingbot instruments --market india` "
                "to find out what shorting requires"
            ),
        }

    try:
        bars, _ = load_bars(cfg)
    except Exception as exc:
        return {"available": False, "reason": str(exc)}

    latest = bars[bars["session"] == bars["session"].max()]
    prices = dict(zip(latest["ticker"], latest["close"], strict=True))

    report = capital_adequacy(
        instruments, prices,
        equity=cfg.backtest.initial_equity,
        max_weight=cfg.portfolio.max_weight,
    )
    payload = report.to_dict()
    payload["available"] = True
    payload["n_no_futures"] = len(instruments) - len(instruments.lot_sizes())
    return payload


def _capital_adequacy_note(
    cfg: Config, portfolio, prices: dict[str, float], equity: float
) -> str:
    """One sentence on whether the short sleeve is reachable at this account size.

    Only applies where shorts are expressed as a derivative — on US cash equity a short is
    just a share count and no lot exists to be indivisible.
    """
    from .data import NSEInstruments, capital_adequacy

    if cfg.market_profile.short_instrument.value != "futures":
        return ""
    shorts = [p.ticker for p in portfolio.positions if p.weight < 0]
    if not shorts or equity <= 0:
        return ""

    instruments = NSEInstruments.load_if_present()
    if instruments is None:
        return ""

    report = capital_adequacy(
        instruments, prices,
        equity=equity, max_weight=cfg.portfolio.max_weight, tickers=shorts,
    )
    return report.verdict() if report.blocked else ""


def _attach_expected_slippage(
    orders: pd.DataFrame,
    cfg: Config,
    *,
    adv_notional: pd.Series | None = None,
    volatility: pd.Series | None = None,
    high_low_range: pd.Series | None = None,
) -> pd.DataFrame:
    """Record what the cost model expects each fill to give up, in basis points.

    Stored on the order rather than recomputed later, and for one reason: it has to be
    the expectation made *before* the fill. Recomputing it afterwards from whatever data
    is on hand would let today's knowledge into yesterday's forecast, which is the same
    mistake the whole point-in-time layer exists to prevent — just moved from the signal
    to the post-trade report.

    Only the spread and impact components go in. Commission, STT and stamp duty are
    charged separately and never appear in the price on the ticket, so including them
    would flatter the model when the comparison is made against a fill price.
    """
    from .backtest.costs import CostModel

    if orders.empty:
        return orders

    model = CostModel.from_config(cfg)
    shorts_are_derivatives = cfg.market_profile.short_instrument.value != "cash_equity"

    def expected(row) -> float | None:
        notional = float(row.get("est_value") or 0.0)
        if notional <= 0:
            return None
        ticker = row["ticker"]
        breakdown = model.trade_cost(
            notional=notional,
            is_buy=row["side"] == "buy",
            is_short_leg=(
                row["instrument"] != "equity"
                if shorts_are_derivatives
                else row["instrument"] == "equity_short"
            ),
            volatility_annual=_lookup(volatility, ticker, 0.30),
            adv_notional=_lookup(adv_notional, ticker, None),
            high_low_range=_lookup(high_low_range, ticker, None),
        )
        return round((breakdown.spread + breakdown.impact) / notional * 10_000.0, 4)

    out = orders.copy()
    out["expected_slip_bps"] = [expected(row) for _, row in out.iterrows()]
    return out


def _lookup(series: pd.Series | None, key: str, default):
    if series is None or key not in series.index:
        return default
    value = series.get(key)
    return default if value is None or pd.isna(value) else float(value)


def _as_date(value: str | date | None) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None


def _held_weights(
    shares: dict[str, float], prices: dict[str, float], equity: float
) -> dict[str, float] | None:
    """Current holdings as portfolio weights, for the no-trade band.

    The band needs last week's *weights*; the adapter tracks *share counts*, because that
    is what a broker fills. This converts one to the other at the decision close.

    Worth spelling out, because the absence of this function was a bug. `signal` used to
    pass ``adapter.current_positions() and None`` — an expression that is always ``None`` —
    so the no-trade band never engaged in production. The band is the single largest
    turnover saving in the system, typically removing about a third of it at weekly
    frequency, and every backtest counted that saving while no live run ever received it.
    Reported Sharpe was therefore systematically better than what the orders would have
    achieved, in the one direction that matters.
    """
    if not shares or equity <= 0:
        return None
    weights = {
        ticker: (count * prices[ticker]) / equity
        for ticker, count in shares.items()
        if prices.get(ticker)
    }
    return weights or None

"""Routes.

Every handler does the same three things: parse the request, call
:mod:`swingbot.service`, hand the result to :mod:`swingbot.web.serialize`. No handler
computes a position, a weight, or a cost.

The long commands go through :class:`~swingbot.web.jobs.JobRunner` and are polled, because
a backtest with the ablation matrix takes minutes and an HTTP request should not be held
open for it.

**Nothing here submits an order.** The dashboard shows you what the pipeline decided and
records which orders *you* placed; the trade itself happens at your broker, by your hand.
``tests/test_architecture.py`` asserts this at the AST level, so it cannot quietly stop
being true.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from ..config import Config, available_markets, load_config
from ..io.runs import find_run, list_runs
from ..io.store import read_json, write_json
from ..service import ServiceError, doctor_report, run_backtest, run_demo, run_signal, run_weekly
from .jobs import JobRunner
from .serialize import backtest_payload, orders_payload, run_detail, run_summary, signal_payload

log = logging.getLogger(__name__)

HERE = Path(__file__).parent

#: The extension talks to the dashboard from an extension origin, which is opaque and
#: unstable across installs, so it is matched by pattern. Loopback is allowed so the
#: dashboard page itself works on either hostname. Nothing else is permitted: this app has
#: no authentication, and a permissive CORS policy would let any website in a logged-in
#: browser start jobs on the user's machine.
CORS_ORIGIN_REGEX = r"^(chrome-extension|moz-extension|safari-web-extension)://[a-z0-9-]+$"
CORS_ORIGINS = [
    f"http://{host}:{port}"
    for host in ("127.0.0.1", "localhost")
    for port in (8765, 8000, 3000)
]


# --------------------------------------------------------------------------------------
# Request bodies
# --------------------------------------------------------------------------------------


class JobRequest(BaseModel):
    """Start one of the pipeline commands."""

    kind: str = Field(description="demo | backtest | signal | run-weekly")
    market: str | None = None
    #: backtest
    ablation: bool = False
    sensitivity: bool = False
    pbo: bool = False
    #: demo
    years: int = 8
    names: int = 0
    with_news: bool = True
    #: signal / run-weekly
    asof: str | None = None
    equity: float | None = None
    notify: bool = False
    use_model: str | None = None
    lookback: int = 14
    #: Config overrides, exactly as `--set` takes them.
    set_values: list[str] = Field(default_factory=list)


class PlacedRequest(BaseModel):
    """Which orders the user has placed at their broker."""

    placed: list[str] = Field(default_factory=list)


class FillRequest(BaseModel):
    """What one order actually filled at.

    ``price`` is the price on the ticket, not including commission or taxes — those are
    charged separately and never appear in it, which is exactly why the slippage
    comparison is against the model's spread-and-impact component alone.
    """

    client_order_id: str
    placed: bool = True
    price: float | None = None
    quantity: float | None = None


class FillsRequest(BaseModel):
    fills: list[FillRequest] = Field(default_factory=list)


# --------------------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------------------


def build_app(
    cfg: Config,
    *,
    runs_dir: Path | str | None = None,
    profile: str | None = None,
    set_values: list[str] | None = None,
) -> FastAPI:
    """Assemble the dashboard around one starting config.

    ``profile`` and ``set_values`` are the arguments the server was *started* with, kept so
    a per-request config can be layered on top of them rather than built from the shipped
    defaults. Without them, `swingbot serve --profile live --set run.data_dir=/data` would
    show one configuration and run jobs under another.
    """
    base_runs = Path(runs_dir) if runs_dir else Path(cfg.run.runs_dir)
    startup_profile = profile
    startup_sets = list(set_values or [])
    runner = JobRunner()
    templates = Jinja2Templates(directory=str(HERE / "templates"))

    app = FastAPI(
        title="swingbot dashboard",
        description=(
            "Local research dashboard. Read-only with respect to your broker: it never "
            "holds a credential and never places a trade."
        ),
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_origin_regex=CORS_ORIGIN_REGEX,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")

    app.state.cfg = cfg
    app.state.runs_dir = base_runs
    app.state.runner = runner

    def config_for(market: str | None, set_values: list[str] | None = None) -> Config:
        """Resolve a config for a request, layered on top of how the server was started.

        Rebuilt per request rather than mutated, so two browser tabs on different markets
        cannot interfere and a `--set` override in one job does not leak into the next.

        The startup arguments are re-applied first and the request's overrides go last.
        Skipping that step is not a subtle bug: an earlier version rebuilt the config from
        the shipped defaults whenever a market was named, which silently discarded the
        server's `--profile` and every `--set` it was launched with — including
        `run.data_dir`, so jobs read a different data directory than the dashboard was
        showing.
        """
        requested = list(set_values or [])
        if market is None and not requested:
            return app.state.cfg
        try:
            return load_config(
                market or app.state.cfg.market_profile.name,
                profile=startup_profile,
                set_values=startup_sets + requested,
            )
        except (KeyError, ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None

    def resolve_run(run_id: str) -> Path:
        directory = find_run(app.state.runs_dir, run_id)
        if directory is None:
            raise HTTPException(status_code=404, detail=f"no run matching {run_id!r}")
        return directory

    # ------------------------------------------------------------------------------ page

    @app.get("/", response_class=HTMLResponse)
    def dashboard(request: Request) -> Any:
        return templates.TemplateResponse(
            request,
            "dashboard.html.j2",
            {
                "market": app.state.cfg.market_profile.name,
                "display_name": app.state.cfg.market_profile.display_name,
                "markets": list(available_markets()),
                "runs_dir": str(app.state.runs_dir),
                "config_hash": app.state.cfg.config_hash(),
            },
        )

    # ------------------------------------------------------------------------ environment

    @app.get("/api/health")
    def health(market: str | None = None) -> dict[str, Any]:
        report = doctor_report(config_for(market))
        return {
            "ok": True,
            "market": report.market,
            "display_name": report.display_name,
            "config_hash": report.config_hash,
            "version": report.version,
            "markets": report.markets,
            "packages": report.packages,
            "costs": {
                "round_trip_long_bps": report.round_trip_long_bps,
                "round_trip_short_bps": report.round_trip_short_bps,
                "short_instrument": report.short_instrument,
            },
            "blend": {"price": report.price_weight, "news": report.news_weight},
            "nlp_backend": report.nlp_backend,
            "data": {
                "ready": report.data_ready,
                "error": report.data_error,
                "n_bars": report.n_bars,
                "n_tickers": report.n_tickers,
                "calendar": report.calendar,
                "n_weeks": report.n_weeks,
                "latest_decision": report.latest_decision,
                "latest_entry": report.latest_entry,
                "latest_exit": report.latest_exit,
                "bias_warning": report.bias_warning,
            },
            "news_cache": report.news_cache,
            "n_trial_configs": report.n_trial_configs,
            "n_runs": report.n_runs,
        }

    @app.get("/api/markets")
    def markets() -> dict[str, Any]:
        return {"markets": list(available_markets()), "current": app.state.cfg.market_profile.name}

    # -------------------------------------------------------------------------------- runs

    @app.get("/api/runs")
    def runs(
        market: str | None = None,
        command: str | None = None,
        limit: int = Query(default=40, ge=1, le=200),
    ) -> dict[str, Any]:
        entries = list_runs(app.state.runs_dir, limit=500)
        rows = []
        for meta in entries:
            run_id = meta.get("run_id", "")
            if not run_id:
                continue
            if market and meta.get("market") != market:
                continue
            if command and meta.get("command") != command:
                continue
            rows.append(run_summary(Path(app.state.runs_dir) / run_id, meta))
            if len(rows) >= limit:
                break
        return {"runs": rows, "runs_dir": str(app.state.runs_dir)}

    @app.get("/api/runs/{run_id}")
    def run(run_id: str) -> dict[str, Any]:
        return run_detail(resolve_run(run_id))

    @app.get("/api/runs/{run_id}/report", response_class=HTMLResponse)
    def run_report(run_id: str) -> Any:
        """Serve a run's own tearsheet or signal page.

        Served straight off disk rather than re-rendered, so what the browser shows is
        byte-for-byte the artefact in the run directory — the thing that would be attached
        to an email or checked into a record.
        """
        directory = resolve_run(run_id)
        for name in ("tearsheet.html", "signal.html"):
            candidate = directory / name
            if candidate.exists():
                return FileResponse(candidate, media_type="text/html")
        raise HTTPException(status_code=404, detail=f"run {run_id} has no HTML report")

    # ------------------------------------------------------------------------------ orders

    @app.get("/api/orders/latest")
    def latest_orders(market: str | None = None) -> dict[str, Any]:
        """This week's orders. The browser extension's only endpoint.

        Returns an empty payload rather than a 404 when no signal has been run: "you have
        not produced a signal yet" is a normal state on a fresh install, not an error, and
        the extension renders a prompt for it.
        """
        wanted = market or app.state.cfg.market_profile.name
        directory = _latest_signal_dir(app.state.runs_dir, wanted)
        if directory is None:
            return orders_payload(run_id="", market=wanted, orders=[])

        detail = run_detail(directory)
        # book.json is preferred and targets.json is the fallback: a run made before
        # book.json existed still renders, just without the portfolio's notes.
        book = detail.get("book") or {}
        targets = detail.get("targets") or {}
        meta = read_json(directory / "run.json", {}) or {}

        def field(name: str, default: Any = "") -> Any:
            value = book.get(name, targets.get(name, default))
            return default if value is None else value

        return orders_payload(
            run_id=detail["run_id"],
            market=wanted,
            orders=detail["orders"],
            decision_session=field("decision_session"),
            entry_session=field("entry_session"),
            currency=field("currency"),
            equity=book.get("equity", targets.get("equity")),
            generated_at=meta.get("started_at", ""),
            notes=book.get("notes", []),
            caveats=book.get("caveats", []),
            placed=detail["placed"],
            sequence=book.get("sequence", []),
            execution=detail.get("execution", {}),
        )

    @app.get("/api/orders/{run_id}/placed")
    def get_placed(run_id: str) -> dict[str, Any]:
        directory = resolve_run(run_id)
        record = read_execution_record(directory)
        return {"run_id": run_id, "placed": placed_ids(record), "orders": record}

    @app.post("/api/orders/{run_id}/placed")
    def set_placed(run_id: str, body: PlacedRequest) -> dict[str, Any]:
        """Record which orders the user placed.

        Bookkeeping about the past, not an instruction about the future: a note that you
        keyed a trade in yourself. Stored in the run directory so it survives a server
        restart and so next week has a record of what actually happened.

        Ticks are replaced wholesale but any fill *price* already recorded is kept —
        unticking a row is a correction to the checklist, not a statement that the price
        you wrote down was wrong.
        """
        directory = resolve_run(run_id)
        wanted = {str(item) for item in body.placed}
        record = read_execution_record(directory)
        for order_id in set(record) | wanted:
            entry = record.setdefault(order_id, {})
            entry["placed"] = order_id in wanted
        write_execution_record(directory, run_id, record)
        return {"run_id": run_id, "placed": placed_ids(record), "orders": record}

    @app.post("/api/orders/{run_id}/fills")
    def set_fills(run_id: str, body: FillsRequest) -> dict[str, Any]:
        """Record what orders actually filled at, and return the slippage that implies.

        This is the loop the rest of the system was missing. Every backtest charges a
        modelled cost and sweeps it to show the strategy does not depend on the exact
        figure — honest, but still an assumption. A fill price makes it measurable.
        """
        directory = resolve_run(run_id)
        record = read_execution_record(directory)
        for fill in body.fills:
            entry = record.setdefault(fill.client_order_id, {})
            entry["placed"] = fill.placed
            if fill.price is not None:
                entry["price"] = float(fill.price)
                entry["at"] = _now()
            if fill.quantity is not None:
                entry["quantity"] = float(fill.quantity)
        write_execution_record(directory, run_id, record)
        return {"run_id": run_id, "placed": placed_ids(record), "orders": record}

    @app.get("/api/orders/{run_id}/slippage")
    def slippage(run_id: str) -> dict[str, Any]:
        """What this week's fills cost, against what the cost model predicted."""
        from ..tca import analyse_fills

        directory = resolve_run(run_id)
        detail = run_detail(directory)
        report = analyse_fills(detail["orders"], read_execution_record(directory))
        report.run_id = detail["run_id"]
        return report.to_dict()

    # -------------------------------------------------------------------------------- jobs

    @app.post("/api/jobs")
    def start_job(body: JobRequest) -> dict[str, Any]:
        job_cfg = config_for(body.market, body.set_values)
        market = job_cfg.market_profile.name
        kind = body.kind.strip().lower()

        if kind == "demo":
            def work() -> tuple[str, dict[str, Any]]:
                out = run_demo(
                    job_cfg, years=body.years, n_names=body.names, with_news=body.with_news
                )
                return "", {
                    "prices": str(out.prices_path),
                    "news": str(out.news_path) if out.news_path else "",
                    "n_analysed": out.n_analysed,
                }
            options = {"years": body.years, "names": body.names, "with_news": body.with_news}

        elif kind == "backtest":
            def work() -> tuple[str, dict[str, Any]]:
                out = run_backtest(
                    job_cfg,
                    ablation=body.ablation,
                    sensitivity=body.sensitivity,
                    pbo=body.pbo,
                )
                return out.run_id, backtest_payload(out)
            options = {
                "ablation": body.ablation, "sensitivity": body.sensitivity, "pbo": body.pbo
            }

        elif kind == "signal":
            def work() -> tuple[str, dict[str, Any]]:
                out = run_signal(
                    job_cfg,
                    asof=body.asof,
                    equity=body.equity,
                    notify=body.notify,
                    use_model=body.use_model,
                )
                return out.run_id, signal_payload(out, market=market)
            options = {"asof": body.asof, "equity": body.equity, "use_model": body.use_model}

        elif kind in ("run-weekly", "run_weekly", "weekly"):
            def work() -> tuple[str, dict[str, Any]]:
                out, messages = run_weekly(
                    job_cfg, lookback=body.lookback, equity=body.equity, notify=body.notify
                )
                payload = signal_payload(out, market=market)
                payload["refresh"] = messages
                return out.run_id, payload
            kind = "run-weekly"
            options = {"lookback": body.lookback, "equity": body.equity}

        else:
            raise HTTPException(
                status_code=400,
                detail=f"unknown job kind {body.kind!r}. Known: demo, backtest, signal, run-weekly.",
            )

        job = runner.submit(kind, market, work, options=options)
        return job.to_dict()

    @app.get("/api/jobs")
    def jobs(limit: int = Query(default=20, ge=1, le=100)) -> dict[str, Any]:
        return {
            "jobs": [job.to_dict(with_log=False) for job in runner.recent(limit)],
            "busy": runner.busy,
        }

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str, log_from: int = Query(default=0, ge=0)) -> dict[str, Any]:
        found = runner.get(job_id)
        if found is None:
            raise HTTPException(status_code=404, detail=f"no job {job_id!r}")
        return found.to_dict(log_from=log_from) | {"log_from": log_from}

    # -------------------------------------------------------------------------- error shape

    @app.exception_handler(ServiceError)
    def service_error(_request: Request, exc: ServiceError) -> JSONResponse:
        """A refusal the user can act on becomes a 400 with the original wording.

        ``ServiceError`` messages are written for a person — "not enough history for the
        configured folds", "that pinned model is 40 weeks stale" — so they are passed
        through unchanged rather than replaced by a generic message that would send the
        user to the server log.
        """
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.on_event("shutdown")
    def shutdown() -> None:
        runner.close()

    return app


def _now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat(timespec="seconds")


def read_execution_record(directory: Path) -> dict[str, dict[str, Any]]:
    """What was placed and at what price, keyed by client order id.

    ``placed.json`` is read as a fallback so runs written before fill capture existed
    still show their tick list. Same reason the book falls back to ``targets.json``: a
    run directory is a durable record, and a new field must not make an old one
    unreadable.
    """
    record = read_json(directory / "execution.json", {}) or {}
    orders = record.get("orders")
    if isinstance(orders, dict):
        return {str(k): dict(v) for k, v in orders.items() if isinstance(v, dict)}

    legacy = read_json(directory / "placed.json", {}) or {}
    return {str(order_id): {"placed": True} for order_id in legacy.get("placed", [])}


def write_execution_record(directory: Path, run_id: str, record: dict[str, dict]) -> None:
    write_json(directory / "execution.json", {"run_id": run_id, "orders": record})


def placed_ids(record: dict[str, dict]) -> list[str]:
    return sorted(k for k, v in record.items() if v.get("placed"))


def _latest_signal_dir(runs_dir: Path | str, market: str) -> Path | None:
    """The newest ``signal`` run for a market.

    Run ids are timestamp-prefixed, so ``list_runs`` already returns newest first and no
    sorting by mtime is needed — which also means a run directory copied from another
    machine sorts by when it was *made*, not when it was copied.
    """
    for meta in list_runs(runs_dir, limit=500):
        if meta.get("market") != market:
            continue
        if meta.get("command") not in ("signal", "run-weekly"):
            continue
        run_id = meta.get("run_id")
        if run_id:
            candidate = Path(runs_dir) / run_id
            if candidate.is_dir():
                return candidate
    return None

"""The dashboard's HTTP surface.

Two things are worth testing here, and they are not the same thing.

The **shape** of what the API returns, because the browser extension is a separate program
that parses it. A field rename in `serialize.py` is a change to a published interface, and
these tests are what turn that from a silent breakage into a failing build.

The **absence** of an order-submission route. ``tests/test_architecture.py`` asserts that
at the AST level; this asserts it on the assembled surface, which catches a route added
through a router or a mount that an import scan would not see. Two cheap checks on the
property that lets this app run without a broker credential.

Everything is skipped when the ``web`` extra is not installed, because the whole point of
that extra is that a CLI-only install works without it.
"""

from __future__ import annotations

import datetime as dt
import json
import time

import pytest

pytest.importorskip("fastapi", reason="the dashboard needs the [web] extra")

from fastapi.testclient import TestClient  # noqa: E402

from swingbot.config import load_config  # noqa: E402
from swingbot.web import create_app  # noqa: E402

DECISION = dt.date(2026, 9, 4)
ENTRY = dt.date(2026, 9, 8)
RUN_ID = f"{DECISION:%Y%m%d}T163000-us-signal-abcdef12"


# --------------------------------------------------------------------------------------
# A run directory on disk, written the way `service.run_signal` writes one
# --------------------------------------------------------------------------------------


ORDERS_CSV = """ticker,side,quantity,order_type,instrument,limit_price,est_price,est_value,client_order_id,tag,expected_slip_bps
AAA,buy,12,market,equity,,100.0,1200.0,20260908-AAA,us-weekly,6.0
BBB,sell,8,market,futures,,50.0,400.0,20260908-BBB,us-weekly lot-unknown,7.5
"""


@pytest.fixture()
def runs_dir(tmp_path):
    """One signal run and one backtest run, so every panel has something to show."""
    signal = tmp_path / RUN_ID
    signal.mkdir(parents=True)
    (signal / "orders.csv").write_text(ORDERS_CSV)
    (signal / "run.json").write_text(json.dumps({
        "run_id": RUN_ID, "command": "signal", "market": "us",
        "config_hash": "abcdef1234", "git_revision": "deadbee",
        "started_at": "2026-09-04T16:30:00+00:00",
        "finished_at": "2026-09-04T16:31:00+00:00",
        "summary": {"n_positions": 12, "n_orders": 2},
    }))
    (signal / "book.json").write_text(json.dumps({
        "run_id": RUN_ID, "market": "us",
        "decision_session": str(DECISION), "entry_session": str(ENTRY),
        "currency": "USD", "equity": 50000.0,
        "gross": 0.94, "net": 0.11, "n_long": 7, "n_short": 5, "risk_scale": 1.0,
        "notes": ["3 order(s) truncated by the 5.0% participation cap (AAA by 1.20%)"],
        "caveats": ["universe file has no delisted names: results are survivorship-biased"],
        "positions": [
            {"ticker": "AAA", "weight": 0.08, "score": 0.31, "instrument": "equity",
             "sector": "Tech", "side": "long"},
            {"ticker": "BBB", "weight": -0.06, "score": -0.24, "instrument": "futures",
             "sector": "Energy", "side": "short"},
        ],
    }))
    (signal / "targets.json").write_text(json.dumps({
        "run_id": RUN_ID, "market": "us", "decision_session": str(DECISION),
        "entry_session": str(ENTRY), "currency": "USD", "equity": 50000.0, "orders": [],
    }))
    (signal / "signal.html").write_text("<h1>signal</h1>")
    (signal / "model").mkdir()
    (signal / "model" / "model.pkl").write_bytes(b"not-a-real-pickle")

    backtest = tmp_path / "20260901T090000-us-backtest-abcdef12"
    backtest.mkdir(parents=True)
    (backtest / "run.json").write_text(json.dumps({
        "run_id": backtest.name, "command": "backtest", "market": "us",
        "config_hash": "abcdef1234", "started_at": "2026-09-01T09:00:00+00:00",
    }))
    (backtest / "metrics.json").write_text(json.dumps({
        "sharpe": 0.62, "ann_return": 0.11, "max_drawdown": -0.14,
        "turnover": 0.31, "n_periods": 260,
    }))
    (backtest / "verdicts.json").write_text(json.dumps({
        "verdicts": ["leak check: shuffled labels find nothing",
                     "news does not measurably help"],
    }))
    (backtest / "deflated.json").write_text(json.dumps({
        "observed_sharpe": 0.62, "probability": 0.81, "n_trials": 7,
        "verdict": "marginal once the trial count is accounted for",
    }))
    (backtest / "tearsheet.html").write_text("<h1>tearsheet</h1>")
    return tmp_path


#: `csv` only, against an empty directory, so a job fails in milliseconds.
#:
#: The shipped chain ends in `synthetic`, which would happily generate eight years of bars
#: for the whole S&P 500 and then run a real backtest over it. That is a fine thing for
#: `swingbot demo` to do and a terrible thing for a unit test, so the fallback is removed
#: rather than waited on. The job machinery — worker thread, log capture, failure state —
#: is exercised either way.
NO_DATA = "data.providers=[csv]"


def _sets(runs_dir, tmp_path) -> list[str]:
    return [
        f"run.runs_dir={runs_dir}",
        f"run.data_dir={tmp_path / 'data'}",
        NO_DATA,
    ]


@pytest.fixture()
def client(runs_dir, tmp_path):
    # `set_values` is passed to create_app as well as to load_config. That is the whole
    # point: a request naming a market re-resolves the config, and it has to re-resolve
    # through these same overrides rather than through the shipped defaults — otherwise a
    # job would read the repository's real data directory instead of this empty one.
    overrides = _sets(runs_dir, tmp_path)
    cfg = load_config("us", set_values=overrides)
    with TestClient(create_app(cfg, runs_dir=runs_dir, set_values=overrides)) as test_client:
        yield test_client


# --------------------------------------------------------------------------------------
# The read-only guarantee, checked on the assembled surface
# --------------------------------------------------------------------------------------

#: Words that would appear in the path of a route that acts on a broker.
FORBIDDEN_PATH_WORDS = ("broker", "credential", "api-key", "apikey", "token", "login")


def test_no_route_looks_like_it_talks_to_a_broker(client):
    """A surface-level companion to the AST check in test_architecture.py.

    An import scan cannot see a route added through an included router or a mount, so the
    assembled route table is checked too. Cheap, and it fails for the right reason.
    """
    paths = [route.path for route in client.app.routes]
    offenders = [
        path for path in paths
        if any(word in path.lower() for word in FORBIDDEN_PATH_WORDS)
    ]
    assert not offenders, (
        f"routes that look broker-facing: {offenders}. This dashboard is read-only with "
        "respect to a broker and holds no credentials."
    )


def test_the_only_write_route_is_the_placed_checklist(client):
    """Everything that mutates state is either a job or a note about the past.

    Enumerated rather than described, so adding a POST is a deliberate act that updates
    this list — and the reviewer of that change has to say what it writes and why.
    """
    writes = sorted({
        route.path
        for route in client.app.routes
        if getattr(route, "methods", None) and route.methods & {"POST", "PUT", "PATCH", "DELETE"}
    })
    assert writes == [
        "/api/jobs",                        # starts a pipeline command
        "/api/orders/{run_id}/fills",       # what a trade actually filled at
        "/api/orders/{run_id}/placed",      # the tick list
    ], writes


# --------------------------------------------------------------------------------------
# Shape — the contract the browser extension parses
# --------------------------------------------------------------------------------------


def test_health_reports_the_environment(client):
    payload = client.get("/api/health").json()
    assert payload["ok"] is True
    assert payload["market"] == "us"
    assert payload["config_hash"]
    assert {"price", "news"} == set(payload["blend"])
    assert payload["costs"]["short_instrument"] in ("cash_equity", "futures", "none")
    # Two runs exist on disk; a dashboard that reported zero would look broken.
    assert payload["n_runs"] == 2
    assert any(row["package"] == "pandas" for row in payload["packages"])
    # The provenance key is part of the shape whether or not data loaded, so the dashboard
    # can read it without guarding on data readiness.
    assert "provenance" in payload["data"]


def test_latest_orders_carries_every_field_the_extension_reads(client):
    payload = client.get("/api/orders/latest?market=us").json()

    assert payload["run_id"] == RUN_ID
    assert payload["entry_session"] == str(ENTRY)
    assert payload["currency"] == "USD"
    assert payload["equity"] == 50000.0
    assert payload["placed"] == []

    # The notes are the reason the extension shows a book that came out small. They live
    # in book.json, which exists because targets.json carries orders and nothing else.
    assert any("participation cap" in note for note in payload["notes"])
    assert any("survivorship" in caveat for caveat in payload["caveats"])

    assert len(payload["orders"]) == 2
    for order in payload["orders"]:
        # Every key the extension's popup renders or keys off.
        assert {"ticker", "side", "quantity", "instrument", "est_value", "client_order_id"} <= set(order)

    futures = [o for o in payload["orders"] if o["instrument"] == "futures"]
    assert futures, "the fixture's futures short must survive serialisation"


def test_latest_orders_is_empty_not_an_error_before_the_first_signal(client, tmp_path):
    """A fresh install has no signal yet. That is a state, not a failure."""
    overrides = [f"run.runs_dir={tmp_path / 'empty'}", NO_DATA]
    cfg = load_config("india", set_values=overrides)
    with TestClient(create_app(cfg, runs_dir=tmp_path / "empty", set_values=overrides)) as fresh:
        payload = fresh.get("/api/orders/latest?market=india").json()
    assert payload["run_id"] == ""
    assert payload["orders"] == []


def test_run_listing_and_detail(client):
    listing = client.get("/api/runs?market=us").json()
    assert len(listing["runs"]) == 2

    signal_row = next(r for r in listing["runs"] if r["command"] == "signal")
    assert signal_row["has_model"] is True, "--use-model needs to know a model was saved"
    assert signal_row["report"] == "signal.html"

    backtest_row = next(r for r in listing["runs"] if r["command"] == "backtest")
    assert backtest_row["sharpe"] == pytest.approx(0.62)
    assert backtest_row["has_model"] is False

    detail = client.get(f"/api/runs/{RUN_ID}").json()
    assert detail["book"]["gross"] == pytest.approx(0.94)
    assert len(detail["book"]["positions"]) == 2
    assert len(detail["orders"]) == 2


def test_backtest_detail_exposes_the_honesty_checks(client):
    """The verdicts and the deflated Sharpe are the point of the checks panel."""
    listing = client.get("/api/runs?market=us&command=backtest").json()
    detail = client.get(f"/api/runs/{listing['runs'][0]['run_id']}").json()

    assert any("leak check" in verdict for verdict in detail["verdicts"])
    assert detail["deflated"]["probability"] == pytest.approx(0.81)
    assert detail["deflated"]["verdict"]


def test_a_run_report_is_served_from_disk(client):
    """Byte-for-byte the artefact in the run directory, not a re-render."""
    response = client.get(f"/api/runs/{RUN_ID}/report")
    assert response.status_code == 200
    assert "signal" in response.text


def test_an_unknown_run_is_a_404(client):
    assert client.get("/api/runs/no-such-run").status_code == 404
    assert client.get("/api/runs/no-such-run/report").status_code == 404


def test_the_dashboard_page_renders(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "swingbot" in response.text
    # The read-only promise is on the page itself, not only in the docs.
    assert "never connects to a broker" in response.text


# --------------------------------------------------------------------------------------
# The placed checklist
# --------------------------------------------------------------------------------------


def test_the_placed_checklist_round_trips_to_the_run_directory(client, runs_dir):
    """Ticks persist where the rest of the run's record lives, not in memory."""
    assert client.get(f"/api/orders/{RUN_ID}/placed").json()["placed"] == []

    saved = client.post(
        f"/api/orders/{RUN_ID}/placed",
        json={"placed": ["20260908-BBB", "20260908-AAA", "20260908-AAA"]},
    ).json()
    # Sorted and de-duplicated, so a double click cannot produce a list that disagrees
    # with itself about how many orders are outstanding.
    assert saved["placed"] == ["20260908-AAA", "20260908-BBB"]

    assert (runs_dir / RUN_ID / "execution.json").exists()
    assert client.get("/api/orders/latest?market=us").json()["placed"] == saved["placed"]


def test_a_fill_price_survives_unticking_the_row(client):
    """A tick is a checklist state; a price is a fact about the past.

    Unticking a row means "I have not placed this after all", not "the price I wrote down
    was wrong". Losing the fill on a mis-click would destroy the only record of what the
    trade actually cost, which is the whole reason for capturing it.
    """
    client.post(
        f"/api/orders/{RUN_ID}/fills",
        json={"fills": [{"client_order_id": "20260908-AAA", "price": 101.25, "placed": True}]},
    )
    client.post(f"/api/orders/{RUN_ID}/placed", json={"placed": []})

    record = client.get(f"/api/orders/{RUN_ID}/placed").json()
    assert record["placed"] == []
    assert record["orders"]["20260908-AAA"]["price"] == pytest.approx(101.25)


def test_slippage_compares_fills_against_the_model(client):
    """The loop the rest of the system was missing, checked end to end.

    The fixture's order carries `expected_slip_bps`, so the report can say not just what
    the fill cost but whether the backtest's cost assumption is holding — which is the
    only question a cost sweep cannot answer.
    """
    # AAA reference is 100.0; filling a buy at 101.0 gives up exactly 100 bps.
    client.post(
        f"/api/orders/{RUN_ID}/fills",
        json={"fills": [{"client_order_id": "20260908-AAA", "price": 101.0}]},
    )
    report = client.get(f"/api/orders/{RUN_ID}/slippage").json()

    assert report["n_filled"] == 1
    assert report["weighted_slippage_bps"] == pytest.approx(100.0, abs=0.5)
    assert report["weighted_expected_bps"] == pytest.approx(6.0)
    assert report["surprise_bps"] == pytest.approx(94.0, abs=0.5)
    # One fill is not evidence, and the verdict has to say so rather than extrapolate.
    assert "too few" in report["verdict"]


def test_slippage_refuses_an_implausible_fill(client):
    """A mistyped digit must be named, not averaged into the result.

    Twenty percent from the reference on a weekly rebalance is a typo. Letting one
    through would move the book-level number enough to make the whole report useless,
    and the user would have no idea why.
    """
    client.post(
        f"/api/orders/{RUN_ID}/fills",
        json={"fills": [{"client_order_id": "20260908-AAA", "price": 1000.0}]},
    )
    report = client.get(f"/api/orders/{RUN_ID}/slippage").json()

    assert report["n_filled"] == 0
    assert any("typo" in line for line in report["dropped"])


def test_clearing_a_price_leaves_the_tick_alone(client):
    """The mirror of the test above, and the reason ``placed`` defaults to "unchanged".

    A fill and a tick are separate facts. With ``placed`` defaulting to ``True`` on the
    request model, correcting a mistyped price by clearing the box would have re-ticked —
    or, sending ``placed: false`` alongside, un-ticked — an order whose placement status
    the user never touched.
    """
    client.post(
        f"/api/orders/{RUN_ID}/fills",
        json={"fills": [{"client_order_id": "20260908-AAA", "price": 101.25, "placed": True}]},
    )
    # Clear the price only.
    client.post(
        f"/api/orders/{RUN_ID}/fills",
        json={"fills": [{"client_order_id": "20260908-AAA", "price": None}]},
    )

    record = client.get(f"/api/orders/{RUN_ID}/placed").json()
    assert record["placed"] == ["20260908-AAA"], "clearing a price must not un-tick"
    assert "price" not in record["orders"]["20260908-AAA"], "the price should be gone"


def test_a_tick_only_update_never_erases_a_recorded_fill(client):
    """Omitting ``price`` leaves the stored one, so a partial update cannot destroy it."""
    client.post(
        f"/api/orders/{RUN_ID}/fills",
        json={"fills": [{"client_order_id": "20260908-AAA", "price": 101.25}]},
    )
    client.post(
        f"/api/orders/{RUN_ID}/fills",
        json={"fills": [{"client_order_id": "20260908-AAA", "placed": True}]},
    )

    record = client.get(f"/api/orders/{RUN_ID}/placed").json()
    assert record["orders"]["20260908-AAA"]["price"] == pytest.approx(101.25)


# --------------------------------------------------------------------------------------
# Cost history — one week is noise, a run of weeks is evidence
# --------------------------------------------------------------------------------------


def test_cost_history_says_nothing_until_a_fill_exists(client):
    payload = client.get("/api/tca?market=us").json()
    assert payload["weeks"] == []
    assert payload["summary"]["n_weeks"] == 0
    assert "no fills recorded" in payload["summary"]["verdict"]


def test_cost_history_refuses_to_draw_a_conclusion_from_one_week(client):
    """The judgement that keeps this honest.

    Weekly slippage is dominated by which way the open happened to gap. A single week
    saying "your broker is cheap" is exactly the kind of number that gets acted on, so
    the verdict names the sample size instead of reporting a comparison.
    """
    client.post(
        f"/api/orders/{RUN_ID}/fills",
        json={"fills": [{"client_order_id": "20260908-AAA", "price": 101.0}]},
    )
    payload = client.get("/api/tca?market=us").json()

    assert len(payload["weeks"]) == 1
    week = payload["weeks"][0]
    assert week["weighted_slippage_bps"] == pytest.approx(100.0, abs=0.5)
    assert "fills" not in week, "the history view is one row per week, not per fill"
    assert payload["summary"]["n_weeks"] == 1
    assert "too few" in payload["summary"]["verdict"]


def test_cost_history_pools_weeks_by_notional(tmp_path):
    """Weighted, not a mean of weekly means.

    A week where you traded ten times as much has to count ten times as much. Averaging
    the weekly averages is how one small, badly-executed week comes to decide what the
    cost model is judged against.
    """
    runs = tmp_path / "runs"
    for week in range(4):
        run_id = f"2026090{week + 1}T163000-us-signal-aaaaaaaa"
        directory = runs / run_id
        directory.mkdir(parents=True)
        # Week 0 trades 10x the size of the others and executes at 10 bps; the rest
        # execute at 110 bps. A plain mean of the four weeks gives 85 bps; weighting by
        # notional gives ~33, and the two answers imply opposite verdicts about the model.
        quantity, price = (1000, 100.10) if week == 0 else (100, 101.10)
        (directory / "orders.csv").write_text(
            "ticker,side,quantity,order_type,instrument,est_price,est_value,"
            "client_order_id,expected_slip_bps\n"
            f"AAA,buy,{quantity},market,equity,100.0,{quantity * 100.0},X{week},40.0\n"
        )
        (directory / "run.json").write_text(json.dumps({
            "run_id": run_id, "command": "signal", "market": "us",
            "started_at": f"2026-09-0{week + 1}T16:30:00+00:00",
        }))
        (directory / "execution.json").write_text(json.dumps({
            "run_id": run_id,
            "orders": {f"X{week}": {"placed": True, "price": price}},
        }))

    cfg = load_config("us", set_values=[f"run.runs_dir={runs}", "data.providers=[csv]"])
    with TestClient(create_app(cfg)) as client:
        summary = client.get("/api/tca?market=us").json()["summary"]

    assert summary["n_weeks"] == 4
    assert summary["realised_bps"] == pytest.approx(33.2, abs=1.0), (
        "a mean of the weekly averages would give ~85 bps and condemn a cost model that "
        "is in fact holding up"
    )
    assert summary["expected_bps"] == pytest.approx(40.0, abs=0.5)
    assert summary["surprise_bps"] == pytest.approx(-6.8, abs=1.0)
    assert "holding up" in summary["verdict"]


def test_cost_history_calls_out_a_cost_model_that_is_understating(tmp_path):
    """The finding that matters, and the reason any of this exists.

    On real NSE data this strategy earned +7.4%/yr gross and paid −8.3%/yr in *modelled*
    costs. If the model understates what a broker really charges, the backtest is
    optimistic by that much on every round trip and the dashboard must say so.
    """
    runs = tmp_path / "runs"
    for week in range(5):
        run_id = f"2026090{week + 1}T163000-us-signal-bbbbbbbb"
        directory = runs / run_id
        directory.mkdir(parents=True)
        (directory / "orders.csv").write_text(
            "ticker,side,quantity,order_type,instrument,est_price,est_value,"
            "client_order_id,expected_slip_bps\n"
            f"AAA,buy,100,market,equity,100.0,10000.0,X{week},10.0\n"
        )
        (directory / "run.json").write_text(json.dumps({
            "run_id": run_id, "command": "signal", "market": "us",
            "started_at": f"2026-09-0{week + 1}T16:30:00+00:00",
        }))
        (directory / "execution.json").write_text(json.dumps({
            "run_id": run_id,
            # Filled 80 bps worse than the reference against a model expecting 10.
            "orders": {f"X{week}": {"placed": True, "price": 100.80}},
        }))

    cfg = load_config("us", set_values=[f"run.runs_dir={runs}", "data.providers=[csv]"])
    with TestClient(create_app(cfg)) as client:
        summary = client.get("/api/tca?market=us").json()["summary"]

    assert summary["surprise_bps"] == pytest.approx(70.0, abs=1.0)
    assert "understating" in summary["verdict"]
    assert "optimistic" in summary["verdict"]


def test_the_checklist_refuses_an_unknown_run(client):
    response = client.post("/api/orders/nope/placed", json={"placed": ["x"]})
    assert response.status_code == 404


# --------------------------------------------------------------------------------------
# Jobs
# --------------------------------------------------------------------------------------


def test_a_request_for_another_market_keeps_the_server_overrides(client, runs_dir, tmp_path):
    """Naming a market must not reset the config to the shipped defaults.

    This is a regression test for a real bug. `config_for` used to rebuild from scratch
    whenever a market was given, discarding the server's `--profile` and every `--set` it
    was started with. The visible symptom was a job reading a different `run.data_dir`
    than the dashboard was displaying — which in a test meant a "no data" job quietly
    running a full backtest against the repository's real price cache.
    """
    payload = client.get("/api/health?market=india").json()
    assert payload["market"] == "india", "the requested market must be honoured"
    # `data.providers=[csv]` came from the server's overrides. If they had been dropped,
    # the chain would end in `synthetic` and report data as ready.
    assert payload["data"]["ready"] is False, (
        "the server's --set overrides were discarded when re-resolving for india"
    )


def test_an_unknown_job_kind_is_rejected_with_a_useful_message(client):
    response = client.post("/api/jobs", json={"kind": "rm-rf", "market": "us"})
    assert response.status_code == 400
    assert "known" in response.json()["detail"].lower()


def test_a_job_runs_and_captures_the_pipeline_log(client):
    """A job on an empty data directory must fail *cleanly*, and say why in its log.

    Deliberately a failing job: it is fast, it needs no synthetic market, and it exercises
    the part most likely to be wrong — that an exception inside the worker thread becomes
    a reported job state rather than a hung poll or a 500 on the next request. The log
    assertion is what proves the logging-handler wiring works at all, and that wiring is
    the whole reason the browser can watch a backtest.
    """
    started = client.post("/api/jobs", json={"kind": "backtest", "market": "us"}).json()
    assert started["state"] in ("queued", "running")

    for _ in range(200):
        job = client.get(f"/api/jobs/{started['id']}").json()
        if job["state"] in ("done", "failed"):
            break
        time.sleep(0.05)
    else:  # pragma: no cover - only on a pathologically slow machine
        pytest.fail(f"job never finished: {job}")

    assert job["state"] == "failed", "no price data exists, so the backtest cannot run"
    assert job["error"], "a failed job must carry its reason"
    assert job["log"], "the job captured no log lines — the logging handler is not wired up"

    listing = client.get("/api/jobs").json()
    assert any(entry["id"] == started["id"] for entry in listing["jobs"])


def test_job_log_polling_returns_only_new_lines(client):
    """The browser appends; it must not be handed the whole log on every poll."""
    started = client.post("/api/jobs", json={"kind": "backtest", "market": "us"}).json()
    for _ in range(200):
        job = client.get(f"/api/jobs/{started['id']}").json()
        if job["state"] in ("done", "failed"):
            break
        time.sleep(0.05)

    full = client.get(f"/api/jobs/{started['id']}?log_from=0").json()
    assert full["log"]
    tail = client.get(f"/api/jobs/{started['id']}?log_from={full['n_log']}").json()
    assert tail["log"] == []

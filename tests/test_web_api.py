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


ORDERS_CSV = """ticker,side,quantity,order_type,instrument,limit_price,est_price,est_value,client_order_id
AAA,buy,12,market,equity,,100.0,1200.0,20260908-AAA
BBB,sell,8,market,futures,,50.0,400.0,20260908-BBB
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
    assert writes == ["/api/jobs", "/api/orders/{run_id}/placed"], writes


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

    assert (runs_dir / RUN_ID / "placed.json").exists()
    assert client.get("/api/orders/latest?market=us").json()["placed"] == saved["placed"]


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

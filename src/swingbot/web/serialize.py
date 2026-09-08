"""Result objects and run directories to JSON-safe dictionaries.

Kept separate from the routes for one reason: JSON has no NaN, no ``numpy.float64``, no
``datetime.date``, and pandas produces all three constantly. A single ``jsonable`` pass
here means a route never has to think about it, and a NaN Sharpe on a two-week backtest
becomes ``null`` rather than a 500 from the encoder.

The shapes defined here are the contract the dashboard and the browser extension both
read, so a field rename is a change to a published interface, not an internal detail.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..io.store import read_json


def jsonable(value: Any) -> Any:
    """Recursively convert pandas/numpy/date values into JSON-safe Python.

    ``NaN`` and ``inf`` become ``None``. That loses the distinction between "not computed"
    and "not finite", which is the right trade for a display layer: both mean "there is no
    number to show here", and a client that receives ``null`` renders an em dash instead of
    crashing on ``NaN`` in strict JSON.
    """
    if value is None:
        return None
    if isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        return None if (math.isnan(value) or math.isinf(value)) else value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return None if (math.isnan(number) or math.isinf(number)) else number
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, datetime | pd.Timestamp):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set):
        return [jsonable(v) for v in value]
    if isinstance(value, pd.Series):
        return jsonable(value.to_dict())
    if isinstance(value, pd.DataFrame):
        return frame_rows(value)
    if value is pd.NaT:
        return None
    return str(value)


def frame_rows(frame: pd.DataFrame | None, *, limit: int | None = None) -> list[dict[str, Any]]:
    """A DataFrame as a list of JSON-safe row dicts."""
    if frame is None or frame.empty:
        return []
    block = frame.head(limit) if limit else frame
    return [jsonable(row) for row in block.to_dict("records")]


# --------------------------------------------------------------------------------------
# Orders — the shape the browser extension reads
# --------------------------------------------------------------------------------------


def orders_payload(
    *,
    run_id: str,
    market: str,
    orders: list[dict[str, Any]],
    decision_session: Any = "",
    entry_session: Any = "",
    currency: str = "",
    equity: float | None = None,
    generated_at: str = "",
    notes: list[str] | None = None,
    caveats: list[str] | None = None,
    placed: list[str] | None = None,
    sequence: list[dict[str, Any]] | None = None,
    execution: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """The one order shape. Both the dashboard and the extension render from this."""
    return jsonable({
        "run_id": run_id,
        "market": market,
        "currency": currency,
        "decision_session": decision_session,
        "entry_session": entry_session,
        "equity": equity,
        "generated_at": generated_at,
        "notes": list(notes or []),
        "caveats": list(caveats or []),
        "orders": orders,
        "placed": list(placed or []),
        # The order to work the tickets in, plus what has actually been done so far.
        # Both surfaces render from these rather than from the file's own row order,
        # which is sorted for tidiness and is close to the worst order to trade in.
        "sequence": list(sequence or []),
        "execution": dict(execution or {}),
    })


def signal_payload(outcome, *, market: str, placed: list[str] | None = None) -> dict[str, Any]:
    """A :class:`~swingbot.service.SignalOutcome` as JSON."""
    payload = orders_payload(
        run_id=outcome.run_id,
        market=market,
        orders=frame_rows(outcome.orders_frame),
        decision_session=outcome.decision_session,
        entry_session=outcome.entry_session,
        currency=outcome.currency,
        equity=outcome.equity,
        notes=outcome.notes,
        caveats=outcome.caveats,
        placed=placed,
    )
    payload.update(jsonable({
        "book": frame_rows(outcome.book_frame),
        "gross": outcome.portfolio.gross,
        "net": outcome.portfolio.net,
        "n_long": outcome.portfolio.n_long,
        "n_short": outcome.portfolio.n_short,
        "risk_scale": outcome.portfolio.risk_scale,
        "calibration": outcome.calibration,
        "pinned_model": outcome.pinned_model,
        "notified": outcome.notified,
        "messages": list(outcome.execution.messages),
        "directory": outcome.directory,
    }))
    return payload


def backtest_payload(outcome) -> dict[str, Any]:
    """A :class:`~swingbot.service.BacktestOutcome` as JSON."""
    result = outcome.result
    return jsonable({
        "run_id": outcome.run_id,
        "variant": outcome.variant,
        "directory": outcome.directory,
        "metrics": result.metrics.to_dict(),
        "cost_breakdown": result.cost_breakdown,
        "notes": list(result.notes),
        "caveats": list(outcome.caveats),
        "panel_rows": outcome.panel_rows,
        "panel_weeks": outcome.panel_weeks,
        "n_price_features": outcome.n_price_features,
        "n_news_features": outcome.n_news_features,
        "n_trials": outcome.n_trials,
        "deflated": outcome.deflated.to_dict() | {"verdict": outcome.deflated.verdict()},
        "verdicts": outcome.verdicts,
        "ablation": frame_rows(outcome.ablation.table()) if outcome.ablation is not None else [],
        "news_effect": (
            outcome.ablation.news_effect
            if outcome.ablation is not None and outcome.ablation.news_effect
            else None
        ),
        "cost_rows": frame_rows(outcome.cost_rows),
        "pbo": (
            outcome.pbo.to_dict() | {"paths": frame_rows(outcome.pbo.table())}
            if outcome.pbo is not None and outcome.pbo.is_valid
            else None
        ),
        "has_tearsheet": outcome.tearsheet is not None,
    })


# --------------------------------------------------------------------------------------
# Run directories
# --------------------------------------------------------------------------------------

#: Report files a run may contain, in the order the UI should prefer them.
REPORT_FILES = ("tearsheet.html", "signal.html")


def run_summary(directory: Path, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    """One row for the run history list.

    Cheap on purpose — this reads only the small JSON files, never the parquet or the
    tearsheet, because the history view lists dozens of runs at once.
    """
    meta = meta if meta is not None else read_json(directory / "run.json", {}) or {}
    metrics = read_json(directory / "metrics.json", {}) or {}
    report = next((name for name in REPORT_FILES if (directory / name).exists()), "")

    return jsonable({
        "run_id": meta.get("run_id", directory.name),
        "command": meta.get("command", ""),
        "market": meta.get("market", ""),
        "config_hash": meta.get("config_hash", ""),
        "git_revision": meta.get("git_revision", ""),
        "started_at": meta.get("started_at", ""),
        "finished_at": meta.get("finished_at", ""),
        "duration_seconds": meta.get("duration_seconds"),
        "summary": meta.get("summary", {}),
        "pinned_model": meta.get("pinned_model", ""),
        # What --use-model needs: a run is only reproducible if it saved its model.
        "has_model": (directory / "model" / "model.pkl").exists(),
        "report": report,
        "sharpe": metrics.get("sharpe"),
        "ann_return": metrics.get("ann_return"),
        "max_drawdown": metrics.get("max_drawdown"),
        "turnover": metrics.get("turnover"),
        "n_periods": metrics.get("n_periods"),
    })


def run_detail(directory: Path) -> dict[str, Any]:
    """Everything about one run that the dashboard shows, read back from disk.

    Read from the run directory rather than from a cached in-memory object, so a run made
    by the CLI weeks ago renders identically to one the dashboard just produced. The run
    directory is the system's record; this is the reader for it.
    """
    detail = run_summary(directory)
    detail["metrics"] = jsonable(read_json(directory / "metrics.json", {}) or {})
    detail["deflated"] = jsonable(read_json(directory / "deflated.json", {}) or {})
    detail["pbo"] = jsonable(read_json(directory / "pbo.json", {}) or {})
    detail["verdicts"] = list(
        (read_json(directory / "verdicts.json", {}) or {}).get("verdicts", [])
    )
    detail["targets"] = jsonable(read_json(directory / "targets.json", {}) or {})
    # book.json carries the target book and the portfolio's own notes. targets.json is the
    # execution adapter's file and holds only the orders.
    detail["book"] = jsonable(read_json(directory / "book.json", {}) or {})
    detail["model_card"] = jsonable(read_json(directory / "model" / "model_card.json", {}) or {})
    execution = read_json(directory / "execution.json", {}) or {}
    orders_record = execution.get("orders")
    if isinstance(orders_record, dict):
        detail["execution"] = jsonable(orders_record)
        detail["placed"] = sorted(
            k for k, v in orders_record.items() if isinstance(v, dict) and v.get("placed")
        )
    else:
        # Runs written before fill capture existed carry only a tick list.
        detail["execution"] = {}
        detail["placed"] = list(
            (read_json(directory / "placed.json", {}) or {}).get("placed", [])
        )

    detail["orders"] = _read_csv_rows(directory / "orders.csv")
    detail["ablation"] = _read_csv_rows(directory / "ablation.csv")
    detail["cost_rows"] = _read_csv_rows(directory / "cost_sensitivity.csv")
    detail["pbo_paths"] = _read_csv_rows(directory / "pbo_paths.csv")
    return detail


def _read_csv_rows(path: Path, *, limit: int = 500) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        return frame_rows(pd.read_csv(path), limit=limit)
    except Exception:
        # A truncated or half-written CSV should show as "no rows" in the UI, not take the
        # whole run detail page down with it.
        return []

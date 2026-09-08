"""The service layer's contract with its two front ends.

Most of what `service.py` does is covered already: the golden backtest exercises
`run_backtest`, `test_model_pinning.py` covers `load_pinned_model`, and
`test_web_api.py` drives the whole thing over HTTP. What is left, and what is here, are the
small pieces where a wrong answer is silent.

`_held_weights` gets the most attention, because its absence was a live bug. `signal` used
to pass ``adapter.current_positions() and None`` into the constructor — an expression that
evaluates to ``None`` for every possible input — so the no-trade band never engaged in
production. The band is the largest single turnover saving in the system, and every
backtest counted it while no live run ever received it: reported Sharpe was better than the
orders could have achieved, in the one direction that flatters.
"""

from __future__ import annotations

import datetime as dt

import pytest

from swingbot.config import load_config
from swingbot.service import (
    DoctorReport,
    ServiceError,
    _as_date,
    _held_weights,
    doctor_report,
)

# --------------------------------------------------------------------------------------
# Holdings to weights — the no-trade band's missing input
# --------------------------------------------------------------------------------------


def test_holdings_become_signed_weights():
    """Shares times price over equity, with shorts staying negative."""
    weights = _held_weights(
        {"AAA": 100.0, "BBB": -50.0},
        {"AAA": 20.0, "BBB": 40.0},
        equity=10_000.0,
    )
    assert weights == pytest.approx({"AAA": 0.20, "BBB": -0.20})


def test_a_non_empty_book_never_converts_to_nothing():
    """The regression guard.

    The old expression ``current_positions() and None`` is ``None`` for every input, so it
    passed review looking like a placeholder and behaved like a disabled feature. Any
    replacement has to return something for a real book.
    """
    weights = _held_weights({"AAA": 10.0}, {"AAA": 100.0}, equity=1_000.0)
    assert weights, "holdings with a price and an equity must produce weights"


def test_no_holdings_means_no_previous_weights():
    """``None``, not ``{}``.

    The constructor treats a falsy value as "there is no previous book", which skips the
    band entirely. An empty dict would take the same path, but saying ``None`` keeps the
    intent legible at the call site: the first run of a fresh account has no history, it
    does not have an empty history.
    """
    assert _held_weights({}, {"AAA": 10.0}, equity=1000.0) is None
    assert _held_weights({"AAA": 1.0}, {}, equity=1000.0) is None


def test_a_name_with_no_price_is_dropped_not_valued_at_zero():
    """A missing price means unknown weight, which is not the same as no position.

    Valuing it at zero would tell the band the position had been closed, and the band
    would then happily leave a real holding untouched while believing it was flat.
    """
    weights = _held_weights(
        {"AAA": 100.0, "HALTED": 100.0}, {"AAA": 20.0}, equity=10_000.0
    )
    assert set(weights) == {"AAA"}


def test_zero_or_negative_equity_is_refused():
    """Dividing by it would produce infinities that propagate into every weight."""
    assert _held_weights({"AAA": 1.0}, {"AAA": 10.0}, equity=0.0) is None
    assert _held_weights({"AAA": 1.0}, {"AAA": 10.0}, equity=-5.0) is None


# --------------------------------------------------------------------------------------
# Date coercion
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("", None),
        ("not-a-date", None),
        ("2026-09-04", dt.date(2026, 9, 4)),
        (dt.date(2026, 9, 4), dt.date(2026, 9, 4)),
    ],
)
def test_as_date_accepts_what_a_front_end_actually_sends(value, expected):
    """A web form sends ``""`` and a CLI flag sends ``None``; both mean "not specified".

    Returning ``None`` for junk rather than raising is deliberate here: the caller's next
    step is to fall back to the latest decision date, which is the right answer for an
    unspecified date and a much better outcome than a 500 from a typo in a query string.
    """
    assert _as_date(value) == expected


# --------------------------------------------------------------------------------------
# doctor
# --------------------------------------------------------------------------------------


def test_doctor_never_raises_on_a_machine_with_no_data(tmp_path):
    """`doctor` exists to explain a broken setup, so it must survive one.

    A fresh clone has no price data at all. If this raised, the one command whose job is to
    tell you what is missing would be the command that could not run.
    """
    cfg = load_config(
        "us",
        set_values=[
            f"run.data_dir={tmp_path / 'data'}",
            f"run.runs_dir={tmp_path / 'runs'}",
            "data.providers=[csv]",
        ],
    )
    report = doctor_report(cfg)

    assert isinstance(report, DoctorReport)
    assert report.data_ready is False
    assert report.data_error, "a failure must come back with its reason attached"
    assert report.market == "us"
    assert report.config_hash
    # The package census still works — that is the half of `doctor` that always applies.
    assert any(row["package"] == "pandas" and row["status"] == "ok" for row in report.packages)


def test_optional_packages_are_reported_as_optional_not_missing():
    """A missing optional extra is information, not an error.

    `fastapi` is in this list because the dashboard is an optional install; someone running
    `doctor` on a CLI-only setup should not be told something is MISSING when nothing is
    wrong.
    """
    report = doctor_report(load_config("us"))
    statuses = {row["package"]: row["status"] for row in report.packages}
    for name in ("lightgbm", "duckdb", "anthropic", "transformers", "fastapi"):
        assert statuses[name] in ("ok", "optional"), (name, statuses[name])
    assert statuses["pandas"] == "ok"


# --------------------------------------------------------------------------------------
# The error type both front ends depend on
# --------------------------------------------------------------------------------------


def test_service_error_is_catchable_without_importing_a_front_end():
    """The reason this type exists.

    These refusals used to be raised as ``typer.Exit``, which meant an HTTP handler had to
    import the CLI framework to catch "not enough history for the configured folds" — and
    if it forgot, the user got a 500 instead of the sentence explaining what to change.
    """
    assert issubclass(ServiceError, RuntimeError)
    assert not issubclass(ServiceError, SystemExit)

    with pytest.raises(ServiceError, match="wrong turn"):
        raise ServiceError("you took a wrong turn")

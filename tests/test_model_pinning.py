"""Model pinning: reproducing a past decision instead of refitting near it.

`signal` used to refit the model from scratch every week and save nothing. The book it
produced could be described — here are the weights, here are the orders — but never
explained, because the function that generated it no longer existed anywhere. "Why did it
buy that?" had no answer, and `SchemaMismatchError`, written to catch exactly the failure
a stale model causes, could never fire because no model was ever loaded.

Two guarantees are worth testing. A pinned model must reproduce its predictions *exactly*,
because approximately is useless for an audit. And it must refuse to be used when refusal
is the safe answer: a changed feature schema silently misaligns columns, and a fit from
last year is not a forecast for this week.
"""

from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

import pytest
import typer

from swingbot.cli import _load_pinned_model
from swingbot.config import load_config
from swingbot.features.pipeline import price_feature_columns
from swingbot.model import build_model
from swingbot.model.registry import SchemaMismatchError, load_model, save_model

TRAIN_END = dt.date(2023, 6, 30)


def _decision(session: dt.date):
    return SimpleNamespace(decision_session=session)


@pytest.fixture(scope="module")
def fitted(alpha_panel, cfg_us):
    panel, _, _ = alpha_panel
    columns = price_feature_columns(panel)
    model = build_model("ridge", cfg_us, seed=7)
    model.fit(
        panel[columns], panel["label"],
        sample_weight=panel.get("sample_weight"), week_index=panel["decision_session"],
    )
    return model, columns, panel


def _save(fitted, tmp_path, cfg, *, run_id="20230630T000000-us-signal-abcdef12", train_end=TRAIN_END):
    model, columns, panel = fitted
    directory = tmp_path / run_id
    save_model(
        model, directory / "model",
        market="us", config_hash=cfg.config_hash(),
        train_start=dt.date(2018, 1, 5), train_end=train_end,
        n_train_rows=len(panel), price_columns=columns,
    )
    return run_id


def test_a_pinned_model_reproduces_its_predictions_exactly(fitted, tmp_path):
    """Bit-for-bit, not close. An audit trail that drifts is not an audit trail."""
    model, columns, panel = fitted
    week = panel.loc[panel["decision_session"] == panel["decision_session"].max()]

    save_model(model, tmp_path / "model", market="us", config_hash="x", price_columns=columns)
    reloaded, card = load_model(tmp_path / "model", expected_features=columns)

    before = model.predict(week[columns]).to_numpy()
    after = reloaded.predict(week[columns]).to_numpy()
    assert (before == after).all(), "a reloaded model gave different predictions"
    assert card is not None and card.feature_names


def test_loading_refuses_a_feature_schema_that_no_longer_matches(fitted, tmp_path):
    """The failure this guards is silent, which is why it has to be loud.

    Drop a feature and the estimator does not complain: the remaining columns shift into
    the missing one's position and it keeps emitting confident numbers that mean nothing.
    """
    model, columns, _ = fitted
    save_model(model, tmp_path / "model", market="us", config_hash="x", price_columns=columns)

    with pytest.raises(SchemaMismatchError):
        load_model(tmp_path / "model", expected_features=columns[:-3])


def test_a_stale_pinned_model_is_refused(fitted, tmp_path):
    """A fit from long before the decision date is not a forecast for it."""
    cfg = load_config("us", set_values=[f"run.runs_dir={tmp_path}", "model.max_model_age_weeks=8"])
    run_id = _save(fitted, tmp_path, cfg)
    _, columns, _ = fitted

    with pytest.raises(typer.Exit):
        _load_pinned_model(cfg, run_id, columns, _decision(TRAIN_END + dt.timedelta(weeks=40)))


def test_a_fresh_pinned_model_is_accepted(fitted, tmp_path):
    cfg = load_config("us", set_values=[f"run.runs_dir={tmp_path}", "model.max_model_age_weeks=8"])
    run_id = _save(fitted, tmp_path, cfg)
    _, columns, _ = fitted

    model, card = _load_pinned_model(
        cfg, run_id, columns, _decision(TRAIN_END + dt.timedelta(weeks=2))
    )
    assert model is not None
    assert card.train_end == str(TRAIN_END)


def test_the_staleness_limit_is_the_thing_that_decides(fitted, tmp_path):
    """Raising the limit must be what makes an old model usable — nothing else.

    Reproducing a decision from two years ago is a legitimate thing to want. It should
    require saying so, which is what makes the default a guard rather than an obstacle.
    """
    _, columns, _ = fitted
    old = _decision(TRAIN_END + dt.timedelta(weeks=40))

    strict = load_config("us", set_values=[f"run.runs_dir={tmp_path}", "model.max_model_age_weeks=8"])
    run_id = _save(fitted, tmp_path, strict)
    with pytest.raises(typer.Exit):
        _load_pinned_model(strict, run_id, columns, old)

    permissive = load_config(
        "us", set_values=[f"run.runs_dir={tmp_path}", "model.max_model_age_weeks=104"]
    )
    model, _ = _load_pinned_model(permissive, run_id, columns, old)
    assert model is not None


def test_an_unknown_run_id_fails_rather_than_refitting(tmp_path):
    """Silently refitting when the pin cannot be found would defeat the whole point."""
    cfg = load_config("us", set_values=[f"run.runs_dir={tmp_path}"])
    with pytest.raises(typer.Exit):
        _load_pinned_model(cfg, "no-such-run", ["a"], _decision(TRAIN_END))


def test_a_run_without_a_saved_model_fails_clearly(tmp_path):
    cfg = load_config("us", set_values=[f"run.runs_dir={tmp_path}"])
    (tmp_path / "20230101T000000-us-signal-deadbeef").mkdir(parents=True)
    with pytest.raises(typer.Exit):
        _load_pinned_model(
            cfg, "20230101T000000-us-signal-deadbeef", ["a"], _decision(TRAIN_END)
        )

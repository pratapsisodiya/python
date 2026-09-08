"""Probability of backtest overfitting: the statistic, and the runner around it.

Two layers, tested separately because they fail differently.

The **statistic** is arithmetic and is tested against matrices whose answer is known by
construction. There is no sampling noise to hide behind: a matrix built so the in-sample
winner always finishes last must yield exactly 1.0.

The **runner** is where the mistakes actually live — a fold that trains on its own test
window, a recorded "winner" that is not the in-sample maximum, an out-of-sample rank that
does not match the row it came from. Those are exact properties, so they are asserted
exactly rather than statistically. A statistical assertion over six combinatorial paths
would be a coin flip dressed as a test.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from swingbot.backtest.overfit import Candidate, default_candidates, run_pbo
from swingbot.config import load_config
from swingbot.types import DECISION_SESSION, LABEL, LABEL_T0, LABEL_T1, TICKER
from swingbot.validation.deflated import probability_of_backtest_overfitting

# A deliberately small grid: the runner fits every candidate on every path, so the default
# six-candidate grid over 6 paths is 36 model fits, which is not a unit test.
CANDIDATES = [
    Candidate("ridge_light", "ridge", {"alpha": 1.0}),
    Candidate("ridge_heavy", "ridge", {"alpha": 500.0}),
]

N_NAMES = 30
N_WEEKS = 120


# --------------------------------------------------------------------------------- stat


def test_pbo_is_one_when_the_in_sample_winner_always_finishes_last():
    """Constructed so selection is maximally wrong. The answer is 1.0, not "high"."""
    paths, configs = 10, 5
    in_sample = np.tile(np.arange(configs, dtype=float), (paths, 1))
    out_of_sample = np.tile(np.arange(configs, dtype=float)[::-1], (paths, 1))
    assert probability_of_backtest_overfitting(in_sample, out_of_sample) == 1.0


def test_pbo_is_zero_when_the_ranking_is_stable():
    paths, configs = 10, 5
    ranking = np.tile(np.arange(configs, dtype=float), (paths, 1))
    assert probability_of_backtest_overfitting(ranking, ranking) == 0.0


def test_pbo_rejects_mismatched_matrices():
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(np.zeros((3, 4)), np.zeros((3, 5)))


# ------------------------------------------------------------------------------- runner


@pytest.fixture(scope="module")
def noise_panel():
    """A panel with real features and labels that owe them nothing.

    Pure noise is the right fixture for the runner's structural properties: any candidate
    that appears to win did so by luck, which is exactly the regime PBO exists to measure,
    and it keeps the test from accidentally depending on a model actually working.
    """
    rng = np.random.default_rng(4)
    rows = []
    week = dt.date(2018, 1, 5)
    for _ in range(N_WEEKS):
        features = rng.normal(0, 1, (N_NAMES, 4))
        rows.append(
            pd.DataFrame(
                {
                    TICKER: [f"T{i:02d}" for i in range(N_NAMES)],
                    DECISION_SESSION: week,
                    LABEL_T0: week + dt.timedelta(days=3),
                    LABEL_T1: week + dt.timedelta(days=10),
                    "px_a": features[:, 0],
                    "px_b": features[:, 1],
                    "px_c": features[:, 2],
                    "px_d": features[:, 3],
                    LABEL: rng.normal(0, 0.03, N_NAMES),
                    "sample_weight": 1.0,
                }
            )
        )
        week += dt.timedelta(days=7)
    return pd.concat(rows, ignore_index=True)


@pytest.fixture(scope="module")
def pbo_result(noise_panel):
    cfg = load_config("us")
    return run_pbo(
        noise_panel,
        cfg,
        feature_columns=["px_a", "px_b", "px_c", "px_d"],
        candidates=CANDIDATES,
        n_groups=4,
        n_test_groups=2,
    )


def test_the_runner_produces_paths(pbo_result):
    assert pbo_result.is_valid
    assert pbo_result.n_paths >= 4, "four groups taken two at a time should give six paths"
    assert pbo_result.in_sample.shape == (pbo_result.n_paths, len(CANDIDATES))
    assert pbo_result.in_sample.shape == pbo_result.out_of_sample.shape
    assert 0.0 <= pbo_result.pbo <= 1.0


def test_the_recorded_winner_is_the_in_sample_maximum(pbo_result):
    """The selection record has to describe what the matrix actually says.

    A mismatch here would mean the reported PBO and the reported per-path table are
    describing different runs, which is the kind of discrepancy nobody notices until the
    number is being defended.
    """
    for row, scores in zip(pbo_result.selections, pbo_result.in_sample, strict=True):
        expected = pbo_result.candidates[int(np.nanargmax(scores))]
        assert row["chosen"] == expected, row


def test_the_out_of_sample_rank_matches_the_matrix(pbo_result):
    for row, oos in zip(pbo_result.selections, pbo_result.out_of_sample, strict=True):
        chosen = pbo_result.candidates.index(row["chosen"])
        expected_rank = int(np.argsort(np.argsort(-oos))[chosen]) + 1
        assert row["out_of_sample_rank"] == expected_rank, row


def test_combinatorial_folds_never_train_on_their_own_test_window(noise_panel):
    """The purge, checked on the splitter the PBO runner actually uses.

    Combinatorial folds have training data on both sides of every test block, so a label
    that spans the boundary can leak in either direction. This is the property that makes
    the resulting PBO worth reading at all: paths built from contaminated folds would
    produce an encouragingly low number for entirely the wrong reason.
    """
    from swingbot.validation.splits import CombinatorialPurgedCV

    splitter = CombinatorialPurgedCV(n_groups=4, n_test_groups=2, embargo_weeks=2)
    folds = list(splitter.split(noise_panel))
    assert folds, "fixture produced no combinatorial folds"

    for fold in folds:
        train = noise_panel.iloc[fold.train_idx]
        test = noise_panel.iloc[fold.test_idx]
        assert not set(train.index) & set(test.index)

        lo = pd.to_datetime(test[LABEL_T0]).min()
        hi = pd.to_datetime(test[LABEL_T1]).max()
        t0 = pd.to_datetime(train[LABEL_T0])
        t1 = pd.to_datetime(train[LABEL_T1])
        overlapping = ((t0 <= hi) & (t1 >= lo)).sum()
        assert overlapping == 0, (
            f"fold {fold.index} trains on {overlapping} row(s) whose label span crosses "
            f"its test window {lo.date()} to {hi.date()}"
        )


def test_the_default_grid_spans_both_model_families():
    """PBO measures the cost of choosing, so the grid has to contain a real choice."""
    kinds = {c.kind for c in default_candidates(load_config("us"))}
    assert kinds == {"ridge", "gbdt"}


def test_an_empty_panel_reports_invalid_rather_than_zero():
    """PBO 0.00 means "selection generalises". An empty run must not claim that."""
    result = run_pbo(pd.DataFrame(), load_config("us"), feature_columns=[])
    assert not result.is_valid
    assert "not enough history" in result.verdict()

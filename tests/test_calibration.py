"""Score calibration, and the point-in-time rule that makes it legitimate.

A calibrator maps a model's score onto an expected return, which is what any decision
phrased in money — the Kelly ceiling, position sizing in currency, an expected-cost
comparison — actually needs. The map is trivial to fit and trivial to fit *wrongly*: fit
it on the predictions it will scale and it has been shown the answer, at which point the
backtest reports a beautifully calibrated model that falls apart the first live week.

So the tests here are less about the isotonic fit, which sklearn already tests, than about
where its training data came from. The load-bearing assertion is
:func:`test_a_fold_is_calibrated_only_on_strictly_earlier_folds`, which plants an inverted
relationship inside the final fold and requires the calibration for that fold to be blind
to it.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from swingbot.model.calibrate import (
    MIN_OBSERVATIONS,
    ScoreCalibrator,
    attach_expected_returns,
    calibration_report,
    calibrators_by_fold,
)
from swingbot.types import DECISION_SESSION, LABEL

N_FOLDS = 3
WEEKS_PER_FOLD = 40
NAMES = 40


def _frame(fold_slopes: list[float], *, seed: int = 5) -> pd.DataFrame:
    """A prediction frame with a per-fold relationship between score and outcome."""
    rng = np.random.default_rng(seed)
    rows = []
    week = dt.date(2018, 1, 5)
    for fold, slope in enumerate(fold_slopes):
        for _ in range(WEEKS_PER_FOLD):
            scores = rng.normal(0, 1, NAMES)
            rows.append(
                pd.DataFrame(
                    {
                        DECISION_SESSION: week,
                        "prediction": scores,
                        LABEL: scores * slope + rng.normal(0, 0.002, NAMES),
                        "fold": fold,
                    }
                )
            )
            week += dt.timedelta(days=7)
    return pd.concat(rows, ignore_index=True)


def test_the_first_fold_is_uncalibrated_by_construction():
    """Nothing precedes fold 0, so it gets NaN — never a passthrough of the raw score.

    NaN is the honest answer and it is also the safe one: sizing treats it as "no ceiling
    available" and says so, whereas a score handed back unchanged looks exactly like a
    return and is how the original Kelly bug survived review.
    """
    out = attach_expected_returns(_frame([0.02] * N_FOLDS))
    first = out.loc[out["fold"] == 0, "expected_return"]
    assert first.isna().all(), "fold 0 was calibrated on something, which cannot exist"

    later = out.loc[out["fold"] > 0, "expected_return"]
    assert later.notna().all(), "later folds should be calibrated"


def test_a_fold_is_calibrated_only_on_strictly_earlier_folds():
    """Plant an inverted relationship inside the last fold; its calibration must miss it.

    Folds 0 and 1 have a positive score-to-return relationship. Fold 2 has the opposite.
    A calibrator fitted on fold 2's own rows would learn the inversion and map its best
    scores to negative expected returns. Fitted correctly — on folds 0 and 1 only — it
    still maps them positive, and is simply wrong about fold 2, which is precisely what an
    out-of-sample estimate is entitled to be.
    """
    frame = _frame([0.02, 0.02, -0.02])
    out = attach_expected_returns(frame)

    last = out.loc[out["fold"] == 2]
    best = last.nlargest(200, "prediction")["expected_return"].mean()
    worst = last.nsmallest(200, "prediction")["expected_return"].mean()

    assert best > worst, (
        f"the highest scores in the final fold were calibrated to {best:+.5f} against "
        f"{worst:+.5f} for the lowest — the map learned that fold's own inverted "
        "outcome, which means it was fitted on data it must not have seen"
    )


def test_calibrator_training_window_ends_before_the_fold_it_serves():
    """The audit trail has to agree with the arithmetic."""
    frame = _frame([0.02] * N_FOLDS)
    calibrators = calibrators_by_fold(frame)

    for fold in (1, 2):
        fold_start = frame.loc[frame["fold"] == fold, DECISION_SESSION].min()
        assert calibrators[fold].fitted_through < fold_start, (
            f"fold {fold} starts {fold_start} but its calibrator claims to be fitted "
            f"through {calibrators[fold].fitted_through}"
        )
    assert not calibrators[0].is_fitted


def test_calibration_is_monotone_and_lands_in_return_units():
    """A higher score may never imply a lower expected return, and the output is a return.

    The second half is the whole point. Scores here span roughly [-3, 3]; a return over a
    one-week hold does not. If the calibrated output still spans the score's range, no map
    was applied.

    Monotonicity is asserted *within* a fold, not across the pooled frame. Each fold has
    its own map, fitted on a different amount of history, so two folds can legitimately
    disagree about what a given score is worth. Pooling them and demanding one monotone
    sequence would be asserting that the calibration never learns anything new.
    """
    out = attach_expected_returns(_frame([0.02] * N_FOLDS))
    calibrated = out.dropna(subset=["expected_return"])

    for fold, block in calibrated.groupby("fold"):
        values = block.sort_values("prediction")["expected_return"].to_numpy()
        assert np.all(np.diff(values) >= -1e-12), f"fold {fold} output is not monotone"

    assert calibrated["expected_return"].abs().max() < 0.5, (
        "calibrated values are still on the score's scale, not a return's"
    )
    assert calibrated["prediction"].abs().max() > 1.0, "fixture scores are too small to test this"


def test_an_unfitted_calibrator_returns_nan_rather_than_the_score():
    scores = pd.Series([0.3, -0.2, 0.1])
    out = ScoreCalibrator().transform(scores)
    assert out.isna().all()
    assert ScoreCalibrator().describe() == "uncalibrated"


def test_a_thin_sample_declines_to_fit():
    """Below the observation floor an isotonic fit mostly describes its own noise."""
    rng = np.random.default_rng(1)
    n = MIN_OBSERVATIONS - 1
    scores = pd.Series(rng.normal(0, 1, n))
    calibrator = ScoreCalibrator().fit(scores, scores * 0.02)
    assert not calibrator.is_fitted


def test_calibration_report_is_diagonal_when_the_map_holds():
    """Predicted versus realised, bucketed. A working map sits close to the diagonal."""
    out = attach_expected_returns(_frame([0.02] * N_FOLDS))
    report = calibration_report(out)
    assert not report.empty

    assert report["predicted"].is_monotonic_increasing
    # The relationship is stationary across folds here, so the map should be accurate,
    # not merely ordered. A systematic gap is the signature of a fit on its own output.
    assert report["error"].abs().max() < 0.01, report


@pytest.mark.parametrize("column", [DECISION_SESSION, LABEL, "prediction"])
def test_attach_is_a_no_op_on_an_empty_frame(column):
    empty = pd.DataFrame(columns=[column, "fold"])
    assert attach_expected_returns(empty).empty

"""Turning a score into an expected return.

Every model in this package emits a *score*: a cross-sectional rank in roughly
``[-0.5, 0.5]`` that says which names look better than which. That is exactly what a
ranking model should produce, and it is enough for "buy the top eight". It is not enough
for any decision phrased in money, because a score of 0.3 is not a 30 percent expected
return — it is not a return at all.

That distinction was not academic here. ``kelly_cap`` computes ``mu / sigma^2`` and was
being handed the raw score as ``mu``. With a score of 0.3 against 35 percent volatility
the implied ceiling came out around 0.6, against typical weights near 0.07, so the
``min()`` never bound. Kelly sizing was offered in the config, documented as a risk
control, and could not fire.

This module fits the missing map: an isotonic regression from score to realised excess
return. Isotonic rather than linear because the only thing worth assuming is that a
higher score should not imply a lower expected return; the shape between is data's
business, and it is usually flatter in the tails than a straight line would claim.

**The point-in-time discipline is the whole design.** A calibrator fitted on the same
predictions it will scale is fitted on the answer, and its output would look
extraordinarily well calibrated in a backtest and fall apart live. So the map is fitted
on the *previous* walk-forward fold's test predictions and applied to the current one.
Those predictions were genuinely out of sample when they were produced, the fit reads only
realised past outcomes, and no nested inner split is needed. The first fold has no prior
fold and therefore runs uncalibrated, which it reports rather than hides.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from ..types import DECISION_SESSION, LABEL

log = logging.getLogger(__name__)

#: Below this many observations an isotonic fit is mostly describing its own noise.
MIN_OBSERVATIONS = 400


@dataclass(slots=True)
class ScoreCalibrator:
    """Monotone map from model score to expected excess return over the hold window."""

    #: Fitted on data up to and including this decision date. Audit trail, never a feature.
    fitted_through: object = None
    n_observations: int = 0
    _model: IsotonicRegression | None = field(default=None, repr=False)
    #: Realised spread of the calibrated values, for sanity reporting.
    output_range: tuple[float, float] = (0.0, 0.0)

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    def fit(self, scores: pd.Series, realised: pd.Series) -> ScoreCalibrator:
        """Fit the map on scores whose outcomes are already known."""
        frame = pd.DataFrame({"score": scores, "y": realised}).replace(
            [np.inf, -np.inf], np.nan
        ).dropna()

        if len(frame) < MIN_OBSERVATIONS or frame["score"].nunique() < 10:
            log.info(
                "calibration skipped: %d usable observation(s), %d distinct score(s)",
                len(frame), frame["score"].nunique() if len(frame) else 0,
            )
            return self

        model = IsotonicRegression(out_of_bounds="clip", increasing=True)
        model.fit(frame["score"].to_numpy(), frame["y"].to_numpy())

        self._model = model
        self.n_observations = len(frame)
        predicted = model.predict(frame["score"].to_numpy())
        self.output_range = (float(np.min(predicted)), float(np.max(predicted)))
        return self

    def transform(self, scores: pd.Series) -> pd.Series:
        """Map scores to expected excess returns, or return NaN when unfitted.

        NaN rather than a passthrough of the raw score: a caller that needs a return must
        be able to tell that it did not get one. Returning the score unchanged is how the
        original bug survived — it looked like a number in the right place.
        """
        if self._model is None:
            return pd.Series(np.nan, index=scores.index, name="expected_return")
        values = self._model.predict(scores.to_numpy(dtype=float))
        return pd.Series(values, index=scores.index, name="expected_return")

    def describe(self) -> str:
        if not self.is_fitted:
            return "uncalibrated"
        low, high = self.output_range
        return (
            f"calibrated on {self.n_observations} obs through {self.fitted_through}, "
            f"expected return spans {low:+.4f} to {high:+.4f}"
        )


def calibrators_by_fold(
    predictions: pd.DataFrame,
    *,
    prediction_column: str = "prediction",
    label_column: str = LABEL,
    fold_column: str = "fold",
) -> dict[int, ScoreCalibrator]:
    """One calibrator per fold, each fitted only on strictly earlier folds.

    Fold 0 gets an unfitted calibrator because nothing precedes it. Fold *k* is fitted on
    the pooled out-of-sample predictions of folds 0 through *k-1*, so the map applied to
    a week's scores was built entirely from outcomes that had already occurred.
    """
    out: dict[int, ScoreCalibrator] = {}
    if predictions.empty or fold_column not in predictions.columns:
        return out

    folds = sorted(predictions[fold_column].unique())
    for position, fold in enumerate(folds):
        calibrator = ScoreCalibrator()
        if position > 0:
            earlier = predictions.loc[predictions[fold_column].isin(folds[:position])]
            calibrator.fitted_through = earlier[DECISION_SESSION].max()
            calibrator.fit(earlier[prediction_column], earlier[label_column])
        out[fold] = calibrator
    return out


def attach_expected_returns(
    predictions: pd.DataFrame,
    *,
    prediction_column: str = "prediction",
    label_column: str = LABEL,
    fold_column: str = "fold",
    output_column: str = "expected_return",
) -> pd.DataFrame:
    """Add a point-in-time calibrated expected return column to a prediction frame.

    Rows in the first fold carry NaN, which downstream sizing must treat as "no expected
    return available" rather than as zero.
    """
    if predictions.empty:
        return predictions

    out = predictions.copy()
    out[output_column] = np.nan

    calibrators = calibrators_by_fold(
        out,
        prediction_column=prediction_column,
        label_column=label_column,
        fold_column=fold_column,
    )
    if not calibrators:
        return out

    for fold, calibrator in calibrators.items():
        mask = out[fold_column] == fold
        if not calibrator.is_fitted or not bool(mask.any()):
            continue
        out.loc[mask, output_column] = calibrator.transform(
            out.loc[mask, prediction_column]
        ).to_numpy()

    covered = float(out[output_column].notna().mean())
    log.info(
        "calibration covers %.0f%% of out-of-sample rows (%d fold(s), first fold "
        "uncalibrated by construction)",
        covered * 100, len(calibrators),
    )
    return out


def calibration_report(predictions: pd.DataFrame, *, n_buckets: int = 10) -> pd.DataFrame:
    """Predicted versus realised return by bucket. Straight to the tearsheet.

    A calibrator that is working produces a diagonal: the names it says should return 2
    percent do return about 2 percent on average. Systematic overshoot is the signature of
    a model that has been fitted on its own predictions.
    """
    if predictions.empty or "expected_return" not in predictions.columns:
        return pd.DataFrame()

    frame = predictions[["expected_return", LABEL]].dropna()
    if len(frame) < n_buckets * 10:
        return pd.DataFrame()

    try:
        buckets = pd.qcut(
            frame["expected_return"], n_buckets, labels=False, duplicates="drop"
        )
    except ValueError:
        return pd.DataFrame()

    grouped = frame.groupby(buckets).agg(
        n=("expected_return", "size"),
        predicted=("expected_return", "mean"),
        realised=(LABEL, "mean"),
    )
    grouped["error"] = grouped["predicted"] - grouped["realised"]
    return grouped.reset_index(drop=True)

"""Gradient-boosted trees, the workhorse forecaster.

Two things are deliberately conservative here, and both are because of how little data a
weekly strategy actually has. Ten years is about 520 decision dates, and the names within
a date are correlated, so the effective sample is far smaller than the row count
suggests. A deep, heavily-tuned booster will fit that noise perfectly.

So the defaults are shallow (15 leaves), heavily regularised, and slow-learning, and
there is no automatic hyperparameter search over the full sample. Tuning, when it
happens, belongs inside a training fold.

LightGBM is used when available and scikit-learn's ``HistGradientBoostingRegressor``
otherwise, so the system has no hard dependency on it. Both handle NaN natively, which is
why no imputation happens here: a missing feature is information, and filling it with a
median throws that away.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

try:  # pragma: no cover - import guard
    import lightgbm as lgb

    HAS_LIGHTGBM = True
except ImportError:  # pragma: no cover
    HAS_LIGHTGBM = False


class GBDTModel:
    """Gradient boosting on cross-sectionally ranked features."""

    name = "gbdt"

    def __init__(
        self,
        *,
        n_estimators: int = 300,
        learning_rate: float = 0.03,
        num_leaves: int = 15,
        min_child_samples: int = 80,
        subsample: float = 0.8,
        colsample_bytree: float = 0.7,
        reg_lambda: float = 5.0,
        seed: int = 7,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.seed = seed
        self.feature_names: list[str] = []
        self._model = None
        self._backend = "lightgbm" if HAS_LIGHTGBM else "sklearn"

    @property
    def backend(self) -> str:
        return self._backend

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        sample_weight: pd.Series | None = None,
        week_index: pd.Series | None = None,  # noqa: ARG002
    ) -> GBDTModel:
        self.feature_names = list(X.columns)
        mask = y.notna()
        X_fit, y_fit = X.loc[mask], y.loc[mask]
        weights = sample_weight.loc[mask].to_numpy(dtype=float) if sample_weight is not None else None

        if HAS_LIGHTGBM:
            self._model = lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                min_child_samples=self.min_child_samples,
                subsample=self.subsample,
                subsample_freq=1,
                colsample_bytree=self.colsample_bytree,
                reg_lambda=self.reg_lambda,
                random_state=self.seed,
                n_jobs=-1,
                verbose=-1,
            )
            self._model.fit(X_fit, y_fit, sample_weight=weights)
        else:
            from sklearn.ensemble import HistGradientBoostingRegressor

            self._model = HistGradientBoostingRegressor(
                max_iter=self.n_estimators,
                learning_rate=self.learning_rate,
                max_leaf_nodes=self.num_leaves,
                min_samples_leaf=self.min_child_samples,
                l2_regularization=self.reg_lambda,
                random_state=self.seed,
            )
            self._model.fit(X_fit, y_fit, sample_weight=weights)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self._model is None:
            raise RuntimeError("GBDTModel.predict called before fit")
        aligned = X.reindex(columns=self.feature_names)
        values = self._model.predict(aligned)
        return pd.Series(values, index=X.index, name="prediction")

    def feature_importance(self) -> pd.Series:
        if self._model is None:
            return pd.Series(dtype=float)
        if HAS_LIGHTGBM and hasattr(self._model, "feature_importances_"):
            values = np.asarray(self._model.feature_importances_, dtype=float)
        else:
            # HistGradientBoosting exposes no native importance, so permutation
            # importance would be needed; return an empty series rather than a fake one.
            return pd.Series(dtype=float)
        total = values.sum()
        normalised = values / total if total > 0 else values
        return pd.Series(
            normalised, index=self.feature_names, name="importance"
        ).sort_values(ascending=False)

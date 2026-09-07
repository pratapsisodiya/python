"""Ridge regression on cross-sectionally ranked features.

The interpretable half of the model set. Its coefficients are directly readable as "this
feature gets this much weight", which is worth a great deal when a gradient booster
produces a number nobody can explain and you have to decide whether to trust it with
money.

It also serves as a sanity check. If ridge and the booster disagree wildly about which
features matter, the booster is probably fitting noise, because at a weekly horizon with
a few hundred weeks of data there is not enough signal to support a genuinely complex
function.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


class RidgeModel:
    """Ridge on pre-ranked features, with explicit missing handling."""

    name = "ridge"

    def __init__(self, *, alpha: float = 10.0, fit_intercept: bool = False) -> None:
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.feature_names: list[str] = []
        self._model: Ridge | None = None
        self._fill: pd.Series | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        sample_weight: pd.Series | None = None,
        week_index: pd.Series | None = None,  # noqa: ARG002
    ) -> RidgeModel:
        self.feature_names = list(X.columns)

        # Impute with the TRAINING median only. Computing a fill value from the full
        # panel would leak the future distribution into every historical row, which is a
        # quiet leak that no amount of careful splitting elsewhere would catch.
        self._fill = X.median(numeric_only=True)
        filled = X.fillna(self._fill).fillna(0.0)

        mask = y.notna()
        if sample_weight is not None:
            weights = sample_weight.loc[mask].to_numpy(dtype=float)
        else:
            weights = None

        self._model = Ridge(alpha=self.alpha, fit_intercept=self.fit_intercept)
        self._model.fit(filled.loc[mask], y.loc[mask], sample_weight=weights)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self._model is None:
            raise RuntimeError("RidgeModel.predict called before fit")
        aligned = X.reindex(columns=self.feature_names)
        filled = aligned.fillna(self._fill).fillna(0.0)
        return pd.Series(self._model.predict(filled), index=X.index, name="prediction")

    def feature_importance(self) -> pd.Series:
        if self._model is None:
            return pd.Series(dtype=float)
        return pd.Series(
            np.abs(self._model.coef_), index=self.feature_names, name="importance"
        ).sort_values(ascending=False)

    def coefficients(self) -> pd.Series:
        """Signed coefficients, which is what makes this model worth keeping."""
        if self._model is None:
            return pd.Series(dtype=float)
        return pd.Series(self._model.coef_, index=self.feature_names, name="coefficient")

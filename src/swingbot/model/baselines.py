"""Null models and baselines.

These are the most important models in the package, and it is worth being blunt about
why. A strategy's absolute Sharpe ratio is nearly meaningless on its own. What matters is
whether it beats the thing you would have got for free, and there are several free
things:

* **ZeroModel** — hold nothing. Establishes the cost floor.
* **RandomSignModel** — random positions at matched turnover and volatility target. Some
  Sharpe comes from rebalancing and volatility targeting alone, independent of any
  forecast. This measures how much.
* **ShuffledLabelModel** — the real model, trained on labels permuted *within* each week.
  This must produce an IC near zero. If it does not, the pipeline is leaking and every
  other number in the report is void. It is the single most valuable diagnostic here.
* **MomentumOnlyModel** — the known, published, free factor. A model that cannot beat
  short-horizon reversal has not earned its complexity.

The point of shuffling within a week rather than across the whole panel is subtle but
matters: shuffling globally would also destroy the time-series structure, so a
near-zero result would not isolate cross-sectional leakage. Shuffling within a week
keeps everything except the name-to-outcome mapping.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class ZeroModel:
    """Predicts nothing. The cost floor."""

    name = "zero"

    def __init__(self) -> None:
        self.feature_names: list[str] = []

    def fit(self, X, y, *, sample_weight=None, week_index=None):  # noqa: ARG002
        self.feature_names = list(X.columns)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(0.0, index=X.index, name="prediction")

    def feature_importance(self) -> pd.Series:
        return pd.Series(dtype=float)


class RandomSignModel:
    """Seeded random scores. Measures the Sharpe available from rebalancing alone."""

    name = "random"

    def __init__(self, *, seed: int = 7) -> None:
        self.seed = seed
        self.feature_names: list[str] = []

    def fit(self, X, y, *, sample_weight=None, week_index=None):  # noqa: ARG002
        self.feature_names = list(X.columns)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        # Seeded off the row count so the same panel always yields the same draw, which
        # keeps the baseline reproducible across runs.
        rng = np.random.default_rng(self.seed + len(X))
        return pd.Series(rng.normal(0.0, 1.0, len(X)), index=X.index, name="prediction")

    def feature_importance(self) -> pd.Series:
        return pd.Series(dtype=float)


class ShuffledLabelModel:
    """Wraps a real model and trains it on labels shuffled within each week.

    The leak detector. Everything about the pipeline is identical except that the
    name-to-outcome mapping has been destroyed, so any remaining predictive power is
    leakage by definition.
    """

    name = "shuffled"

    def __init__(self, inner, *, seed: int = 7) -> None:
        self.inner = inner
        self.seed = seed
        self.feature_names: list[str] = []

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        sample_weight: pd.Series | None = None,
        week_index: pd.Series | None = None,
    ):
        self.feature_names = list(X.columns)

        # The seed is derived per training set, not fixed across the whole run, and this
        # matters more than it looks.
        #
        # A random permutation leaves a residual correlation with the original of order
        # 1/sqrt(n) — about 0.01 on a panel of this size. That is harmless in itself, but
        # walk-forward folds are expanding and therefore overlap heavily, so reusing one
        # seed applies a correlated permutation to nested training sets. Every fold's
        # residual then leans the same way, and a canary that averages across folds sees
        # a consistent bias instead of independent noise. On this codebase that produced
        # a shuffled-label IC of +0.021 at t = 3.0 across 14 folds — a false leak warning
        # generated entirely by the canary's own seeding.
        #
        # Mixing the training set's shape and span into the seed gives each fold an
        # independent permutation while keeping the whole thing reproducible.
        fingerprint = (len(X), int(X.shape[1]), hash(tuple(y.index[:8])) & 0xFFFF)
        rng = np.random.default_rng([self.seed, *fingerprint])

        shuffled = y.copy()
        if week_index is not None:
            for _, positions in y.groupby(week_index, sort=False).groups.items():
                # copy() because pandas hands back a read-only view.
                values = y.loc[positions].to_numpy().copy()
                rng.shuffle(values)
                shuffled.loc[positions] = values
        else:
            values = y.to_numpy().copy()
            rng.shuffle(values)
            shuffled = pd.Series(values, index=y.index, name=y.name)

        self.inner.fit(X, shuffled, sample_weight=sample_weight, week_index=week_index)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return self.inner.predict(X)

    def feature_importance(self) -> pd.Series:
        return self.inner.feature_importance()


class MomentumOnlyModel:
    """A fixed linear combination of published factors. No fitting.

    Short-horizon reversal plus intermediate momentum, both of which are well documented
    in US and Indian equities. Sign convention: ``reversal_1w`` is already negated in the
    feature block, so a positive weight means betting on reversal.
    """

    name = "momentum"

    def __init__(
        self,
        *,
        weights: dict[str, float] | None = None,
    ) -> None:
        self.weights = weights or {
            "reversal_1w": 0.5,
            "mom_12w": 0.3,
            "mom_52w_skip4w": 0.2,
        }
        self.feature_names: list[str] = []

    def fit(self, X, y, *, sample_weight=None, week_index=None):  # noqa: ARG002
        self.feature_names = [c for c in self.weights if c in X.columns]
        if not self.feature_names:
            raise ValueError(
                f"MomentumOnlyModel needs at least one of {sorted(self.weights)}; "
                f"panel has {list(X.columns)[:10]}..."
            )
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        score = pd.Series(0.0, index=X.index)
        for column in self.feature_names:
            score = score + self.weights[column] * X[column].fillna(0.0)
        return score.rename("prediction")

    def feature_importance(self) -> pd.Series:
        return pd.Series(
            {k: abs(v) for k, v in self.weights.items() if k in self.feature_names},
            name="importance",
        ).sort_values(ascending=False)


class SubsetModel:
    """Restricts an inner model to a subset of columns.

    How the ablation builds price-only and news-only variants: same model class, same
    hyperparameters, same folds, different feature slice. Holding everything else fixed
    is what makes the resulting Sharpe difference attributable to the features rather
    than to an incidental difference in configuration.
    """

    def __init__(self, inner, columns: list[str], *, name: str = "subset") -> None:
        self.inner = inner
        self.columns = list(columns)
        self.name = name
        self.feature_names: list[str] = []

    def _slice(self, X: pd.DataFrame) -> pd.DataFrame:
        present = [c for c in self.columns if c in X.columns]
        if not present:
            raise ValueError(f"{self.name}: none of its columns are in the panel")
        return X[present]

    def fit(self, X, y, *, sample_weight=None, week_index=None):
        sliced = self._slice(X)
        self.feature_names = list(sliced.columns)
        self.inner.fit(sliced, y, sample_weight=sample_weight, week_index=week_index)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return self.inner.predict(self._slice(X))

    def feature_importance(self) -> pd.Series:
        return self.inner.feature_importance()

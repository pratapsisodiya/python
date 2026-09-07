"""Model protocol.

One method pair, deliberately. ``fit`` takes sample weights and a week index because both
are needed to train honestly on overlapping labels: the weights downrank redundant
samples, and the week index lets a model do grouped operations without guessing which
rows belong together.

``predict`` returns an expected **cross-sectional excess return**, comparable within a
week. Not a price, not a probability, not a score on an arbitrary scale. Every consumer
downstream — the blender, the portfolio constructor, the IC calculation — relies on that
contract.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class ReturnModel(Protocol):
    name: str
    feature_names: list[str]

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        sample_weight: pd.Series | None = None,
        week_index: pd.Series | None = None,
    ) -> ReturnModel: ...

    def predict(self, X: pd.DataFrame) -> pd.Series: ...

    def feature_importance(self) -> pd.Series:
        """Per-feature importance, empty when the model does not expose one."""
        ...

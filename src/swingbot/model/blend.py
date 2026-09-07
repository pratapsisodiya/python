"""The two-block blender: price history versus news.

This module is where the user's instruction that roughly ninety percent of the forecast
should come from price history and chart structure, and ten percent from news, becomes an
explicit, auditable number rather than a hope.

The alternative would be to hand every feature to one model and trust it to weight them
sensibly. That fails for two reasons. A gradient booster given a small number of news
columns and sixty price columns will distribute importance in a way nobody can read off
as a ratio, so the 90/10 split becomes unverifiable. And news features are missing for
most name-weeks, so a single model learns "news present" as a proxy for something else
entirely.

Training two models on two feature blocks and combining their **ranks** fixes both. The
weights are explicit, the contribution of each block is measurable, and a name with no
news simply falls back to its price score rather than being penalised.

Ranks rather than raw predictions because the two models have no reason to share a scale.
Averaging a booster's output with a ridge's output directly would let whichever happens to
have larger variance dominate, regardless of the configured weights.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..types import DECISION_SESSION


class BlendedModel:
    """Combines a price model and a news model at configured weights."""

    name = "blend"

    def __init__(
        self,
        price_model,
        news_model=None,
        *,
        price_columns: list[str] | None = None,
        news_columns: list[str] | None = None,
        price_weight: float = 0.90,
        news_weight: float = 0.10,
    ) -> None:
        self.price_model = price_model
        self.news_model = news_model
        self.price_columns = price_columns or []
        self.news_columns = news_columns or []
        total = price_weight + news_weight
        self.price_weight = price_weight / total if total > 0 else 1.0
        self.news_weight = news_weight / total if total > 0 else 0.0
        self.feature_names: list[str] = []
        self._news_active = False

    # ------------------------------------------------------------------------- fit

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        sample_weight: pd.Series | None = None,
        week_index: pd.Series | None = None,
    ) -> BlendedModel:
        price_cols = [c for c in self.price_columns if c in X.columns] or list(X.columns)
        self.price_model.fit(
            X[price_cols], y, sample_weight=sample_weight, week_index=week_index
        )
        self.feature_names = list(price_cols)

        news_cols = [c for c in self.news_columns if c in X.columns]
        # The news block trains only on rows that actually carry news. Training it on
        # rows where every news feature is null would teach it that null means neutral,
        # which is not the same claim as "no information".
        if self.news_model is not None and news_cols and self.news_weight > 0:
            has_news = X[news_cols].notna().any(axis=1) & y.notna()
            if int(has_news.sum()) >= 200:
                self.news_model.fit(
                    X.loc[has_news, news_cols],
                    y.loc[has_news],
                    sample_weight=(
                        sample_weight.loc[has_news] if sample_weight is not None else None
                    ),
                    week_index=(week_index.loc[has_news] if week_index is not None else None),
                )
                self.feature_names = list(price_cols) + list(news_cols)
                self._news_active = True
        return self

    # --------------------------------------------------------------------- predict

    def predict(self, X: pd.DataFrame) -> pd.Series:
        price_cols = [c for c in self.price_columns if c in X.columns] or list(X.columns)
        price_score = self.price_model.predict(X[price_cols])
        price_rank = _rank_within(price_score, X)

        if not self._news_active or self.news_weight <= 0:
            return price_rank.rename("prediction")

        news_cols = [c for c in self.news_columns if c in X.columns]
        has_news = X[news_cols].notna().any(axis=1)
        news_rank = pd.Series(0.0, index=X.index)
        if bool(has_news.any()):
            raw = self.news_model.predict(X.loc[has_news, news_cols])
            news_rank.loc[has_news] = _rank_within(raw, X.loc[has_news])

        # Rows without news contribute nothing from the news block and keep their full
        # price weight, rather than being dragged toward the middle of the book.
        news_component = np.where(has_news, self.news_weight * news_rank, 0.0)
        price_component = np.where(
            has_news, self.price_weight * price_rank, price_rank
        )
        return pd.Series(
            price_component + news_component, index=X.index, name="prediction"
        )

    # ------------------------------------------------------------------ diagnostics

    def feature_importance(self) -> pd.Series:
        price = self.price_model.feature_importance()
        if not self._news_active:
            return price
        news = self.news_model.feature_importance()
        combined = pd.concat(
            [price * self.price_weight, news * self.news_weight]
        )
        return combined.sort_values(ascending=False)

    def block_contribution(self, X: pd.DataFrame) -> dict[str, float]:
        """Realised share of score dispersion contributed by each block.

        Reported in the tearsheet so the configured 90/10 can be checked against what the
        blend actually did rather than taken on trust.

        Measured on the standard-deviation scale, not variance. Variance share is
        quadratic in the weight, so a configured 10 percent news weight would report as
        about 1 percent and read as though the news block were doing nothing. The
        standard-deviation share is linear in the weight, so it lands near the configured
        split when both blocks have similar dispersion, and departs from it only when the
        news block genuinely spreads names less, or covers fewer of them, than the price
        block. That is the deviation worth seeing.
        """
        if not self._news_active:
            return {"price": 1.0, "news": 0.0, "news_coverage": 0.0}

        price_cols = [c for c in self.price_columns if c in X.columns] or list(X.columns)
        news_cols = [c for c in self.news_columns if c in X.columns]
        has_news = X[news_cols].notna().any(axis=1)
        coverage = float(has_news.mean())

        price_sd = float(np.nanstd(_rank_within(self.price_model.predict(X[price_cols]), X)))
        news_sd = 0.0
        if bool(has_news.any()):
            raw = self.news_model.predict(X.loc[has_news, news_cols])
            news_sd = float(np.nanstd(_rank_within(raw, X.loc[has_news])))

        price_part = self.price_weight * price_sd
        news_part = self.news_weight * news_sd * coverage
        total = price_part + news_part
        if total <= 0:
            return {"price": 1.0, "news": 0.0, "news_coverage": coverage}
        return {
            "price": price_part / total,
            "news": news_part / total,
            "news_coverage": coverage,
            "configured_price_weight": self.price_weight,
            "configured_news_weight": self.news_weight,
        }


def _rank_within(scores: pd.Series, panel: pd.DataFrame) -> pd.Series:
    """Rank scores within each decision date, mapped to [-0.5, 0.5].

    Falls back to a global rank when the panel carries no date column, which only
    happens in unit tests that pass a bare feature matrix.
    """
    if DECISION_SESSION in panel.columns:
        grouped = scores.groupby(panel[DECISION_SESSION], sort=False)
        return grouped.rank(pct=True, method="average") - 0.5
    return scores.rank(pct=True, method="average") - 0.5

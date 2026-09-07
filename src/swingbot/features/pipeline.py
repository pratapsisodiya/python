"""Feature pipeline: compose transformers, enforce point-in-time, assemble the panel.

The pipeline is deliberately thin. It does three things and nothing else:

1. Runs each registered block over the bar panel.
2. Joins the blocks on ``(ticker, session)``, which cannot silently misalign because
   every block returns those keys.
3. Restricts to decision dates, joins membership and labels, and normalises
   cross-sectionally.

The ordering matters. Raw feature computation happens on the **full daily panel**, because
a rolling window needs its history. Cross-sectional normalisation happens **after**
restricting to decision dates, because ranking within a date is only meaningful across the
names actually scored on that date. Doing it the other way round would rank each name
against every session in history, which is both wrong and a leak.
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from ..config import Config
from ..pit import PITGuard, ensure_utc, truncate_sessions
from ..types import (
    DECISION_SESSION,
    LABEL,
    NEWS_FEATURE_PREFIX,
    SECTOR,
    SESSION,
    TICKER,
)
from .base import REGISTRY, BaseTransformer, ensure_sorted
from .cross_sectional import (
    cross_sectional_rank,
    drop_thin_dates,
    sector_neutralize,
)

log = logging.getLogger(__name__)

#: Blocks that make up the price side of the 90/10 blend.
PRICE_BLOCKS = ("price", "moments", "technical", "patterns", "regime")


class FeaturePipeline:
    """Builds the feature panel from bars."""

    def __init__(
        self,
        transformers: list[BaseTransformer],
        *,
        min_names_per_week: int = 20,
        sector_neutralize_features: bool = False,
    ) -> None:
        self.transformers = transformers
        self.min_names_per_week = min_names_per_week
        self.sector_neutralize_features = sector_neutralize_features
        self._fitted = False

    # ---------------------------------------------------------------- construction

    @classmethod
    def from_config(cls, cfg: Config, *, blocks: list[str] | None = None) -> FeaturePipeline:
        names = blocks if blocks is not None else list(PRICE_BLOCKS)
        return cls(
            REGISTRY.build_all(names),
            min_names_per_week=cfg.features.min_names_per_week,
            sector_neutralize_features=cfg.features.sector_neutralize,
        )

    @property
    def warmup_sessions(self) -> int:
        return max((t.warmup_sessions for t in self.transformers), default=0)

    @property
    def feature_names(self) -> list[str]:
        out: list[str] = []
        for transformer in self.transformers:
            out.extend(transformer.outputs)
        return out

    # --------------------------------------------------------------------- fitting

    def fit(self, bars: pd.DataFrame, train_mask: pd.Series | None = None) -> FeaturePipeline:
        for transformer in self.transformers:
            transformer.fit(bars, train_mask)
        self._fitted = True
        return self

    # -------------------------------------------------------------------- raw pass

    def transform_raw(
        self, bars: pd.DataFrame, *, asof: date | None = None
    ) -> pd.DataFrame:
        """Run every block over the daily panel and join the results.

        ``asof`` truncates the panel *physically* before any block sees it, which is a
        stronger guarantee than filtering afterwards: a leaky transformer cannot read
        rows that are not in the frame.
        """
        if bars.empty:
            return pd.DataFrame()

        panel = ensure_sorted(bars)
        if asof is not None:
            panel = truncate_sessions(panel, asof)
            if panel.empty:
                return pd.DataFrame()

        guard_ts = (
            ensure_utc(pd.Timestamp(asof, tz="UTC") + pd.Timedelta(days=1))
            if asof is not None
            else None
        )

        merged: pd.DataFrame | None = None
        seen: set[str] = set()

        for transformer in self.transformers:
            if guard_ts is not None:
                with PITGuard(guard_ts, context=f"{transformer.name} features") as guard:
                    guard.check(panel)
                    block = transformer.transform(panel)
                    guard.check_output(block)
            else:
                block = transformer.transform(panel)

            collisions = (set(block.columns) - {TICKER, SESSION}) & seen
            if collisions:
                raise ValueError(
                    f"Transformer {transformer.name!r} re-declares existing feature(s): "
                    f"{sorted(collisions)}"
                )
            seen |= set(block.columns) - {TICKER, SESSION}

            merged = (
                block
                if merged is None
                else merged.merge(block, on=[TICKER, SESSION], how="outer", validate="one_to_one")
            )

        return merged if merged is not None else pd.DataFrame()

    # ------------------------------------------------------------------ full build

    def build(
        self,
        bars: pd.DataFrame,
        *,
        decision_sessions: list[date],
        membership: pd.DataFrame | None = None,
        labels: pd.DataFrame | None = None,
        news_features: pd.DataFrame | None = None,
        asof: date | None = None,
        normalize: bool = True,
    ) -> pd.DataFrame:
        """Assemble the modelling panel, one row per (decision date, ticker)."""
        raw = self.transform_raw(bars, asof=asof)
        if raw.empty:
            return pd.DataFrame()

        wanted = set(decision_sessions)
        panel = raw.loc[raw[SESSION].isin(wanted)].copy()
        if panel.empty:
            return pd.DataFrame()
        panel = panel.rename(columns={SESSION: DECISION_SESSION})

        if membership is not None and not membership.empty:
            members = membership.rename(columns={SESSION: DECISION_SESSION})
            panel = panel.merge(
                members[[DECISION_SESSION, TICKER, SECTOR]],
                on=[DECISION_SESSION, TICKER],
                how="inner",
            )
        elif SECTOR not in panel.columns:
            panel[SECTOR] = "Unknown"

        if news_features is not None and not news_features.empty:
            news = news_features.rename(columns={SESSION: DECISION_SESSION})
            panel = panel.merge(news, on=[DECISION_SESSION, TICKER], how="left")

        if labels is not None and not labels.empty:
            panel = panel.merge(labels, on=[DECISION_SESSION, TICKER], how="left")

        panel = drop_thin_dates(
            panel, date_column=DECISION_SESSION, min_names=self.min_names_per_week
        )
        if panel.empty:
            return panel

        if normalize:
            panel = self.normalize(panel)

        return panel.sort_values([DECISION_SESSION, TICKER]).reset_index(drop=True)

    def normalize(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Cross-sectional rank within each decision date.

        Causal by construction: a within-date transform reads only rows from that date,
        so it cannot leak across time however it is applied. This is the safe
        alternative to fitting a scaler on the full panel, which is the single most
        common leak in retail backtests.
        """
        columns = [c for c in feature_columns(panel) if c in panel.columns]
        if not columns:
            return panel

        out = cross_sectional_rank(
            panel,
            columns,
            date_column=DECISION_SESSION,
            min_names=self.min_names_per_week,
        )
        if self.sector_neutralize_features:
            out = sector_neutralize(out, columns, date_column=DECISION_SESSION)
        return out


# --------------------------------------------------------------------------------------
# Column helpers
# --------------------------------------------------------------------------------------

#: Columns that are keys, labels or bookkeeping, never model inputs.
NON_FEATURE_COLUMNS = frozenset(
    {
        TICKER,
        SESSION,
        DECISION_SESSION,
        SECTOR,
        LABEL,
        "entry_session",
        "exit_session",
        "label_t0",
        "label_t1",
        "sample_weight",
        "raw_return",
        "available_at",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "split_factor",
        "div_cash",
        "is_delisted",
        "adjustment_factor",
        "adv_notional",
        "liquid",
        "n_sessions",
        "week_end",
    }
)


def feature_columns(panel: pd.DataFrame) -> list[str]:
    """Every modelling feature in a panel, in a stable order."""
    return [
        c
        for c in panel.columns
        if c not in NON_FEATURE_COLUMNS and not str(c).startswith("_")
    ]


def price_feature_columns(panel: pd.DataFrame) -> list[str]:
    """The price block: everything that is not a news feature.

    Split by prefix rather than by a maintained list, so a new news feature lands on the
    correct side of the 90/10 blend automatically.
    """
    return [c for c in feature_columns(panel) if not c.startswith(NEWS_FEATURE_PREFIX)]


def news_feature_columns(panel: pd.DataFrame) -> list[str]:
    return [c for c in feature_columns(panel) if c.startswith(NEWS_FEATURE_PREFIX)]

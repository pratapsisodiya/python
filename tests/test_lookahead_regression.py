"""The look-ahead regression test. The one that makes everything else trustworthy.

For a sampled decision date, features are built twice: once from the full panel, and once
from a panel **physically truncated** after that date. Every value at the cutoff must be
bitwise identical.

Truncation rather than filtering is the point. Filtering after the fact would still let a
transformer read future rows while computing; deleting them means it cannot. Any of the
following fails this test, which is most of the ways real backtests leak:

* a scaler or winsoriser fitted on the whole panel
* a rank or z-score computed across all dates instead of within one
* a retroactively split-adjusted price
* a forward-fill or centred rolling window that crosses the cutoff
* a "days since" counter derived from a full-series extreme

It is parametrised over the transformer registry, so a new feature block is covered the
moment it is registered and a leaky one cannot be added unnoticed.
"""

from __future__ import annotations

import numpy as np
import pytest

from swingbot.features.base import REGISTRY
from swingbot.features.pipeline import FeaturePipeline

BLOCK_NAMES = REGISTRY.names()


def _values_at(frame, session, columns):
    block = frame.loc[frame["session"] == session].sort_values("ticker")
    return block[["ticker", *columns]].reset_index(drop=True)


def _assert_identical(full, truncated, columns, *, label: str) -> None:
    assert len(full) == len(truncated), (
        f"{label}: truncation changed the row count at the cutoff "
        f"({len(full)} vs {len(truncated)}) — a transformer is reading future rows"
    )
    assert full["ticker"].tolist() == truncated["ticker"].tolist()

    for column in columns:
        a = full[column].to_numpy(dtype=float)
        b = truncated[column].to_numpy(dtype=float)
        both_nan = np.isnan(a) & np.isnan(b)
        same = both_nan | (a == b)
        if not same.all():
            bad = int((~same).sum())
            worst = float(np.nanmax(np.abs(a - b)))
            raise AssertionError(
                f"{label}: feature {column!r} differs for {bad} name(s) when future data "
                f"is removed (max absolute difference {worst:.6g}). This is look-ahead: "
                f"the value at the cutoff depends on data after it."
            )


@pytest.mark.parametrize("block_name", BLOCK_NAMES)
def test_block_has_no_lookahead(block_name, alpha_bars, grid):
    """Each registered block individually, at three cutoffs across the sample."""
    pipeline = FeaturePipeline([REGISTRY.build(block_name)], min_names_per_week=5)
    columns = [c for c in pipeline.feature_names]

    sessions = [g.decision_session for g in grid]
    # Early, middle and late. The late one matters most: it has the most history behind
    # it and therefore the most opportunity for a long window to reach forward.
    cutoffs = [sessions[len(sessions) // 4], sessions[len(sessions) // 2], sessions[-3]]

    full = pipeline.transform_raw(alpha_bars)
    for cutoff in cutoffs:
        truncated = pipeline.transform_raw(alpha_bars, asof=cutoff)
        _assert_identical(
            _values_at(full, cutoff, columns),
            _values_at(truncated, cutoff, columns),
            columns,
            label=f"{block_name} @ {cutoff}",
        )


def test_composed_pipeline_has_no_lookahead(alpha_bars, grid, cfg_us):
    """The whole composed pipeline, which can leak even when each block is clean."""
    pipeline = FeaturePipeline.from_config(cfg_us)
    pipeline.min_names_per_week = 5
    columns = list(pipeline.feature_names)

    sessions = [g.decision_session for g in grid]
    cutoff = sessions[len(sessions) // 2]

    full = pipeline.transform_raw(alpha_bars)
    truncated = pipeline.transform_raw(alpha_bars, asof=cutoff)
    _assert_identical(
        _values_at(full, cutoff, columns),
        _values_at(truncated, cutoff, columns),
        columns,
        label=f"composed pipeline @ {cutoff}",
    )


def test_normalisation_is_within_date_only(alpha_panel):
    """Cross-sectional ranks must use only the rows of their own date.

    Verified by the ranks summing to (very nearly) zero within every date: a rank
    computed across the whole panel would not.
    """
    panel, _, _ = alpha_panel
    from swingbot.features.pipeline import feature_columns

    columns = feature_columns(panel)[:8]
    for column in columns:
        by_date = panel.groupby("decision_session")[column].mean().dropna()
        assert by_date.abs().max() < 0.05, (
            f"{column!r} has a non-zero within-date mean, so it was not ranked within "
            "its own date — normalisation is crossing dates"
        )


def test_truncation_actually_removes_rows(alpha_bars, grid):
    """Guard on the test itself: truncation must really delete future rows.

    Without this, a broken ``truncate_sessions`` would make every test above pass
    vacuously by comparing the full panel against itself.
    """
    from swingbot.pit import truncate_sessions

    cutoff = grid[len(grid) // 2].decision_session
    truncated = truncate_sessions(alpha_bars, cutoff)
    assert len(truncated) < len(alpha_bars)
    assert truncated["session"].max() <= cutoff
    assert alpha_bars["session"].max() > cutoff

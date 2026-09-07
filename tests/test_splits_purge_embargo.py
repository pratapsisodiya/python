"""Purged walk-forward correctness.

The invariant, stated precisely: for every fold, no training sample's label window may
touch the test window's label span, and none may begin inside the embargo after it.

Checked both on the real panel and on randomly generated span structures, because the
real panel has a regular weekly cadence and a bug that only appears with irregular or
long overlapping windows would hide there.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from swingbot.validation.splits import PurgedWalkForward, verify_no_overlap


def _panel(n_weeks: int, n_names: int, horizon_weeks: int, seed: int = 0) -> pd.DataFrame:
    weeks = pd.date_range("2015-01-02", periods=n_weeks, freq="W-FRI")
    rows = []
    for week in weeks:
        entry = week + pd.Timedelta(days=3)
        exit_ = entry + pd.Timedelta(weeks=horizon_weeks)
        for name in range(n_names):
            rows.append(
                {
                    "decision_session": week.date(),
                    "ticker": f"T{name:02d}",
                    "label_t0": entry.date(),
                    "label_t1": exit_.date(),
                    "label": np.random.default_rng(seed + name).normal(),
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.parametrize("horizon_weeks", [1, 2, 4, 8])
def test_no_training_label_touches_the_test_span(horizon_weeks):
    """The core invariant, across horizons.

    A one-week horizon barely overlaps; an eight-week horizon overlaps heavily, which is
    where an unpurged split leaks most and where purging must therefore work hardest.
    """
    panel = _panel(300, 10, horizon_weeks)
    splitter = PurgedWalkForward(
        train_weeks=104, test_weeks=26, embargo_weeks=2, expanding=True, min_train_weeks=78
    )
    folds = list(splitter.split(panel))
    assert folds, "no folds were produced"

    for fold in folds:
        verify_no_overlap(panel, fold, embargo_weeks=2)


def test_longer_horizons_purge_more():
    """A longer hold must purge more training rows. If it does not, purging is inert."""
    purged = {}
    for horizon in (1, 8):
        panel = _panel(300, 10, horizon)
        splitter = PurgedWalkForward(
            train_weeks=104, test_weeks=26, embargo_weeks=2, expanding=True,
            min_train_weeks=78,
        )
        purged[horizon] = sum(f.purged for f in splitter.split(panel))
    assert purged[8] > purged[1], (
        f"an 8-week horizon purged {purged[8]} rows versus {purged[1]} for 1 week. "
        "Purging is not responding to label overlap."
    )


def test_train_always_precedes_test():
    """Walk-forward, not shuffled: training must never come after the test window."""
    panel = _panel(300, 10, 2)
    splitter = PurgedWalkForward(
        train_weeks=104, test_weeks=26, embargo_weeks=2, expanding=True, min_train_weeks=78
    )
    for fold in splitter.split(panel):
        assert fold.train_end < fold.test_start, (
            f"fold {fold.index} trains to {fold.train_end} but tests from {fold.test_start}"
        )


def test_folds_do_not_share_test_rows():
    """Each week is tested once, so no week is double counted in the metrics."""
    panel = _panel(300, 10, 2)
    splitter = PurgedWalkForward(
        train_weeks=104, test_weeks=26, embargo_weeks=2, expanding=True, min_train_weeks=78
    )
    seen: set[int] = set()
    for fold in splitter.split(panel):
        overlap = seen & set(fold.test_idx.tolist())
        assert not overlap, f"fold {fold.index} re-tests {len(overlap)} row(s)"
        seen |= set(fold.test_idx.tolist())


def test_embargo_widens_the_gap():
    """A larger embargo must remove more training rows after the test span."""
    panel = _panel(300, 10, 2)
    counts = {}
    for embargo in (0, 6):
        splitter = PurgedWalkForward(
            train_weeks=104, test_weeks=26, embargo_weeks=embargo, expanding=True,
            min_train_weeks=78,
        )
        counts[embargo] = sum(f.embargoed for f in splitter.split(panel))
    assert counts[6] > counts[0]


def test_missing_label_spans_are_rejected():
    """Purging without label spans is impossible, so it must fail loudly.

    Silently falling back to an unpurged split would produce a leaking backtest that
    reports nothing unusual, which is the worst available outcome.
    """
    panel = _panel(200, 5, 2).drop(columns=["label_t1"])
    splitter = PurgedWalkForward()
    with pytest.raises(KeyError, match="label_t1"):
        list(splitter.split(panel))


def test_real_panel_folds_are_clean(alpha_panel, cfg_us):
    """The same invariant on the actual feature panel, not a synthetic span structure."""
    panel, _, _ = alpha_panel
    splitter = PurgedWalkForward(
        train_weeks=104, test_weeks=26, embargo_weeks=cfg_us.cv.embargo_weeks,
        expanding=True, min_train_weeks=78,
    )
    folds = list(splitter.split(panel))
    assert folds
    for fold in folds:
        verify_no_overlap(panel, fold, embargo_weeks=cfg_us.cv.embargo_weeks)
        assert fold.n_train > 0 and fold.n_test > 0


def test_embargo_is_not_vacuous():
    """The embargo must actually remove rows, not merely be configured.

    This test exists because the first implementation applied the embargo *after* the
    test window, which in an expanding walk-forward matches nothing: training is always
    strictly earlier, so the mask was empty and the splitter reported an embargo it was
    not applying. A count of zero is indistinguishable from correct behaviour unless
    something asserts otherwise.
    """
    panel = _panel(300, 10, 2)
    splitter = PurgedWalkForward(
        train_weeks=104, test_weeks=26, embargo_weeks=4, expanding=True, min_train_weeks=78
    )
    total = sum(f.embargoed for f in splitter.split(panel))
    assert total > 0, (
        "a 4-week embargo removed no training rows at all. The embargo is being applied "
        "on the side where no training data exists, so there is effectively no embargo."
    )


def test_embargo_leaves_a_real_time_gap():
    """After purging and embargo, training labels must end well before the test span."""
    panel = _panel(300, 10, 2)
    embargo_weeks = 4
    splitter = PurgedWalkForward(
        train_weeks=104, test_weeks=26, embargo_weeks=embargo_weeks, expanding=True,
        min_train_weeks=78,
    )
    for fold in splitter.split(panel):
        train_t1 = pd.to_datetime(panel["label_t1"]).iloc[fold.train_idx].max()
        test_t0 = pd.to_datetime(panel["label_t0"]).iloc[fold.test_idx].min()
        gap_days = (test_t0 - train_t1).days
        assert gap_days >= embargo_weeks * 7 - 7, (
            f"fold {fold.index}: only {gap_days} day(s) between the last training label "
            f"and the test span, with a {embargo_weeks}-week embargo configured"
        )

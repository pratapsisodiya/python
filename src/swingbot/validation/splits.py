"""Purged, embargoed walk-forward cross-validation.

Standard k-fold on time-series financial data is not merely suboptimal, it is invalid,
and the reason is worth stating precisely because it is the difference between a backtest
that means something and one that does not.

A weekly label spans the sessions from entry to exit. When the hold is longer than the
rebalance step, consecutive labels overlap: the label for week 10 and the label for week
11 both contain the move on the sessions they share. If week 10 lands in training and
week 11 in test, the model has literally seen part of the test outcome. Random shuffling
guarantees this happens on most folds.

Two corrections, both from López de Prado's *Advances in Financial Machine Learning*:

**Purging.** Drop any training sample whose label span intersects the test window.

**Embargo.** Additionally drop training samples starting shortly *after* the test window.
Features are serially correlated, so information leaks backwards across the boundary too:
a training sample just after the test period carries feature values built from data
inside it.

Walk-forward on top of both, because a model that trains on 2023 to predict 2019 is
answering a question nobody will ever ask.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from ..types import DECISION_SESSION, LABEL_T0, LABEL_T1


@dataclass(frozen=True, slots=True)
class Fold:
    """One walk-forward fold."""

    index: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    purged: int
    embargoed: int

    @property
    def n_train(self) -> int:
        return len(self.train_idx)

    @property
    def n_test(self) -> int:
        return len(self.test_idx)

    def describe(self) -> str:
        return (
            f"fold {self.index}: train {self.train_start}..{self.train_end} "
            f"({self.n_train} rows) test {self.test_start}..{self.test_end} "
            f"({self.n_test} rows), purged {self.purged}, embargoed {self.embargoed}"
        )


class PurgedWalkForward:
    """Expanding or rolling walk-forward with purging and an embargo.

    Windows are expressed in **weeks** because the decision grid is weekly, which keeps
    the configuration readable and independent of how many names are in the universe.
    """

    def __init__(
        self,
        *,
        train_weeks: int = 156,
        test_weeks: int = 26,
        embargo_weeks: int = 2,
        expanding: bool = True,
        min_train_weeks: int = 104,
    ) -> None:
        self.train_weeks = train_weeks
        self.test_weeks = test_weeks
        self.embargo_weeks = embargo_weeks
        self.expanding = expanding
        self.min_train_weeks = min_train_weeks

    # ------------------------------------------------------------------------ public

    def split(self, panel: pd.DataFrame) -> Iterator[Fold]:
        """Yield folds over a panel carrying ``decision_session`` and label spans."""
        if panel.empty:
            return

        required = {DECISION_SESSION, LABEL_T0, LABEL_T1}
        missing = required - set(panel.columns)
        if missing:
            raise KeyError(
                f"PurgedWalkForward needs {sorted(required)}; missing {sorted(missing)}. "
                "Label spans are what make purging possible."
            )

        decisions = pd.to_datetime(panel[DECISION_SESSION])
        t0 = pd.to_datetime(panel[LABEL_T0])
        t1 = pd.to_datetime(panel[LABEL_T1])

        weeks = np.sort(decisions.unique())
        n_weeks = len(weeks)
        if n_weeks < self.min_train_weeks + self.test_weeks:
            return

        positions = np.arange(len(panel))
        fold_index = 0
        start = 0

        while True:
            train_end_pos = (
                self.train_weeks + start if not self.expanding else self.train_weeks + start
            )
            if self.expanding:
                train_start_pos = 0
                train_end_pos = max(self.min_train_weeks, self.train_weeks) + start
            else:
                train_start_pos = start
                train_end_pos = start + self.train_weeks

            test_start_pos = train_end_pos
            test_end_pos = test_start_pos + self.test_weeks

            if test_start_pos >= n_weeks:
                break
            test_end_pos = min(test_end_pos, n_weeks)
            if test_end_pos - test_start_pos < max(1, self.test_weeks // 4):
                break

            train_lo = pd.Timestamp(weeks[train_start_pos])
            train_hi = pd.Timestamp(weeks[train_end_pos - 1])
            test_lo = pd.Timestamp(weeks[test_start_pos])
            test_hi = pd.Timestamp(weeks[test_end_pos - 1])

            candidate_train = (decisions >= train_lo) & (decisions <= train_hi)
            test_mask = (decisions >= test_lo) & (decisions <= test_hi)

            # The test window spans from the earliest label start to the latest label
            # end, not merely the decision dates, because a label reaches forward.
            test_span_lo = t0.loc[test_mask].min()
            test_span_hi = t1.loc[test_mask].max()
            if pd.isna(test_span_lo) or pd.isna(test_span_hi):
                break

            # Purge: any training label whose span touches the test span.
            overlaps = (t0 <= test_span_hi) & (t1 >= test_span_lo)
            purged_mask = candidate_train & overlaps

            # Embargo, applied to the LEADING edge of the test window.
            #
            # This is the detail that distinguishes a walk-forward embargo from the
            # k-fold one described in the literature. In k-fold, training data exists on
            # both sides of the test block, so the embargo trails it. In an expanding
            # walk-forward, training is always strictly before the test window, so a
            # trailing embargo can never match a single row — it is silently vacuous, and
            # a splitter that applies one has no embargo at all while appearing to.
            #
            # The gap that does matter here sits just before the test window: a training
            # label ending a day before the test span is very nearly the same
            # observation as the first test sample, because features and volatility are
            # serially correlated across that boundary. So training labels are required
            # to finish at least `embargo_weeks` before the test span opens.
            embargo_start = test_span_lo - pd.Timedelta(weeks=self.embargo_weeks)
            embargo_mask = (
                candidate_train & (~overlaps) & (t1 >= embargo_start) & (t1 < test_span_lo)
            )

            train_mask = candidate_train & (~purged_mask) & (~embargo_mask)

            if int(train_mask.sum()) > 0 and int(test_mask.sum()) > 0:
                yield Fold(
                    index=fold_index,
                    train_idx=positions[train_mask.to_numpy()],
                    test_idx=positions[test_mask.to_numpy()],
                    train_start=train_lo.date(),
                    train_end=train_hi.date(),
                    test_start=test_lo.date(),
                    test_end=test_hi.date(),
                    purged=int(purged_mask.sum()),
                    embargoed=int(embargo_mask.sum()),
                )
                fold_index += 1

            start += self.test_weeks
            if test_end_pos >= n_weeks:
                break

    def n_splits(self, panel: pd.DataFrame) -> int:
        return sum(1 for _ in self.split(panel))


def verify_no_overlap(panel: pd.DataFrame, fold: Fold, *, embargo_weeks: int = 0) -> None:
    """Assert a fold's training labels never touch its test span.

    Used by the property test. Kept here rather than in the test file so the invariant
    lives next to the code that must uphold it.
    """
    t0 = pd.to_datetime(panel[LABEL_T0])
    t1 = pd.to_datetime(panel[LABEL_T1])

    test_lo = t0.iloc[fold.test_idx].min()
    test_hi = t1.iloc[fold.test_idx].max()

    train_t0 = t0.iloc[fold.train_idx]
    train_t1 = t1.iloc[fold.train_idx]

    overlapping = (train_t0 <= test_hi) & (train_t1 >= test_lo)
    if bool(overlapping.any()):
        raise AssertionError(
            f"fold {fold.index}: {int(overlapping.sum())} training label(s) overlap the "
            f"test span {test_lo.date()}..{test_hi.date()}"
        )

    if embargo_weeks > 0:
        # Leading-edge embargo: no training label may finish inside the gap immediately
        # before the test span. See the note in PurgedWalkForward.split for why the
        # embargo sits on this side for a walk-forward split.
        embargo_start = test_lo - pd.Timedelta(weeks=embargo_weeks)
        inside = (train_t1 >= embargo_start) & (train_t1 < test_lo)
        if bool(inside.any()):
            raise AssertionError(
                f"fold {fold.index}: {int(inside.sum())} training label(s) end inside "
                f"the {embargo_weeks}-week embargo before {test_lo.date()}"
            )


class CombinatorialPurgedCV:
    """Combinatorially purged cross-validation, for probability of backtest overfitting.

    Splits the timeline into ``n_groups`` blocks and tests every combination of
    ``n_test_groups`` of them, purging and embargoing around each. That produces many
    distinct backtest paths from one dataset, which is what PBO needs: an estimate of how
    often the configuration that looked best in-sample underperforms out of sample.

    Expensive, so it is wired into the report rather than the default training loop.
    """

    def __init__(
        self,
        *,
        n_groups: int = 8,
        n_test_groups: int = 2,
        embargo_weeks: int = 2,
    ) -> None:
        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.embargo_weeks = embargo_weeks

    def split(self, panel: pd.DataFrame) -> Iterator[Fold]:
        from itertools import combinations

        if panel.empty:
            return
        decisions = pd.to_datetime(panel[DECISION_SESSION])
        t0 = pd.to_datetime(panel[LABEL_T0])
        t1 = pd.to_datetime(panel[LABEL_T1])

        weeks = np.sort(decisions.unique())
        if len(weeks) < self.n_groups * 2:
            return
        groups = np.array_split(weeks, self.n_groups)
        positions = np.arange(len(panel))

        for fold_index, combo in enumerate(
            combinations(range(self.n_groups), self.n_test_groups)
        ):
            test_weeks = np.concatenate([groups[i] for i in combo])
            test_mask = decisions.isin(test_weeks)
            if not bool(test_mask.any()):
                continue

            test_span_lo = t0.loc[test_mask].min()
            test_span_hi = t1.loc[test_mask].max()
            overlaps = (t0 <= test_span_hi) & (t1 >= test_span_lo)
            # Combinatorial folds have training data on BOTH sides of each test block, so
            # unlike the walk-forward case both embargo directions can bind.
            after_end = test_span_hi + pd.Timedelta(weeks=self.embargo_weeks)
            before_start = test_span_lo - pd.Timedelta(weeks=self.embargo_weeks)
            embargo_mask = (~overlaps) & (
                ((t0 > test_span_hi) & (t0 <= after_end))
                | ((t1 >= before_start) & (t1 < test_span_lo))
            )

            train_mask = (~test_mask) & (~overlaps) & (~embargo_mask)
            if not bool(train_mask.any()):
                continue

            yield Fold(
                index=fold_index,
                train_idx=positions[train_mask.to_numpy()],
                test_idx=positions[test_mask.to_numpy()],
                train_start=decisions[train_mask].min().date(),
                train_end=decisions[train_mask].max().date(),
                test_start=decisions[test_mask].min().date(),
                test_end=decisions[test_mask].max().date(),
                purged=int((overlaps & (~test_mask)).sum()),
                embargoed=int(embargo_mask.sum()),
            )


def sessions_to_weeks(days: int) -> int:
    return max(1, round(days / 5.0))


def weeks_between(start: date, end: date) -> int:
    return max(0, (end - start) // timedelta(weeks=1))

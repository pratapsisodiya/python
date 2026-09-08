"""Probability of backtest overfitting.

The deflated Sharpe ratio asks whether *this* result survives the number of trials that
were run. PBO asks a different and in some ways harder question: does the *procedure* of
picking the best-looking configuration generalise at all?

The mechanism, from Bailey, Borwein, López de Prado and Zhu. Split the timeline into
blocks and take every combination of them as a test set — that yields many distinct
backtest paths from one history rather than the single path a walk-forward gives. On each
path, fit every candidate configuration on the training blocks, rank them by in-sample
performance, then look at where the in-sample winner actually landed out of sample. PBO is
the fraction of paths where it landed below the median.

The number to know: **PBO above 0.5 means the selection procedure is worse than picking at
random.** That is not a rare pathology. It is the normal outcome when a handful of
configurations are compared on a decade of weekly data, which is exactly the situation
this project is in — and precisely why the candidate grid below is small and fixed rather
than generated from whatever search happened to be running.

Expensive: ``C(n_groups, n_test_groups)`` folds times the number of candidates, each a
full model fit. Off by default, behind ``swingbot backtest --pbo``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from ..config import Config
from ..model import build_model
from ..types import DECISION_SESSION, LABEL
from ..validation.deflated import probability_of_backtest_overfitting
from ..validation.splits import CombinatorialPurgedCV

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Candidate:
    """One configuration a user might plausibly have chosen between."""

    name: str
    kind: str
    overrides: dict


def default_candidates(cfg: Config) -> list[Candidate]:
    """A small, fixed grid spanning the choices a user actually faces.

    Deliberately not a wide search. PBO measures the cost of *choosing*, so the grid has
    to be the set of things someone would really pick between — model family, how hard the
    trees are allowed to fit, how much the ridge is penalised. Padding it with near
    duplicates would inflate the path count and flatter the result, because near-identical
    configurations rarely disagree about which is best.
    """
    return [
        Candidate("ridge_light", "ridge", {"alpha": 1.0}),
        Candidate("ridge_default", "ridge", {"alpha": cfg.model.ridge.alpha}),
        Candidate("ridge_heavy", "ridge", {"alpha": 100.0}),
        Candidate("gbdt_shallow", "gbdt", {"num_leaves": 7, "n_estimators": 150}),
        Candidate("gbdt_default", "gbdt", {}),
        Candidate("gbdt_deep", "gbdt", {"num_leaves": 63, "n_estimators": 600}),
    ]


def _build(cfg: Config, candidate: Candidate, seed: int):
    local = cfg.model_copy(deep=True)
    block = getattr(local.model, candidate.kind)
    for key, value in candidate.overrides.items():
        setattr(block, key, value)
    return build_model(candidate.kind, local, seed=seed)


def _information_coefficient(frame: pd.DataFrame) -> float:
    """Mean per-week Spearman IC.

    Per week, then averaged — never pooled. Pooling a decade of predictions into one
    correlation treats several hundred cross-sectionally correlated names in one week as
    independent observations, which is the same mistake, one level down, that this whole
    validation package exists to avoid.
    """
    weekly = []
    for _, block in frame.groupby(DECISION_SESSION, sort=False):
        clean = block[["prediction", LABEL]].dropna()
        if len(clean) < 5 or clean["prediction"].nunique() < 3:
            continue
        rho = stats.spearmanr(clean["prediction"], clean[LABEL]).statistic
        if np.isfinite(rho):
            weekly.append(float(rho))
    return float(np.mean(weekly)) if weekly else float("nan")


@dataclass(slots=True)
class PBOResult:
    pbo: float = 0.0
    n_paths: int = 0
    candidates: list[str] = field(default_factory=list)
    #: Per path: which candidate won in sample, and its out-of-sample rank among all.
    selections: list[dict] = field(default_factory=list)
    in_sample: np.ndarray | None = None
    out_of_sample: np.ndarray | None = None

    @property
    def is_valid(self) -> bool:
        return self.n_paths > 0

    def verdict(self) -> str:
        if not self.is_valid:
            return "not enough history to form combinatorial paths"
        if self.pbo > 0.5:
            return (
                "selecting the best-looking configuration is worse than choosing at "
                "random — treat any single reported result as noise"
            )
        if self.pbo > 0.25:
            return "configuration choice carries real overfitting risk"
        return "the selection procedure generalises across combinatorial paths"

    def to_dict(self) -> dict:
        return {
            "pbo": self.pbo,
            "n_paths": self.n_paths,
            "n_candidates": len(self.candidates),
            "candidates": list(self.candidates),
            "verdict": self.verdict(),
        }

    def table(self) -> pd.DataFrame:
        return pd.DataFrame(self.selections)


def run_pbo(
    panel: pd.DataFrame,
    cfg: Config,
    *,
    feature_columns: list[str],
    candidates: list[Candidate] | None = None,
    n_groups: int = 8,
    n_test_groups: int = 2,
) -> PBOResult:
    """Fit every candidate on every combinatorial path and measure the cost of choosing."""
    candidates = candidates or default_candidates(cfg)
    result = PBOResult(candidates=[c.name for c in candidates])
    if panel.empty:
        return result

    splitter = CombinatorialPurgedCV(
        n_groups=n_groups,
        n_test_groups=n_test_groups,
        embargo_weeks=cfg.cv.embargo_weeks,
    )

    is_rows: list[list[float]] = []
    oos_rows: list[list[float]] = []
    selections: list[dict] = []

    for fold in splitter.split(panel):
        train = panel.iloc[fold.train_idx]
        test = panel.iloc[fold.test_idx]
        if train[LABEL].notna().sum() < 200 or test[LABEL].notna().sum() < 50:
            continue

        path_is: list[float] = []
        path_oos: list[float] = []
        for candidate in candidates:
            model = _build(cfg, candidate, cfg.run.seed)
            model.fit(
                train[feature_columns],
                train[LABEL],
                sample_weight=train.get("sample_weight"),
                week_index=train[DECISION_SESSION],
            )
            for source, sink in ((train, path_is), (test, path_oos)):
                scored = source.copy()
                scored["prediction"] = model.predict(source[feature_columns]).to_numpy()
                sink.append(_information_coefficient(scored))

        if not np.isfinite(path_is).any() or not np.isfinite(path_oos).any():
            continue

        is_rows.append(path_is)
        oos_rows.append(path_oos)

        winner = int(np.nanargmax(path_is))
        order = np.argsort(np.argsort(-np.asarray(path_oos)))
        selections.append(
            {
                "path": fold.index,
                "test_start": fold.test_start,
                "test_end": fold.test_end,
                "chosen": candidates[winner].name,
                "in_sample_ic": round(path_is[winner], 5),
                "out_of_sample_ic": round(path_oos[winner], 5),
                "out_of_sample_rank": int(order[winner]) + 1,
                "of": len(candidates),
            }
        )
        log.info(
            "PBO path %d: chose %s (IS IC %.4f), landed %d/%d out of sample",
            fold.index, candidates[winner].name, path_is[winner],
            int(order[winner]) + 1, len(candidates),
        )

    if not is_rows:
        return result

    result.in_sample = np.asarray(is_rows)
    result.out_of_sample = np.asarray(oos_rows)
    result.n_paths = len(is_rows)
    result.selections = selections
    result.pbo = probability_of_backtest_overfitting(result.in_sample, result.out_of_sample)
    return result

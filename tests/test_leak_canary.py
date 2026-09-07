"""Leak canaries: three assertions that together bound what the harness can be doing.

The look-ahead test proves features do not read the future. These prove the *modelling*
path is sound, which is a different question — a perfectly causal feature set can still
be scored by a leaking evaluation loop.

1. **Shuffled labels must find nothing.** Train the real model on labels permuted within
   each week, destroying the name-to-outcome mapping while leaving everything else intact.
   Any remaining predictive power is leakage by definition.
2. **A deliberately cheating variant must score higher.** Shift the features forward so
   the model sees next week's values. If that does *not* improve the score, the evaluation
   loop is inert and the "honest" result is measuring nothing.
3. **The honest variant must find the injected signal.** On a market with planted alpha,
   the model must detect it. A pipeline that silently reports "no alpha" on data that
   contains it would look identical to an honest negative result.

Only all three together are informative. Passing (1) alone is consistent with a pipeline
that is simply broken.
"""

from __future__ import annotations

from swingbot.backtest import walk_forward_predict
from swingbot.features.pipeline import price_feature_columns
from swingbot.model import ShuffledLabelModel, build_model
from swingbot.validation import fold_clustered_ic, summarize_ic
from swingbot.validation.splits import PurgedWalkForward


def _splitter(cfg):
    return PurgedWalkForward(
        train_weeks=104, test_weeks=26, embargo_weeks=cfg.cv.embargo_weeks,
        expanding=True, min_train_weeks=78,
    )


def _run(panel, cfg, factory, columns):
    return walk_forward_predict(
        panel, factory, feature_columns=columns, splitter=_splitter(cfg)
    )


def test_honest_model_finds_injected_alpha(alpha_panel, cfg_us):
    """Canary 3: the pipeline must detect a signal that is genuinely there."""
    panel, _, _ = alpha_panel
    columns = price_feature_columns(panel)
    predictions = _run(panel, cfg_us, lambda: build_model("gbdt", cfg_us), columns)
    assert not predictions.empty, "no out-of-sample predictions were produced"

    ic = summarize_ic(predictions)
    assert ic.mean_ic > 0.01, (
        f"the model found IC {ic.mean_ic:.4f} on data with planted alpha. The pipeline "
        "is broken, and a genuine negative result would look exactly like this."
    )
    assert ic.t_stat > 2.0


def test_shuffled_labels_find_nothing(alpha_panel, cfg_us):
    """Canary 1: the leak test. Clustered by fold, not by week.

    Fold clustering is essential here. The model is refit once per fold, so its IC has a
    fixed sign for every week within that fold and week-level clustering overstates the
    sample size by roughly the number of weeks per fold.
    """
    panel, _, _ = alpha_panel
    columns = price_feature_columns(panel)
    predictions = _run(
        panel, cfg_us,
        lambda: ShuffledLabelModel(build_model("gbdt", cfg_us), seed=cfg_us.run.seed),
        columns,
    )
    assert not predictions.empty

    ic = fold_clustered_ic(predictions)
    assert not (ic.mean_ic > 0.005 and ic.t_stat > 2.5), (
        f"shuffled labels still predict: IC {ic.mean_ic:+.4f} at t={ic.t_stat:.2f} across "
        f"{ic.n_periods} fold(s). Something is leaking the outcome into the features or "
        "into the training split."
    )


def test_cheating_variant_scores_higher(alpha_panel, cfg_us):
    """Canary 2: a model given the future must beat one that is not.

    Proves the evaluation loop is live. A pipeline where the features are silently
    disconnected from the scoring would pass canary 1 trivially and mean nothing.
    """
    panel, _, _ = alpha_panel
    columns = price_feature_columns(panel)

    honest = _run(panel, cfg_us, lambda: build_model("gbdt", cfg_us), columns)
    honest_ic = summarize_ic(honest).mean_ic

    # Shift each name's features one week back in time, so the row for week t carries the
    # features of week t+1 while keeping week t's label. Deliberate look-ahead.
    cheat = panel.sort_values(["ticker", "decision_session"]).copy()
    cheat[columns] = cheat.groupby("ticker", sort=False)[columns].shift(-1)
    cheat = cheat.dropna(subset=columns, how="all")

    cheating = _run(cheat, cfg_us, lambda: build_model("gbdt", cfg_us), columns)
    cheating_ic = summarize_ic(cheating).mean_ic

    assert cheating_ic > honest_ic + 0.01, (
        f"a model shown next week's features scored IC {cheating_ic:.4f} versus "
        f"{honest_ic:.4f} honestly. The evaluation loop is not responding to the features "
        "it is given, so the honest number is not measuring anything."
    )


def test_null_market_yields_nothing(null_panel, cfg_us):
    """The strongest single check: no signal in, no signal out.

    A harness that finds alpha in a market generated without any is leaking, and this
    catches leaks the shuffled canary can miss because it removes the effect at the source
    rather than at the label.
    """
    panel, _, _ = null_panel
    columns = price_feature_columns(panel)
    predictions = _run(panel, cfg_us, lambda: build_model("gbdt", cfg_us), columns)
    assert not predictions.empty

    ic = summarize_ic(predictions)
    assert abs(ic.mean_ic) < 0.03, (
        f"found IC {ic.mean_ic:+.4f} on a market with no injected signal. The harness is "
        "manufacturing alpha from nothing."
    )

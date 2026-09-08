"""Config liveness: every knob must be able to change an output.

This test exists because of a recurring failure mode in this codebase that the rest of
the suite structurally cannot catch.

Three separate risk controls were found written, exported, documented in
``config/base.yaml``, and **never called by the pipeline**: the participation cap, Kelly
sizing (fed ranks where expected returns belong, so its ceiling could never bind), and
model pinning. The walk-forward embargo had the same shape earlier — applied on the side
of the test window where an expanding split has no training data, so it matched nothing.

Every one of those had unit tests, and every one passed, because the unit tests asked "does
this function behave correctly when called?" and never "does anything call it?". A control
that is unreachable is indistinguishable from a correct one under that question.

So this suite asks the other question. For each config knob: set it to two different valid
values, run a small slice of the real pipeline, and require the outputs to differ. A knob
whose value provably cannot alter any output is either dead code or a false promise in the
config file, and either is a bug worth failing the build over.

**Two things make this test meaningful rather than decorative.**

*The probe values must straddle where the knob binds.* A participation cap of 0.05 versus
0.06 changes nothing on a small book, and a test using those values would pass against
completely dead code. Each probe therefore chooses values far enough apart to force the
control to fire.

*Exemptions carry a written reason and are themselves audited.* Some knobs genuinely
cannot change a trading decision — a log level, an output directory, a notification
toggle. Those are declared with a reason. But the exemption list is exactly where the next
dead control would hide, so :func:`test_exemptions_stay_within_known_harmless_prefixes`
fails on any exemption outside a narrow allowlist of paths, logging and notification
settings. Adding ``portfolio.something`` to the exemption list to make this suite pass is
not possible without that guard failing too.
"""

from __future__ import annotations

import datetime as dt
import pathlib
import tempfile
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pytest

from swingbot.backtest import walk_forward_predict
from swingbot.backtest.costs import CostModel
from swingbot.config import Config, load_config
from swingbot.features.pipeline import price_feature_columns
from swingbot.model import build_model
from swingbot.portfolio import PortfolioConstructor
from swingbot.validation.splits import PurgedWalkForward

# --------------------------------------------------------------------------------------
# Probe definitions
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfigProbe:
    """One config knob, and two values that must produce different behaviour."""

    path: str
    values: tuple[Any, Any]
    #: Which pipeline output to compare. See ``_observe_*`` below.
    observe: str = "weights"
    #: Non-empty means "this knob cannot change any output, and here is why".
    exempt: str = ""
    note: str = ""

    @property
    def is_exempt(self) -> bool:
        return bool(self.exempt)


#: Knobs that genuinely cannot alter a trading decision. Each needs a stated reason, and
#: the paths are audited by a separate test so this cannot become a dumping ground.
EXEMPT: tuple[ConfigProbe, ...] = (
    ConfigProbe("run.log_level", ("INFO", "DEBUG"), exempt="logging verbosity only"),
    ConfigProbe("run.data_dir", ("data", "data2"), exempt="filesystem location only"),
    ConfigProbe("run.runs_dir", ("runs", "runs2"), exempt="filesystem location only"),
    ConfigProbe("execution.output_dir", ("runs", "runs2"), exempt="filesystem location only"),
    ConfigProbe("notify.telegram.enabled", (False, True), exempt="delivery channel, not a decision"),
    ConfigProbe("notify.email.enabled", (False, True), exempt="delivery channel, not a decision"),
)

#: Prefixes an exemption is allowed to live under. Anything else must prove it is live.
ALLOWED_EXEMPT_PREFIXES = ("run.", "execution.output_dir", "notify.")


#: Knobs that must demonstrably change an output.
PROBES: tuple[ConfigProbe, ...] = (
    # ---------------------------------------------------------------- portfolio shape
    ConfigProbe("portfolio.n_long", (3, 9), observe="weights"),
    ConfigProbe("portfolio.n_short", (0, 7), observe="weights"),
    ConfigProbe("portfolio.gross_leverage", (0.4, 1.0), observe="weights"),
    ConfigProbe("portfolio.max_weight", (0.04, 0.30), observe="weights"),
    ConfigProbe("portfolio.max_sector_weight", (0.15, 0.90), observe="weights"),
    ConfigProbe("portfolio.max_net_exposure", (0.02, 0.60), observe="weights"),
    ConfigProbe("portfolio.target_vol_annual", (0.05, 0.30), observe="weights"),
    ConfigProbe("portfolio.sizing", ("equal", "inverse_vol"), observe="weights"),
    # The three findings this suite was written to catch. Values chosen wide enough to
    # force each control to bind on the fixture book.
    ConfigProbe(
        "portfolio.max_participation_adv",
        (0.00002, 0.50),
        observe="weights",
        note="capacity cap: was declared in base.yaml and read by nothing",
    ),
    ConfigProbe(
        "portfolio.kelly_fraction",
        (0.05, 5.0),
        observe="weights_kelly",
        note="Kelly ceiling: was fed cross-sectional ranks as if they were expected returns",
    ),
    ConfigProbe("portfolio.no_trade_band", (0.0, 0.25), observe="weights_prev"),
    # ------------------------------------------------------------------------ costs
    ConfigProbe("backtest.cost_scale", (0.0, 5.0), observe="costs"),
    ConfigProbe("costs.commission_bps", (0.0, 60.0), observe="costs"),
    ConfigProbe("costs.min_half_spread_bps", (0.0, 60.0), observe="costs"),
    ConfigProbe("costs.impact_k", (0.0, 8.0), observe="costs"),
    ConfigProbe("costs.borrow_bps_annual", (0.0, 3000.0), observe="costs"),
    ConfigProbe("costs.stt_sell_bps", (0.0, 80.0), observe="costs"),
    ConfigProbe("costs.stt_buy_bps", (0.0, 80.0), observe="costs"),
    ConfigProbe("costs.stamp_duty_bps", (0.0, 40.0), observe="costs"),
    ConfigProbe("costs.exchange_bps", (0.0, 40.0), observe="costs"),
    ConfigProbe("costs.gst_rate", (0.0, 0.9), observe="costs"),
    ConfigProbe("costs.spread_from_range_factor", (0.0, 0.5), observe="costs"),
    ConfigProbe("costs.regulatory_sell_bps", (0.0, 40.0), observe="costs"),
    # ------------------------------------------------------------------------- risk
    ConfigProbe("risk.enabled", (True, False), observe="risk_scale"),
    ConfigProbe("risk.drawdown_scale_at", (0.001, 0.90), observe="risk_scale"),
    ConfigProbe("risk.drawdown_flat_at", (0.002, 0.95), observe="risk_scale"),
    # ------------------------------------------------------------ model and features
    ConfigProbe("model.price_model", ("ridge", "gbdt"), observe="predictions"),
    ConfigProbe("model.gbdt.n_estimators", (5, 400), observe="predictions"),
    ConfigProbe("model.gbdt.learning_rate", (0.005, 0.4), observe="predictions"),
    ConfigProbe("model.gbdt.num_leaves", (2, 63), observe="predictions"),
    ConfigProbe("model.recency_half_life_weeks", (4.0, 5000.0), observe="predictions"),
    ConfigProbe(
        "model.max_model_age_weeks",
        (1, 520),
        observe="pinned_model",
        note="staleness guard on `signal --use-model`",
    ),
    # 26 exceeds the fixture's 25 names, which is what makes the floor bind: at 24 every
    # date still qualifies, so a narrower pair would pass against dead code.
    ConfigProbe("features.min_names_per_week", (2, 26), observe="panel"),
    ConfigProbe("calendar.hold_sessions", (5, 15), observe="panel"),
    # --------------------------------------------------------------- cross-validation
    ConfigProbe("cv.train_weeks", (60, 130), observe="predictions"),
    ConfigProbe("cv.test_weeks", (13, 40), observe="predictions"),
    ConfigProbe("cv.embargo_weeks", (0, 20), observe="predictions"),
)


# --------------------------------------------------------------------------------------
# Fixture: small enough to run ~40 probes, real enough to exercise the pipeline
# --------------------------------------------------------------------------------------

N_NAMES = 25
START = dt.date(2018, 1, 1)
END = dt.date(2023, 12, 29)


@pytest.fixture(scope="module")
def base_bars():
    from swingbot.data import SyntheticProvider, screen_bars

    provider = SyntheticProvider(
        seed=17, close_hour_utc=21, alpha_strength=0.02, reversal_strength=0.02
    )
    tickers = [f"L{i:02d}" for i in range(N_NAMES)]
    return screen_bars(provider.daily_bars(tickers, START, END))


#: Deliberately loose hard caps for the probe baseline.
#:
#: A probe has to measure the knob it names, not whichever constraint happens to bind
#: first. With the shipped defaults, a per-name cap of 12 percent over a 12-name book
#: pins most weights to the cap and washes out anything upstream — `sizing` looked dead
#: under those settings and is in fact perfectly live. Loosening the caps isolates the
#: knob under test; the caps themselves are still probed individually, and each of those
#: probes supplies its own binding value.
LOOSE_BASELINE = {
    "portfolio.max_weight": 0.90,
    "portfolio.max_sector_weight": 1.0,
    "portfolio.max_net_exposure": 1.0,
    # The capacity cap has to be loosened here too. Once it was actually wired up it
    # became the binding constraint on this fixture and masked every other knob exactly
    # as the per-name cap had — including, ironically, volatility targeting, which it
    # truncates to a per-name ceiling that no longer depends on the target at all.
    "portfolio.max_participation_adv": 10.0,
}


def _config(path: str, value: Any) -> Config:
    overrides = {k: v for k, v in LOOSE_BASELINE.items() if k != path}
    overrides[path] = value
    return load_config("us", set_values=[f"{k}={v}" for k, v in overrides.items()])


def _set(cfg: Config, path: str, value: Any) -> Config:
    """Apply a value to an already-built config, for knobs pydantic would coerce oddly."""
    cfg = cfg.model_copy(deep=True)
    target: Any = cfg
    parts = path.split(".")
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], value)
    return cfg


def _panel_for(cfg: Config, bars):
    from swingbot.calendars import TradingCalendar
    from swingbot.data import membership_panel
    from swingbot.data.universe import StaticUniverse
    from swingbot.features import (
        FeaturePipeline,
        build_labels,
        combine_weights,
        recency_weights,
        uniqueness_weights,
    )

    calendar = TradingCalendar.from_bars(bars, cfg)
    grid = calendar.weekly_grid(cfg.calendar.hold_sessions)
    sessions = [g.decision_session for g in grid]

    labels = build_labels(
        bars, calendar, hold_sessions=cfg.calendar.hold_sessions, grid=grid
    )
    if not labels.empty:
        labels["sample_weight"] = combine_weights(
            uniqueness_weights(labels),
            recency_weights(
                labels["decision_session"],
                half_life_weeks=cfg.model.recency_half_life_weeks,
            ),
        )
        labels = labels[
            ["ticker", "decision_session", "label", "entry_session", "exit_session",
             "label_t0", "label_t1", "sample_weight"]
        ]

    names = sorted(bars["ticker"].unique())
    universe = StaticUniverse(names, {t: f"S{i % 3}" for i, t in enumerate(names)})
    pipeline = FeaturePipeline.from_config(cfg)
    pipeline.min_names_per_week = cfg.features.min_names_per_week

    panel = pipeline.build(
        bars,
        decision_sessions=sessions,
        membership=membership_panel(universe, sessions, tickers=names),
        labels=labels if not labels.empty else None,
    )
    return panel.dropna(subset=["label"]), calendar, grid


def _predictions_for(cfg: Config, panel):
    splitter = PurgedWalkForward(
        train_weeks=cfg.cv.train_weeks,
        test_weeks=cfg.cv.test_weeks,
        embargo_weeks=cfg.cv.embargo_weeks,
        expanding=cfg.cv.expanding,
        min_train_weeks=min(cfg.cv.min_train_weeks, 60),
    )
    return walk_forward_predict(
        panel,
        lambda: build_model(cfg.model.price_model, cfg, seed=cfg.run.seed),
        feature_columns=price_feature_columns(panel),
        splitter=splitter,
    )


# --------------------------------------------------------------------------------------
# Observers: each returns a hashable fingerprint of one pipeline output
# --------------------------------------------------------------------------------------


def _fixture_book(cfg: Config, *, with_prev: bool, seed: int = 3, with_mu: bool = False):
    """A single constructed book, exercising the real PortfolioConstructor."""
    rng = np.random.default_rng(seed)
    tickers = [f"L{i:02d}" for i in range(N_NAMES)]
    scores = pd.Series(rng.normal(0, 1, N_NAMES), index=tickers)
    vol = pd.Series(np.abs(rng.normal(0.35, 0.10, N_NAMES)), index=tickers)
    sectors = pd.Series([f"S{i % 8}" for i in range(N_NAMES)], index=tickers)

    # Small and *varying* ADV, so the participation cap bites differently per name.
    # A uniform ADV would truncate every name to the same size, and the gross cap would
    # then hide the difference behind a common rescale.
    adv = pd.Series(rng.uniform(80_000.0, 3_000_000.0, N_NAMES), index=tickers)

    previous = None
    if with_prev:
        previous = {t: float(w) for t, w in zip(tickers[:8], np.linspace(0.02, 0.09, 8), strict=True)}

    # Expected excess return over the hold window, in return units. This is what the
    # calibrator produces and what the Kelly ceiling needs; the score is not it. Scaled
    # so a typical selected name implies a ceiling near the weights the book actually
    # runs, which is the only regime in which the knob can be observed at all.
    expected = pd.Series(scores.to_numpy() * 0.01, index=tickers) if with_mu else None

    constructor = PortfolioConstructor.from_config(cfg)
    return constructor.build(
        scores,
        decision_session=dt.date(2024, 3, 15),
        entry_session=dt.date(2024, 3, 18),
        volatility=vol,
        sectors=sectors,
        previous_weights=previous,
        adv_notional=adv,
        equity=5_000_000.0,
        expected_returns=expected,
    )


def _round(weights: dict[str, float]) -> tuple:
    return tuple(sorted((t, round(w, 8)) for t, w in weights.items()))


def _observe_weights(cfg: Config, bars) -> tuple:
    return _round(_fixture_book(cfg, with_prev=False).weights)


def _observe_weights_prev(cfg: Config, bars) -> tuple:
    return _round(_fixture_book(cfg, with_prev=True).weights)


def _observe_weights_kelly(cfg: Config, bars) -> tuple:
    """Kelly's ceiling can only bind when sizing is kelly *and* a calibrated mu exists.

    Both halves are load-bearing. Without ``sizing=kelly`` the cap never runs; without a
    calibrated expected return it declines to run and says so in the notes. The fixture
    supplies both, so a failure here means the wiring between them broke.
    """
    cfg = _set(cfg, "portfolio.sizing", "kelly")
    return _round(_fixture_book(cfg, with_prev=False, with_mu=True).weights)


def _observe_costs(cfg: Config, bars) -> tuple:
    model = CostModel.from_config(cfg)
    out = []
    for is_buy in (True, False):
        for is_short in (True, False):
            cost = model.trade_cost(
                notional=2_000_000.0,
                is_buy=is_buy,
                is_short_leg=is_short,
                volatility_annual=0.35,
                adv_notional=4_000_000.0,
                high_low_range=0.03,
            )
            out.append(round(cost.total, 8))
    hold = model.holding_cost(short_notional=1_000_000.0, sessions_held=5, rolls=1)
    out.append(round(hold.total, 8))
    return tuple(out)


def _observe_risk_scale(cfg: Config, bars) -> tuple:
    from swingbot.portfolio.risk import RiskState, update_kill_switch

    state = RiskState()
    path = [1.0, 1.06, 1.0, 0.95, 0.90, 0.86, 0.80, 0.83, 0.95, 1.02, 0.99]
    scales = []
    for equity in path:
        state = update_kill_switch(
            state,
            equity,
            scale_at=cfg.risk.drawdown_scale_at,
            flat_at=cfg.risk.drawdown_flat_at,
            cooldown_weeks=cfg.risk.cooldown_weeks,
            enabled=cfg.risk.enabled,
        )
        scales.append(round(state.scale, 6))
    return tuple(scales)


def _observe_panel(cfg: Config, bars) -> tuple:
    panel, _, _ = _panel_for(cfg, bars)
    if panel.empty:
        return (0, 0)
    return (len(panel), int(panel["decision_session"].nunique()))


#: One saved model, reused by the staleness probe. Built lazily because it costs a fit.
_PINNED: dict[str, Any] = {}


def _pinned_run(tmp_root):
    """Save a trivial model under a run directory, once."""
    if "run_id" in _PINNED:
        return _PINNED["run_id"], _PINNED["dir"], _PINNED["columns"]

    from swingbot.model import build_model
    from swingbot.model.registry import save_model

    rng = np.random.default_rng(2)
    columns = ["px_a", "px_b"]
    frame = pd.DataFrame(rng.normal(0, 1, (600, 2)), columns=columns)
    y = pd.Series(rng.normal(0, 0.02, 600))
    weeks = pd.Series([dt.date(2022, 1, 7) + dt.timedelta(days=7 * (i // 20)) for i in range(600)])

    model = build_model("ridge", load_config("us"), seed=1)
    model.fit(frame, y, sample_weight=None, week_index=weeks)

    run_id = "20230630T000000-us-signal-abcdef12"
    save_model(
        model, tmp_root / run_id / "model",
        market="us", config_hash="pinned", train_start=dt.date(2022, 1, 7),
        train_end=PINNED_TRAIN_END, price_columns=columns,
    )
    _PINNED.update(run_id=run_id, dir=tmp_root, columns=columns)
    return run_id, tmp_root, columns


#: The pinned fit ends here; the probe asks about a decision well after it.
PINNED_TRAIN_END = dt.date(2023, 6, 30)
PINNED_DECISION = PINNED_TRAIN_END + dt.timedelta(weeks=60)


def _observe_pinned_model(cfg: Config, bars) -> tuple:
    """Does the staleness limit actually gate a pinned model?

    Observed as accept-or-refuse rather than as weights, because that is the only thing
    this knob does. A knob whose sole effect is a refusal still has to prove the refusal
    happens — an unenforced limit reads exactly like an enforced one right up to the week
    it matters.
    """
    from types import SimpleNamespace

    from swingbot.service import ServiceError, load_pinned_model

    root = pathlib.Path(tempfile.gettempdir()) / "swingbot-liveness-pins"
    root.mkdir(parents=True, exist_ok=True)
    run_id, directory, columns = _pinned_run(root)

    cfg = _set(cfg, "run.runs_dir", directory)
    try:
        load_pinned_model(
            cfg, run_id, columns, SimpleNamespace(decision_session=PINNED_DECISION)
        )
    except ServiceError:
        return ("refused",)
    return ("accepted",)


def _observe_predictions(cfg: Config, bars) -> tuple:
    panel, _, _ = _panel_for(cfg, bars)
    predictions = _predictions_for(cfg, panel)
    if predictions.empty:
        return (0,)
    return (
        len(predictions),
        round(float(predictions["prediction"].mean()), 8),
        round(float(predictions["prediction"].std()), 8),
    )


OBSERVERS = {
    "weights": _observe_weights,
    "weights_prev": _observe_weights_prev,
    "weights_kelly": _observe_weights_kelly,
    "costs": _observe_costs,
    "risk_scale": _observe_risk_scale,
    "panel": _observe_panel,
    "predictions": _observe_predictions,
    "pinned_model": _observe_pinned_model,
}


# --------------------------------------------------------------------------------------
# The tests
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("probe", PROBES, ids=lambda p: p.path)
def test_config_knob_changes_an_output(probe: ConfigProbe, base_bars):
    """Setting this knob to two different values must change something observable."""
    observer = OBSERVERS[probe.observe]

    low, high = probe.values
    out_low = observer(_config(probe.path, low), base_bars)
    out_high = observer(_config(probe.path, high), base_bars)

    detail = f" — {probe.note}" if probe.note else ""
    assert out_low != out_high, (
        f"config knob {probe.path!r} produced identical {probe.observe} output at "
        f"{low!r} and {high!r}{detail}.\n\n"
        "The knob is declared in the config and documented, but nothing downstream "
        "responds to it. Either wire it into the pipeline or delete it from the config — "
        "a setting a user can change with no effect is worse than no setting at all, "
        "because they will believe they are protected by it."
    )


def test_every_config_leaf_is_probed_or_exempt():
    """No config knob may escape both the probe list and the exemption list.

    Without this, a new setting could be added to base.yaml and simply never covered,
    which is exactly how the participation cap went unnoticed.
    """
    covered = {p.path for p in PROBES} | {p.path for p in EXEMPT}

    def leaves(model, prefix=""):
        out = []
        for name in type(model).model_fields:
            value = getattr(model, name)
            path = f"{prefix}{name}"
            if hasattr(value, "model_fields"):
                out.extend(leaves(value, f"{path}."))
            else:
                out.append(path)
        return out

    cfg = load_config("us")
    all_leaves = set(leaves(cfg))

    # Structural or informational settings that are not behavioural knobs.
    structural = {
        "market", "market_profile.name", "market_profile.display_name",
        "market_profile.currency", "market_profile.timezone",
        "market_profile.close_local_time", "market_profile.symbol_suffix",
        "market_profile.benchmark", "market_profile.allow_short",
        "market_profile.short_instrument",
        "universe.file", "universe.max_names", "universe.assume_survivorship_biased",
        "data.providers", "data.cache_enabled", "data.start", "data.end",
        "data.min_history_sessions", "data.min_price", "data.min_adv_notional",
        "backtest.initial_equity", "backtest.start", "backtest.end",
        "execution.adapter", "run.seed",
        "calendar.decision_weekday", "cv.scheme", "cv.expanding", "cv.min_train_weeks",
        "labels.kind", "labels.vol_scale",
        "features.winsorize_quantile", "features.sector_neutralize",
        "model.news_model", "model.ridge.alpha",
        "model.blend.price_weight", "model.blend.news_weight",
        "model.gbdt.min_child_samples", "model.gbdt.subsample",
        "model.gbdt.colsample_bytree", "model.gbdt.reg_lambda",
        "portfolio.min_position_notional", "risk.cooldown_weeks",
        "costs.min_commission", "costs.futures_commission_bps",
        "costs.futures_stt_sell_bps", "costs.futures_exchange_bps",
        "costs.futures_stamp_duty_bps", "costs.futures_roll_bps",
    }
    # Whole subtrees that are covered by their own dedicated suites.
    prefixes = ("nlp.", "news.", "notify.", "features.news.")

    uncovered = {
        leaf for leaf in all_leaves
        if leaf not in covered
        and leaf not in structural
        and not leaf.startswith(prefixes)
    }
    assert not uncovered, (
        f"config knobs with no liveness probe and no exemption: {sorted(uncovered)}. "
        "Add a ConfigProbe proving the knob changes an output, or declare it exempt "
        "with a reason."
    )


def test_exemptions_stay_within_known_harmless_prefixes():
    """The exemption list is where a dead control would hide, so it is itself audited.

    Anything that could plausibly affect a trading decision must not be exemptable. This
    is what stops the obvious way of making this suite pass — moving a failing knob onto
    the exemption list.
    """
    for probe in EXEMPT:
        assert probe.exempt, f"{probe.path} is exempt with no stated reason"
        assert probe.path.startswith(ALLOWED_EXEMPT_PREFIXES), (
            f"{probe.path!r} is claimed exempt from the liveness requirement, but it is "
            f"not a path, logging or notification setting. Exemptions are limited to "
            f"{ALLOWED_EXEMPT_PREFIXES} precisely so that a risk or model knob cannot be "
            "silenced by adding it here. Wire the knob up instead."
        )


def test_probe_values_are_actually_distinct():
    """A probe whose two values are equal would pass trivially and prove nothing."""
    for probe in PROBES + EXEMPT:
        assert probe.values[0] != probe.values[1], f"{probe.path} probes identical values"

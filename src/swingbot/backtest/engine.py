"""The weekly backtest engine.

One loop, run over the purged walk-forward folds. For each decision date in a test fold:

1. Take the model's out-of-sample prediction. The model never saw this week.
2. Build the target book from it, under the risk limits and the kill-switch scale.
3. Charge the cost of moving from last week's book to this one.
4. Mark the position using the realised open-to-open return over the hold window, which
   is the same quantity the label measured.

The two properties that make this honest, both enforced structurally rather than by
convention:

**Predictions are out-of-sample only.** The engine consumes a prediction frame produced by
:func:`walk_forward_predict`, which fits on purged training rows and predicts the test
fold. There is no code path here that can score a week the model was trained on.

**Costs and the kill-switch act inside the loop.** Both are path-dependent. Charging costs
at the end against average turnover understates them, and applying a drawdown rule to a
finished equity curve grants protection that was never paid for: de-risking into a
drawdown also means being small in the recovery.
"""

from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pandas as pd

from ..calendars import TradingCalendar, WeeklyDecision
from ..config import Config
from ..features.labels import forward_return_matrix
from ..portfolio.construct import PortfolioConstructor
from ..portfolio.risk import RiskState, update_kill_switch
from ..types import DECISION_SESSION, LABEL, SECTOR, TICKER
from ..validation.splits import PurgedWalkForward
from .costs import CostBreakdown, CostModel
from .results import BacktestResult

log = logging.getLogger(__name__)


def walk_forward_predict(
    panel: pd.DataFrame,
    model_factory,
    *,
    feature_columns: list[str],
    splitter: PurgedWalkForward,
    label_column: str = LABEL,
    weight_column: str | None = "sample_weight",
    verbose: bool = False,
) -> pd.DataFrame:
    """Fit on purged training rows, predict the test fold, repeat.

    Returns only the test-fold rows, each carrying an out-of-sample prediction. A week
    never appears twice, and no week's prediction comes from a model that saw it.
    """
    if panel.empty:
        return pd.DataFrame()

    frames = []
    fold_log: list[dict] = []

    for fold in splitter.split(panel):
        train = panel.iloc[fold.train_idx]
        test = panel.iloc[fold.test_idx]

        y_train = train[label_column]
        if y_train.notna().sum() < 200:
            log.warning("fold %d skipped: only %d labels", fold.index, int(y_train.notna().sum()))
            continue

        model = model_factory()
        weights = (
            train[weight_column]
            if weight_column and weight_column in train.columns
            else None
        )
        model.fit(
            train[feature_columns],
            y_train,
            sample_weight=weights,
            week_index=train[DECISION_SESSION],
        )

        predictions = model.predict(test[feature_columns])
        block = test.copy()
        block["prediction"] = predictions.to_numpy()
        block["fold"] = fold.index
        frames.append(block)

        fold_log.append(
            {
                "fold": fold.index,
                "train_rows": fold.n_train,
                "test_rows": fold.n_test,
                "purged": fold.purged,
                "embargoed": fold.embargoed,
            }
        )
        if verbose:
            log.info(fold.describe())

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out.attrs["folds"] = fold_log
    return out


class BacktestEngine:
    """Runs the weekly loop over a prediction frame."""

    def __init__(
        self,
        cfg: Config,
        *,
        constructor: PortfolioConstructor | None = None,
        cost_model: CostModel | None = None,
    ) -> None:
        self.cfg = cfg
        self.constructor = constructor or PortfolioConstructor.from_config(cfg)
        self.cost_model = cost_model or CostModel.from_config(cfg)

    def run(
        self,
        predictions: pd.DataFrame,
        bars: pd.DataFrame,
        calendar: TradingCalendar,
        *,
        name: str = "strategy",
        grid: list[WeeklyDecision] | None = None,
        eligibility: pd.DataFrame | None = None,
        caveats: list[str] | None = None,
    ) -> BacktestResult:
        result = BacktestResult(name=name, caveats=list(caveats or []))
        if predictions.empty:
            result.notes.append("no out-of-sample predictions to trade")
            return result.finalize()

        hold = self.cfg.calendar.hold_sessions
        grid = grid if grid is not None else calendar.weekly_grid(hold)
        grid_by_decision = {g.decision_session: g for g in grid}

        realised = forward_return_matrix(bars, grid)
        weekly_returns_wide = _weekly_return_panel(bars, calendar)
        stats = _rolling_stats(bars)

        previous = pd.Series(dtype=float)
        risk_state = RiskState()
        equity = 1.0

        rows: list[dict] = []
        weights_by_session: dict[date, dict[str, float]] = {}
        total_costs = CostBreakdown()

        for decision_session, block in predictions.groupby(DECISION_SESSION, sort=True):
            decision = grid_by_decision.get(decision_session)
            if decision is None or decision_session not in realised:
                continue

            scores = block.set_index(TICKER)["prediction"].astype(float)
            sectors = (
                block.set_index(TICKER)[SECTOR]
                if SECTOR in block.columns
                else pd.Series("Unknown", index=scores.index)
            )

            session_stats = stats.get(decision_session, {})
            volatility = session_stats.get("volatility", pd.Series(dtype=float))
            adv = session_stats.get("adv_notional", pd.Series(dtype=float))
            hl_range = session_stats.get("hl_range", pd.Series(dtype=float))

            eligible = _eligibility_for(eligibility, decision_session, scores.index)

            risk_state = update_kill_switch(
                risk_state,
                equity,
                scale_at=self.cfg.risk.drawdown_scale_at,
                flat_at=self.cfg.risk.drawdown_flat_at,
                cooldown_weeks=self.cfg.risk.cooldown_weeks,
                enabled=self.cfg.risk.enabled,
            )

            history = _return_history(weekly_returns_wide, decision_session, scores.index)

            portfolio = self.constructor.build(
                scores,
                decision_session=decision_session,
                entry_session=decision.entry_session,
                volatility=volatility,
                sectors=sectors,
                returns_history=history,
                previous_weights=previous.to_dict() if not previous.empty else None,
                risk_scale=risk_state.scale,
                eligible=eligible,
            )

            target = pd.Series(portfolio.weights, dtype=float)

            cost, turnover = self.cost_model.rebalance_cost(
                target,
                previous,
                equity=equity,
                volatility=volatility,
                adv_notional=adv,
                high_low_range=hl_range,
                sessions_held=hold,
            )
            total_costs = total_costs + cost
            cost_fraction = cost.total / equity if equity > 0 else 0.0

            period_returns = realised[decision_session]
            gross_return = float(
                (target * period_returns.reindex(target.index).fillna(0.0)).sum()
            )
            net_return = gross_return - cost_fraction
            equity *= 1.0 + net_return

            rows.append(
                {
                    DECISION_SESSION: decision_session,
                    "gross_return": gross_return,
                    "cost": cost_fraction,
                    "net_return": net_return,
                    "turnover": turnover,
                    "gross_exposure": portfolio.gross,
                    "net_exposure": portfolio.net,
                    "risk_scale": risk_state.scale,
                    "n_positions": len(portfolio.positions),
                }
            )
            weights_by_session[decision_session] = dict(target)
            previous = target

        if not rows:
            result.notes.append("no tradeable weeks")
            return result.finalize()

        frame = pd.DataFrame(rows).set_index(DECISION_SESSION).sort_index()
        result.returns = frame["net_return"]
        result.gross_returns = frame["gross_return"]
        result.costs = frame["cost"]
        result.turnover = frame["turnover"]
        result.gross_exposure = frame["gross_exposure"]
        result.net_exposure = frame["net_exposure"]
        result.risk_scale = frame["risk_scale"]
        result.weights = weights_by_session
        result.predictions = predictions
        result.cost_breakdown = total_costs.to_dict()
        result.notes.extend(risk_state.events)

        return result.finalize()


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _weekly_return_panel(bars: pd.DataFrame, calendar: TradingCalendar) -> pd.DataFrame:
    """Wide frame of weekly close-to-close returns, for the covariance estimate."""
    from ..data.resample import to_weekly

    weekly = to_weekly(bars, calendar)
    if weekly.empty:
        return pd.DataFrame()
    wide = weekly.pivot_table(index="session", columns=TICKER, values="close", aggfunc="last")
    return wide.sort_index().pct_change()


def _return_history(
    weekly_returns: pd.DataFrame,
    decision_session: date,
    tickers: pd.Index,
    *,
    lookback: int = 104,
) -> pd.DataFrame:
    """Trailing weekly returns up to and including the decision date.

    Strictly ``<=`` the decision session, so the covariance used to size this week's book
    never contains a week that has not happened.
    """
    if weekly_returns.empty:
        return pd.DataFrame()
    history = weekly_returns.loc[weekly_returns.index <= decision_session]
    if history.empty:
        return pd.DataFrame()
    columns = [t for t in tickers if t in history.columns]
    return history.tail(lookback)[columns]


def _rolling_stats(bars: pd.DataFrame) -> dict[date, dict[str, pd.Series]]:
    """Per-session volatility, turnover and range, all shifted to stay causal."""
    if bars.empty:
        return {}

    frame = bars.sort_values([TICKER, "session"]).copy()
    log_close = np.log(frame["close"].astype(float).where(lambda s: s > 0))
    frame["_ret"] = log_close - log_close.groupby(frame[TICKER], sort=False).shift(1)

    grouped = frame.groupby(TICKER, sort=False)
    frame["_vol"] = grouped["_ret"].transform(
        lambda s: s.rolling(65, min_periods=30).std().shift(1)
    ) * np.sqrt(252.0)
    notional = frame["close"] * frame["volume"]
    frame["_adv"] = notional.groupby(frame[TICKER], sort=False).transform(
        lambda s: s.rolling(20, min_periods=10).mean().shift(1)
    )
    hl = (frame["high"] - frame["low"]) / frame["close"].replace(0.0, np.nan)
    frame["_hl"] = hl.groupby(frame[TICKER], sort=False).transform(
        lambda s: s.rolling(20, min_periods=10).mean().shift(1)
    )

    out: dict[date, dict[str, pd.Series]] = {}
    for session, block in frame.groupby("session", sort=True):
        indexed = block.set_index(TICKER)
        out[session] = {
            "volatility": indexed["_vol"].dropna(),
            "adv_notional": indexed["_adv"].dropna(),
            "hl_range": indexed["_hl"].dropna(),
        }
    return out


def _eligibility_for(
    eligibility: pd.DataFrame | None, session: date, tickers: pd.Index
) -> pd.Series | None:
    if eligibility is None or eligibility.empty:
        return None
    block = eligibility.loc[eligibility["session"] == session]
    if block.empty:
        return None
    return block.set_index(TICKER)["liquid"].reindex(tickers).fillna(False)

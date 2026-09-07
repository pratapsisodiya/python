"""News features: article-level analyses aggregated to one row per ticker-week.

This is where the point-in-time discipline of the news path is enforced, and it is worth
being precise about the mechanism because the obvious implementation is wrong.

The tempting approach is to join news to bars on date. That silently pulls a story
published at 18:00 on Friday into a decision taken at Friday's close, because both carry
the same date. Every such article is one the strategy could not have read, and they are
disproportionately the important ones: companies announce after the close.

So the join is on **timestamps**, not dates, and it is strict. For a decision at instant
``T``, an article counts only when ``available_at < T``. An article stamped exactly at the
close is excluded, on the grounds that nobody reads and acts on a story in the same
instant it appears. That boundary has its own test.

The features themselves aim at what the literature finds predictive at a weekly horizon:

* **Decay-weighted sentiment** — a story from this morning matters more than one from last
  Tuesday.
* **Abnormal news volume** — how much coverage a name is getting relative to its own
  normal level. Often more predictive than tone, because attention itself moves prices.
* **Novelty-weighted tone** — genuinely new information, distinguished from syndication.
* **Dispersion** — disagreement across stories, which behaves differently from consensus.
* **Event-type exposure** — a handful of specific event families, kept few because each
  one is a parameter fitted on very little data.
"""

from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pandas as pd

from ..calendars import TradingCalendar, WeeklyDecision
from ..types import DECISION_SESSION, NEWS_FEATURE_PREFIX, TICKER

log = logging.getLogger(__name__)

#: Event families given their own exposure feature. Deliberately few: each is a parameter
#: fitted on a small sample, and twenty sparse dummies would be twenty ways to overfit.
EVENT_GROUPS: dict[str, tuple[str, ...]] = {
    "earnings": ("earnings_beat", "earnings_miss", "earnings_inline"),
    "guidance": ("guidance_up", "guidance_down"),
    "corporate_action": ("m_and_a_target", "m_and_a_acquirer", "buyback", "dilution"),
    "regulatory": ("regulatory_approval", "regulatory_action", "litigation"),
    "analyst": ("analyst_action", "credit_rating"),
    "operational": ("contract_win", "product_launch", "operational_disruption"),
}

NEWS_FEATURE_NAMES: tuple[str, ...] = (
    f"{NEWS_FEATURE_PREFIX}sentiment_decay",
    f"{NEWS_FEATURE_PREFIX}score_decay",
    f"{NEWS_FEATURE_PREFIX}sentiment_max_abs",
    f"{NEWS_FEATURE_PREFIX}count",
    f"{NEWS_FEATURE_PREFIX}count_z",
    f"{NEWS_FEATURE_PREFIX}novelty_weighted",
    f"{NEWS_FEATURE_PREFIX}dispersion",
    f"{NEWS_FEATURE_PREFIX}max_magnitude",
    f"{NEWS_FEATURE_PREFIX}days_since_material",
    f"{NEWS_FEATURE_PREFIX}speculative_frac",
    f"{NEWS_FEATURE_PREFIX}recap_frac",
    f"{NEWS_FEATURE_PREFIX}surprise",
) + tuple(f"{NEWS_FEATURE_PREFIX}evt_{group}" for group in EVENT_GROUPS)


def build_news_features(
    analyses: pd.DataFrame,
    calendar: TradingCalendar,
    *,
    grid: list[WeeklyDecision] | None = None,
    hold_sessions: int = 5,
    lookback_days: int = 10,
    half_life_hours: float = 36.0,
    peer_weight: float = 0.3,
    speculative_discount: float = 0.5,
    baseline_weeks: int = 26,
) -> pd.DataFrame:
    """One row per ``(decision_session, ticker)`` of news features.

    ``analyses`` is the frame from :func:`swingbot.news.analyze.records_to_frame`, and
    must carry a timezone-aware ``available_at``.
    """
    if analyses.empty:
        return pd.DataFrame()

    grid = grid if grid is not None else calendar.weekly_grid(hold_sessions)
    if not grid:
        return pd.DataFrame()

    frame = analyses.copy()
    frame["available_at"] = pd.to_datetime(frame["available_at"], utc=True)
    frame = frame.sort_values("available_at")

    lookback = pd.Timedelta(days=lookback_days)
    rows: list[dict] = []

    for decision in grid:
        cutoff = decision.decision_ts
        window_start = cutoff - lookback

        # Strictly before the cutoff. An article stamped exactly at the close was not
        # readable in time to act on that close.
        visible = frame.loc[
            (frame["available_at"] >= window_start) & (frame["available_at"] < cutoff)
        ]
        if visible.empty:
            continue

        age_hours = (cutoff - visible["available_at"]).dt.total_seconds() / 3600.0
        decay = np.power(0.5, age_hours / max(half_life_hours, 1e-6))

        weight = decay.to_numpy(dtype=float)
        weight = weight * np.where(visible["is_company_specific"].to_numpy(), 1.0, peer_weight)
        weight = weight * np.where(
            visible["is_speculative"].to_numpy(), speculative_discount, 1.0
        )
        weight = weight * visible["novelty"].to_numpy(dtype=float).clip(0.05, 1.0)

        block = visible.assign(_w=weight)
        for ticker, group in block.groupby(TICKER, sort=False):
            rows.append(
                _aggregate_one(ticker, decision.decision_session, group)
            )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out = _add_abnormal_volume(out, baseline_weeks=baseline_weeks)
    out = _add_days_since_material(out, frame, grid)
    for column in NEWS_FEATURE_NAMES:
        if column not in out.columns:
            out[column] = np.nan
    return out[[DECISION_SESSION, TICKER, *NEWS_FEATURE_NAMES]].reset_index(drop=True)


def _aggregate_one(ticker: str, decision_session: date, group: pd.DataFrame) -> dict:
    weights = group["_w"].to_numpy(dtype=float)
    total = float(weights.sum())
    sentiment = group["sentiment"].to_numpy(dtype=float)
    score = group["signed_score"].to_numpy(dtype=float)
    novelty = group["novelty"].to_numpy(dtype=float)

    if total <= 1e-9:
        weighted_sentiment = float(np.mean(sentiment)) if len(sentiment) else 0.0
        weighted_score = float(np.mean(score)) if len(score) else 0.0
    else:
        weighted_sentiment = float(np.sum(sentiment * weights) / total)
        weighted_score = float(np.sum(score * weights) / total)

    row: dict = {
        DECISION_SESSION: decision_session,
        TICKER: ticker,
        f"{NEWS_FEATURE_PREFIX}sentiment_decay": weighted_sentiment,
        f"{NEWS_FEATURE_PREFIX}score_decay": weighted_score,
        f"{NEWS_FEATURE_PREFIX}sentiment_max_abs": float(np.max(np.abs(sentiment)))
        if len(sentiment)
        else 0.0,
        f"{NEWS_FEATURE_PREFIX}count": float(len(group)),
        f"{NEWS_FEATURE_PREFIX}novelty_weighted": float(np.sum(score * novelty) / max(len(group), 1)),
        # Disagreement across stories. Distinct from tone: a name with five bullish and
        # five bearish articles is not the same as one with ten neutral ones, and the
        # mean cannot tell them apart.
        f"{NEWS_FEATURE_PREFIX}dispersion": float(np.std(sentiment)) if len(sentiment) > 1 else 0.0,
        f"{NEWS_FEATURE_PREFIX}max_magnitude": float(group["magnitude"].max()),
        f"{NEWS_FEATURE_PREFIX}speculative_frac": float(group["is_speculative"].mean()),
        f"{NEWS_FEATURE_PREFIX}recap_frac": float(group["is_recap"].mean()),
    }

    surprise = group["numeric_surprise_pct"].dropna()
    row[f"{NEWS_FEATURE_PREFIX}surprise"] = (
        float(surprise.mean()) if len(surprise) else np.nan
    )

    event_types = group["event_type"]
    for name, members in EVENT_GROUPS.items():
        mask = event_types.isin(members)
        # Signed exposure, not a count: an earnings beat and an earnings miss are both
        # "earnings" but point in opposite directions, and a plain dummy would net them
        # to the same value.
        row[f"{NEWS_FEATURE_PREFIX}evt_{name}"] = (
            float(np.sum(score[mask.to_numpy()])) if bool(mask.any()) else 0.0
        )
    return row


def _add_abnormal_volume(out: pd.DataFrame, *, baseline_weeks: int) -> pd.DataFrame:
    """Article count relative to the ticker's own trailing normal.

    Shifted by one week so the current week's count is not part of the baseline it is
    compared against. Abnormal attention is one of the more reliable news effects, but
    only when "abnormal" is measured against a past that excludes the present.
    """
    count_col = f"{NEWS_FEATURE_PREFIX}count"
    frame = out.sort_values([TICKER, DECISION_SESSION]).copy()
    grouped = frame.groupby(TICKER, sort=False)[count_col]
    mean = grouped.transform(
        lambda s: s.rolling(baseline_weeks, min_periods=6).mean().shift(1)
    )
    std = grouped.transform(
        lambda s: s.rolling(baseline_weeks, min_periods=6).std().shift(1)
    )
    frame[f"{NEWS_FEATURE_PREFIX}count_z"] = (frame[count_col] - mean) / std.replace(0.0, np.nan)
    return frame


def _add_days_since_material(
    out: pd.DataFrame, analyses: pd.DataFrame, grid: list[WeeklyDecision]
) -> pd.DataFrame:
    """Sessions since the last materially important story for each name.

    A stock that has had no real news for two months behaves differently from one in the
    middle of an event, independently of what the news said.
    """
    column = f"{NEWS_FEATURE_PREFIX}days_since_material"
    material = analyses.loc[
        (analyses["magnitude"] >= 2) & (~analyses["is_recap"].astype(bool))
    ]
    if material.empty:
        out[column] = np.nan
        return out

    # Kept as tz-aware DatetimeIndex throughout. Converting to np.datetime64 silently
    # drops the timezone, and a naive-versus-aware comparison against the decision
    # instant is precisely the kind of off-by-a-session error this module exists to
    # prevent.
    by_ticker = {
        ticker: pd.DatetimeIndex(group["available_at"]).sort_values()
        for ticker, group in material.groupby(TICKER, sort=False)
    }
    cutoffs = {d.decision_session: d.decision_ts for d in grid}

    values = []
    for ticker, session in zip(out[TICKER], out[DECISION_SESSION], strict=True):
        cutoff = cutoffs.get(session)
        stamps = by_ticker.get(ticker)
        if cutoff is None or stamps is None:
            values.append(np.nan)
            continue
        prior = stamps[stamps < cutoff]
        if len(prior) == 0:
            values.append(np.nan)
            continue
        delta = (cutoff - prior[-1]).total_seconds() / 86400.0
        values.append(min(delta, 180.0))
    out[column] = values
    return out


def news_coverage_report(features: pd.DataFrame, panel: pd.DataFrame) -> dict:
    """How much of the panel actually carries news. Reported in the tearsheet.

    The single most important number for interpreting any news result: a Sharpe
    improvement built on 4 percent coverage is a claim about 4 percent of the book.
    """
    if features.empty or panel.empty:
        return {"coverage": 0.0, "n_ticker_weeks": 0, "median_articles": 0.0}
    keyed = set(zip(features[DECISION_SESSION], features[TICKER], strict=True))
    panel_keys = list(zip(panel[DECISION_SESSION], panel[TICKER], strict=True))
    covered = sum(1 for k in panel_keys if k in keyed)
    return {
        "coverage": round(covered / max(len(panel_keys), 1), 4),
        "n_ticker_weeks": len(keyed),
        "median_articles": float(features[f"{NEWS_FEATURE_PREFIX}count"].median()),
    }

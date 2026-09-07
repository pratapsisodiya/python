"""Trial ledger.

Every backtest ever run is recorded here with its config hash, git revision and result.
The ledger exists for one reason: the deflated Sharpe ratio needs an honest count of how
many configurations were tried, and nobody remembers that number accurately. People
remember the six ideas they thought were good and forget the ninety they tuned away.

Recording it automatically removes the temptation.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trials (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       TEXT,
    market       TEXT NOT NULL,
    config_hash  TEXT NOT NULL,
    git_revision TEXT,
    variant      TEXT NOT NULL DEFAULT 'main',
    sharpe       REAL,
    ann_return   REAL,
    max_drawdown REAL,
    turnover     REAL,
    n_periods    INTEGER,
    notes        TEXT,
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_trials_market ON trials(market);
CREATE INDEX IF NOT EXISTS idx_trials_config ON trials(config_hash);
"""


@dataclass(slots=True)
class Trial:
    market: str
    config_hash: str
    variant: str = "main"
    run_id: str = ""
    git_revision: str = ""
    sharpe: float = 0.0
    ann_return: float = 0.0
    max_drawdown: float = 0.0
    turnover: float = 0.0
    n_periods: int = 0
    notes: str = ""


class TrialLedger:
    """Append-only record of backtests, keyed by market."""

    __slots__ = ("path", "_conn")

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> TrialLedger:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

    def record(self, trial: Trial) -> int:
        cursor = self._conn.execute(
            """
            INSERT INTO trials (run_id, market, config_hash, git_revision, variant,
                                sharpe, ann_return, max_drawdown, turnover, n_periods,
                                notes, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                trial.run_id,
                trial.market,
                trial.config_hash,
                trial.git_revision,
                trial.variant,
                float(trial.sharpe),
                float(trial.ann_return),
                float(trial.max_drawdown),
                float(trial.turnover),
                int(trial.n_periods),
                trial.notes,
                datetime.now(UTC).isoformat(),
            ),
        )
        self._conn.commit()
        return int(cursor.lastrowid or 0)

    def count(self, market: str | None = None, *, variant: str = "main") -> int:
        """Distinct configurations tried. This is the ``n`` in the deflated Sharpe.

        Counts distinct config hashes rather than rows, so re-running the same
        configuration does not inflate the penalty.
        """
        if market:
            row = self._conn.execute(
                "SELECT COUNT(DISTINCT config_hash) AS n FROM trials "
                "WHERE market = ? AND variant = ?",
                (market, variant),
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT COUNT(DISTINCT config_hash) AS n FROM trials WHERE variant = ?",
                (variant,),
            ).fetchone()
        return max(1, int(row["n"]))

    def sharpe_variance(self, market: str | None = None, *, variant: str = "main") -> float:
        """Variance of Sharpe across trials, an input to the deflation."""
        frame = self.list(market, variant=variant)
        if len(frame) < 3:
            return 1.0
        variance = float(frame["sharpe"].var(ddof=1))
        return variance if np.isfinite(variance) and variance > 1e-9 else 1.0

    def list(
        self, market: str | None = None, *, variant: str | None = None, limit: int = 500
    ) -> pd.DataFrame:
        query = "SELECT * FROM trials WHERE 1=1"
        params: list = []
        if market:
            query += " AND market = ?"
            params.append(market)
        if variant:
            query += " AND variant = ?"
            params.append(variant)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(query, params).fetchall()
        return pd.DataFrame([dict(r) for r in rows])

    def best(self, market: str, *, variant: str = "main") -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM trials WHERE market = ? AND variant = ? "
            "ORDER BY sharpe DESC LIMIT 1",
            (market, variant),
        ).fetchone()
        return dict(row) if row else None

    def summary(self, market: str | None = None) -> dict:
        frame = self.list(market, limit=10_000)
        if frame.empty:
            return {"n_trials": 0, "n_configs": 0}
        main = frame.loc[frame["variant"] == "main"] if "variant" in frame else frame
        return {
            "n_trials": int(len(frame)),
            "n_configs": int(frame["config_hash"].nunique()),
            "best_sharpe": float(main["sharpe"].max()) if len(main) else 0.0,
            "median_sharpe": float(main["sharpe"].median()) if len(main) else 0.0,
            "sharpe_variance": self.sharpe_variance(market),
        }

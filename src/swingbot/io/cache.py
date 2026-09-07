"""Content-addressed sqlite cache.

Used for news analyses, which are the one genuinely expensive artefact in the system.
The key is a hash of the content plus everything that could change the answer: the
prompt version, the schema version, the backend and the model id. That has two
consequences worth stating plainly.

Changing a prompt does not rewrite history. Old records stay under their old key and new
ones are written alongside, so a backtest pinned to ``prompt_version: v1`` keeps
returning what it always returned.

The cache stores ``published_at`` from the article and ``analyzed_at`` from the wall
clock, and only the former is ever allowed to reach a feature. That is what lets an
analysis run today feed a 2019 backtest without contaminating it.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS analysis_cache (
    cache_key     TEXT PRIMARY KEY,
    content_hash  TEXT NOT NULL,
    backend       TEXT NOT NULL,
    model_id      TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    payload       TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'ok',
    analyzed_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_analysis_content ON analysis_cache(content_hash);
CREATE INDEX IF NOT EXISTS idx_analysis_status ON analysis_cache(status);

CREATE TABLE IF NOT EXISTS spend_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    backend       TEXT NOT NULL,
    model_id      TEXT NOT NULL,
    n_articles    INTEGER NOT NULL,
    usd           REAL NOT NULL,
    at            TEXT NOT NULL
);
"""


def content_hash(text: str) -> str:
    """Stable hash of article text, used as the identity of an article."""
    normalised = " ".join(text.split()).lower()
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def cache_key(
    content_hash_value: str,
    *,
    backend: str,
    model_id: str,
    prompt_version: str,
    schema_version: str,
) -> str:
    raw = f"{content_hash_value}|{backend}|{model_id}|{prompt_version}|{schema_version}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


class ContentCache:
    """A tiny sqlite cache. Safe to delete; expensive to rebuild."""

    __slots__ = ("path", "_conn")

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        # WAL keeps a long analysis run from blocking a concurrent read.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> ContentCache:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

    # ------------------------------------------------------------------------ reads

    def get(self, key: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT payload, status FROM analysis_cache WHERE cache_key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        if row["status"] != "ok":
            return None
        return json.loads(row["payload"])

    def get_many(self, keys: Iterable[str]) -> dict[str, dict[str, Any]]:
        keys = list(keys)
        if not keys:
            return {}
        out: dict[str, dict[str, Any]] = {}
        chunk = 500
        for i in range(0, len(keys), chunk):
            batch = keys[i : i + chunk]
            marks = ",".join("?" * len(batch))
            rows = self._conn.execute(
                f"SELECT cache_key, payload, status FROM analysis_cache WHERE cache_key IN ({marks})",
                batch,
            ).fetchall()
            for row in rows:
                if row["status"] == "ok":
                    out[row["cache_key"]] = json.loads(row["payload"])
        return out

    def has_failure(self, key: str) -> bool:
        row = self._conn.execute(
            "SELECT status FROM analysis_cache WHERE cache_key = ?", (key,)
        ).fetchone()
        return row is not None and row["status"] != "ok"

    # ----------------------------------------------------------------------- writes

    def put(
        self,
        key: str,
        payload: dict[str, Any],
        *,
        content_hash_value: str,
        backend: str,
        model_id: str,
        prompt_version: str,
        schema_version: str,
        status: str = "ok",
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO analysis_cache
                (cache_key, content_hash, backend, model_id, prompt_version,
                 schema_version, payload, status, analyzed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(cache_key) DO UPDATE SET
                payload=excluded.payload,
                status=excluded.status,
                analyzed_at=excluded.analyzed_at
            """,
            (
                key,
                content_hash_value,
                backend,
                model_id,
                prompt_version,
                schema_version,
                json.dumps(payload, default=str),
                status,
                datetime.now(UTC).isoformat(),
            ),
        )
        self._conn.commit()

    def record_failure(self, key: str, reason: str, **meta: str) -> None:
        """Record a failure explicitly rather than silently defaulting.

        A missing analysis becomes a null news feature downstream, which is honest. A
        silently defaulted neutral analysis would look like real evidence of neutrality.
        """
        self.put(
            key,
            {"error": reason},
            content_hash_value=meta.get("content_hash_value", ""),
            backend=meta.get("backend", ""),
            model_id=meta.get("model_id", ""),
            prompt_version=meta.get("prompt_version", ""),
            schema_version=meta.get("schema_version", ""),
            status="failed",
        )

    def log_spend(self, backend: str, model_id: str, n_articles: int, usd: float) -> None:
        self._conn.execute(
            "INSERT INTO spend_log (backend, model_id, n_articles, usd, at) VALUES (?,?,?,?,?)",
            (backend, model_id, n_articles, usd, datetime.now(UTC).isoformat()),
        )
        self._conn.commit()

    # ------------------------------------------------------------------------ stats

    def stats(self) -> dict[str, Any]:
        total = self._conn.execute("SELECT COUNT(*) AS n FROM analysis_cache").fetchone()["n"]
        ok = self._conn.execute(
            "SELECT COUNT(*) AS n FROM analysis_cache WHERE status='ok'"
        ).fetchone()["n"]
        spend = self._conn.execute(
            "SELECT COALESCE(SUM(usd), 0.0) AS s FROM spend_log"
        ).fetchone()["s"]
        by_backend = {
            row["backend"]: row["n"]
            for row in self._conn.execute(
                "SELECT backend, COUNT(*) AS n FROM analysis_cache GROUP BY backend"
            ).fetchall()
        }
        return {
            "entries": int(total),
            "ok": int(ok),
            "failed": int(total) - int(ok),
            "total_spend_usd": float(spend),
            "by_backend": by_backend,
        }

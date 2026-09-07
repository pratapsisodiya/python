"""Run artefacts.

Every command that produces a result writes a directory under ``runs/`` containing the
fully resolved config, the git revision, a manifest of every input file's content hash,
and the outputs. That is the difference between "the backtest said 1.4" and "this exact
code, on these exact bytes, with this exact config, said 1.4".
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .store import read_json, write_json


def _git_revision() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        rev = out.stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        ).stdout.strip()
        return f"{rev}{'-dirty' if dirty else ''}" if rev else "unknown"
    except Exception:
        return "unknown"


@dataclass(slots=True)
class RunContext:
    """A single run's output directory and metadata."""

    run_id: str
    directory: Path
    started_at: datetime
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        runs_dir: Path | str,
        *,
        command: str,
        market: str,
        config_hash: str,
        config_yaml: str = "",
        extra: dict[str, Any] | None = None,
    ) -> RunContext:
        started = datetime.now(UTC)
        run_id = f"{started.strftime('%Y%m%dT%H%M%S')}-{market}-{command}-{config_hash[:8]}"
        directory = Path(runs_dir) / run_id
        directory.mkdir(parents=True, exist_ok=True)

        meta = {
            "run_id": run_id,
            "command": command,
            "market": market,
            "config_hash": config_hash,
            "git_revision": _git_revision(),
            "started_at": started.isoformat(),
            **(extra or {}),
        }
        write_json(directory / "run.json", meta)
        if config_yaml:
            (directory / "config.yaml").write_text(config_yaml)
        return cls(run_id=run_id, directory=directory, started_at=started, meta=meta)

    # -------------------------------------------------------------------- artefacts

    def path(self, name: str) -> Path:
        return self.directory / name

    def write_frame(self, frame: pd.DataFrame, name: str, *, index: bool = False) -> Path:
        target = self.directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if name.endswith(".csv"):
            frame.to_csv(target, index=index)
        else:
            frame.to_parquet(target, index=index)
        return target

    def write_json(self, payload: Any, name: str) -> Path:
        return write_json(self.directory / name, payload)

    def write_text(self, text: str, name: str) -> Path:
        target = self.directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        return target

    def record_manifest(self, manifest: dict[str, str]) -> Path:
        return write_json(self.directory / "manifest.json", manifest)

    def finish(self, summary: dict[str, Any] | None = None) -> Path:
        finished = datetime.now(UTC)
        self.meta["finished_at"] = finished.isoformat()
        self.meta["duration_seconds"] = round(
            (finished - self.started_at).total_seconds(), 3
        )
        if summary:
            self.meta["summary"] = summary
        return write_json(self.directory / "run.json", self.meta)


def list_runs(runs_dir: Path | str, limit: int = 20) -> list[dict[str, Any]]:
    base = Path(runs_dir)
    if not base.exists():
        return []
    entries = []
    for child in sorted(base.iterdir(), reverse=True):
        if not child.is_dir():
            continue
        meta = read_json(child / "run.json")
        if meta:
            entries.append(meta)
        if len(entries) >= limit:
            break
    return entries


def find_run(runs_dir: Path | str, run_id: str) -> Path | None:
    """Locate a run directory by exact id or unique prefix."""
    base = Path(runs_dir)
    exact = base / run_id
    if exact.is_dir():
        return exact
    matches = [c for c in base.glob(f"{run_id}*") if c.is_dir()] if base.exists() else []
    if len(matches) == 1:
        return matches[0]
    return None


def latest_run(runs_dir: Path | str, command: str | None = None) -> Path | None:
    base = Path(runs_dir)
    if not base.exists():
        return None
    for child in sorted(base.iterdir(), reverse=True):
        if not child.is_dir():
            continue
        if command is None:
            return child
        meta = read_json(child / "run.json", {})
        if meta.get("command") == command:
            return child
    return None

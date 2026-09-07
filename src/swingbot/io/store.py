"""Parquet lake with atomic writes and content manifests.

Raw data is immutable and append-only. Everything downstream is derivable and cheap to
rebuild, with one exception: the news-analysis cache, which is the expensive asset and
lives in :mod:`swingbot.io.cache`.

Writes go to a temporary file and are then renamed, so an interrupted run leaves the
previous good file in place rather than a half-written one that reads as valid parquet
with missing rows.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd


class ParquetStore:
    """A small, explicit parquet lake rooted at ``root``."""

    __slots__ = ("root",)

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------------ paths

    def path(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    def exists(self, *parts: str) -> bool:
        return self.path(*parts).exists()

    # ------------------------------------------------------------------------ write

    def write(self, frame: pd.DataFrame, *parts: str, index: bool = False) -> Path:
        """Atomically write a frame to ``root/parts``."""
        target = self.path(*parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), suffix=".parquet.tmp")
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            frame.to_parquet(tmp, index=index)
            tmp.replace(target)
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
        return target

    def append(self, frame: pd.DataFrame, *parts: str, dedupe_on: list[str] | None = None) -> Path:
        """Append rows, optionally dropping duplicates on a key.

        Read-modify-write rather than a partitioned append, which is the right trade at
        this data scale: a weekly universe of a few hundred names over a decade is tens
        of megabytes, and a single file is far easier to reason about.
        """
        target = self.path(*parts)
        if target.exists():
            existing = pd.read_parquet(target)
            combined = pd.concat([existing, frame], ignore_index=True)
        else:
            combined = frame.reset_index(drop=True)
        if dedupe_on:
            combined = combined.drop_duplicates(subset=dedupe_on, keep="last")
        return self.write(combined.reset_index(drop=True), *parts)

    # ------------------------------------------------------------------------- read

    def read(self, *parts: str, columns: list[str] | None = None) -> pd.DataFrame:
        target = self.path(*parts)
        if not target.exists():
            return pd.DataFrame()
        return pd.read_parquet(target, columns=columns)

    def read_many(self, pattern: str, *, subdir: str = "") -> pd.DataFrame:
        base = self.root / subdir if subdir else self.root
        files = sorted(base.glob(pattern))
        if not files:
            return pd.DataFrame()
        frames = [pd.read_parquet(f) for f in files]
        return pd.concat(frames, ignore_index=True)

    def delete(self, *parts: str) -> None:
        self.path(*parts).unlink(missing_ok=True)

    # --------------------------------------------------------------------- manifest

    def manifest(self, paths: Iterable[Path] | None = None) -> dict[str, str]:
        """Content hash of every parquet file under the root.

        Recorded with each run so a result can be tied to the exact bytes that produced
        it. Reproducibility is then a diff, not a recollection.
        """
        out: dict[str, str] = {}
        candidates = list(paths) if paths is not None else sorted(self.root.rglob("*.parquet"))
        for file in candidates:
            if not file.is_file():
                continue
            digest = hashlib.sha256()
            with file.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    digest.update(chunk)
            rel = file.relative_to(self.root).as_posix()
            out[rel] = digest.hexdigest()[:16]
        return out


def write_json(path: Path, payload: Any) -> Path:
    """Atomically write JSON, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".json.tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    return path


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text())

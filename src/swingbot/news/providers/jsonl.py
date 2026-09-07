"""Replayable JSONL news provider.

The provider a backtest actually uses. RSS only returns recent items, so a historical
news corpus has to come from a vendor export, an archive, or accumulated live runs —
whatever the source, dumping it to JSON Lines makes it replayable, diffable and
independent of whether the original source still exists.

Expected fields per line: ``ticker, title, published_at``, plus optional ``body, url,
source, first_seen_at``. A missing ``first_seen_at`` defaults to ``published_at``, which
is the optimistic assumption; where a vendor is known to backdate, set it explicitly.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path

from ...types import RawArticle
from ..dedupe import article_hash

log = logging.getLogger(__name__)


def _parse_ts(value, fallback: datetime | None = None) -> datetime:
    if value is None:
        if fallback is None:
            raise ValueError("missing timestamp and no fallback")
        return fallback
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), UTC)
    text = str(value).strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


class JSONLNewsProvider:
    """Reads a news corpus from JSON Lines."""

    name = "jsonl"

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def available(self) -> bool:
        return self.path.exists()

    def fetch(
        self, tickers: Sequence[str], start: datetime, end: datetime
    ) -> Iterator[RawArticle]:
        if not self.path.exists():
            log.warning("news corpus not found at %s", self.path)
            return

        wanted = set(tickers)
        skipped = 0
        with self.path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue

                ticker = str(record.get("ticker", "")).strip()
                if not ticker or (wanted and ticker not in wanted):
                    continue

                try:
                    published = _parse_ts(record.get("published_at"))
                except Exception:
                    skipped += 1
                    continue
                if not (start <= published <= end):
                    continue

                first_seen = _parse_ts(record.get("first_seen_at"), published)
                title = str(record.get("title", "")).strip()
                body = str(record.get("body", "") or record.get("description", "")).strip()
                if not title and not body:
                    continue

                stub = RawArticle(
                    content_hash="",
                    ticker=ticker,
                    title=title,
                    body=body,
                    url=str(record.get("url", "")),
                    source=str(record.get("source", "jsonl")),
                    published_at=published,
                    first_seen_at=first_seen,
                )
                yield RawArticle(
                    content_hash=record.get("content_hash") or article_hash(stub),
                    ticker=ticker,
                    title=title,
                    body=body,
                    url=str(record.get("url", "")),
                    source=str(record.get("source", "jsonl")),
                    published_at=published,
                    first_seen_at=first_seen,
                )

        if skipped:
            log.warning("skipped %d malformed line(s) in %s", skipped, self.path)


def write_jsonl(articles, path: Path | str) -> Path:
    """Persist articles as JSON Lines, for replay and for sharing a corpus."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for a in articles:
            fh.write(
                json.dumps(
                    {
                        "content_hash": a.content_hash,
                        "ticker": a.ticker,
                        "title": a.title,
                        "body": a.body,
                        "url": a.url,
                        "source": a.source,
                        "published_at": a.published_at.isoformat(),
                        "first_seen_at": a.first_seen_at.isoformat(),
                    }
                )
                + "\n"
            )
    return path

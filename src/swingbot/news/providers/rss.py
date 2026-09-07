"""Google News RSS provider.

The one free news source that reliably works from anywhere, which is why it is the
default. It returns headlines and short descriptions rather than full article bodies, and
that limitation is worth being explicit about: headline-only text is enough for event
classification and sentiment direction, and not enough for extracting numeric surprises.
The coverage report in ``swingbot doctor`` shows what fraction of ingested items are
headline-only so the limitation is visible rather than assumed away.

Two point-in-time properties matter here. RSS returns only recent items, typically the
last month, so it cannot build a historical corpus — it is for the live weekly run, and
history has to accumulate by running it regularly or come from a vendor. And every item
is stamped with both the feed's ``pubDate`` and the moment this pipeline saw it, with
availability taking the later of the two, so a feed that backdates its timestamps cannot
hand a backtest free information.
"""

from __future__ import annotations

import html
import logging
import re
import time
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

from ...types import RawArticle
from ..dedupe import article_hash

log = logging.getLogger(__name__)

_FEED = "https://news.google.com/rss/search"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}
_TAG = re.compile(r"<[^>]+>")
_ITEM = re.compile(r"<item>(.*?)</item>", re.DOTALL)


def _field(block: str, tag: str) -> str:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", block, re.DOTALL)
    if not match:
        return ""
    text = match.group(1).strip()
    if text.startswith("<![CDATA["):
        text = text[9:-3] if text.endswith("]]>") else text[9:]
    return html.unescape(_TAG.sub(" ", text)).strip()


class RSSNewsProvider:
    """Per-ticker Google News search feeds, locale-aware per market."""

    name = "rss"

    def __init__(
        self,
        *,
        hl: str = "en-US",
        gl: str = "US",
        ceid: str = "US:en",
        max_items_per_symbol: int = 50,
        request_delay_seconds: float = 0.5,
        timeout: float = 20.0,
        company_names: dict[str, str] | None = None,
        query_suffix: str = "stock",
    ) -> None:
        self.hl = hl
        self.gl = gl
        self.ceid = ceid
        self.max_items = max_items_per_symbol
        self.request_delay = request_delay_seconds
        self.timeout = timeout
        self.company_names = company_names or {}
        self.query_suffix = query_suffix

    def available(self) -> bool:
        try:
            import httpx  # noqa: F401
        except ImportError:
            return False
        return True

    def _query(self, ticker: str) -> str:
        # Searching the company name beats searching the ticker: "INFY" returns little,
        # "Infosys" returns the actual coverage. Falls back to the ticker when no name
        # is known.
        name = self.company_names.get(ticker, ticker)
        return f'"{name}" {self.query_suffix}'.strip()

    def fetch(
        self, tickers: Sequence[str], start: datetime, end: datetime
    ) -> Iterator[RawArticle]:
        try:
            import httpx
        except ImportError:
            return

        with httpx.Client(headers=_HEADERS, timeout=self.timeout, follow_redirects=True) as client:
            for i, ticker in enumerate(tickers):
                if i:
                    time.sleep(self.request_delay)
                try:
                    response = client.get(
                        _FEED,
                        params={
                            "q": self._query(ticker),
                            "hl": self.hl,
                            "gl": self.gl,
                            "ceid": self.ceid,
                        },
                    )
                    response.raise_for_status()
                except Exception as exc:
                    log.warning("news fetch failed for %s: %s", ticker, exc)
                    continue

                yield from self._parse(response.text, ticker, start, end)

    def _parse(
        self, xml: str, ticker: str, start: datetime, end: datetime
    ) -> Iterator[RawArticle]:
        seen_at = datetime.now(UTC)
        count = 0
        for block in _ITEM.findall(xml):
            if count >= self.max_items:
                break
            title = _field(block, "title")
            if not title:
                continue

            raw_date = _field(block, "pubDate")
            try:
                published = parsedate_to_datetime(raw_date)
                if published.tzinfo is None:
                    published = published.replace(tzinfo=UTC)
                published = published.astimezone(UTC)
            except Exception:
                published = seen_at

            if not (start <= published <= end):
                continue

            description = _field(block, "description")
            source = _field(block, "source") or "google-news"
            url = _field(block, "link")

            # Google appends " - Publisher" to headlines; strip it so the publisher name
            # does not leak into sentiment scoring.
            clean_title = re.sub(rf"\s+-\s+{re.escape(source)}\s*$", "", title).strip()

            article = RawArticle(
                content_hash="",
                ticker=ticker,
                title=clean_title,
                body=description,
                url=url,
                source=source,
                published_at=published,
                first_seen_at=seen_at,
            )
            yield RawArticle(
                content_hash=article_hash(article),
                ticker=ticker,
                title=clean_title,
                body=description,
                url=url,
                source=source,
                published_at=published,
                first_seen_at=seen_at,
            )
            count += 1

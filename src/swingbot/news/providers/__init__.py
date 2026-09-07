"""News sources."""

from .jsonl import JSONLNewsProvider, write_jsonl
from .rss import RSSNewsProvider

__all__ = ["JSONLNewsProvider", "RSSNewsProvider", "write_jsonl"]


def build_provider(cfg, *, company_names: dict[str, str] | None = None):
    """Construct the news provider named in ``cfg.news.provider``."""
    name = cfg.news.provider.strip().lower()
    if name == "rss":
        return RSSNewsProvider(
            hl=cfg.news.rss.hl,
            gl=cfg.news.rss.gl,
            ceid=cfg.news.rss.ceid,
            max_items_per_symbol=cfg.news.rss.max_items_per_symbol,
            request_delay_seconds=cfg.news.rss.request_delay_seconds,
            company_names=company_names,
        )
    if name == "jsonl":
        path = cfg.news.jsonl_path or (cfg.market_dir / "news" / "articles.jsonl")
        return JSONLNewsProvider(path)
    raise ValueError(f"Unknown news provider {cfg.news.provider!r}. Known: rss, jsonl")

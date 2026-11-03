"""TechCompass — 多源聚合器。

把多个 RealtimeSource 统一拉取 → 归一为 TechCompassDocument → 去重 → 汇总。

用于解决「数据源单一、样本数量太少」：
- 一次 query 同时打多个来源（GNews / Reddit / HackerNews / ...）
- 统一 schema，按 url / content_hash 去重
- 返回 per-source 计数与错误，便于判断「每个 query 能收集多少样本」

不做网页爬取，只用公开官方 API。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .realtime_base import RealtimeSource
from .unified_schema import TechCompassDocument


@dataclass
class FetchResult:
    query: str
    documents: list[TechCompassDocument] = field(default_factory=list)
    per_source: dict[str, int] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def total_raw(self) -> int:
        return sum(self.per_source.values())

    @property
    def total_deduped(self) -> int:
        return len(self.documents)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "total_raw": self.total_raw,
            "total_deduped": self.total_deduped,
            "per_source": self.per_source,
            "errors": self.errors,
            "documents": [d.to_dict() for d in self.documents],
        }


class MultiSourceFetcher:
    def __init__(self, sources: list[RealtimeSource]):
        self.sources = sources

    def search(self, query: str, limit_per_source: int = 10) -> FetchResult:
        result = FetchResult(query=query)
        seen: set[str] = set()
        for source in self.sources:
            raw = 0
            try:
                items = source.search(query, limit=limit_per_source)
                raw = len(items)
                for item in items:
                    doc = item.to_document()
                    key = doc.url or doc.content_hash
                    if key and key in seen:
                        continue
                    if key:
                        seen.add(key)
                    result.documents.append(doc)
            except Exception as exc:  # noqa: BLE001
                result.errors[source.name] = str(exc)
            result.per_source[source.name] = raw
        return result


def build_default_fetcher(env_path: str | None = None) -> MultiSourceFetcher:
    """构建默认多源聚合器（无需凭据的来源优先，GNews 有 key 才启用）。"""
    from .realtime_gnews import GNewsSource
    from .realtime_reddit import RedditSource
    from .realtime_hackernews import HackerNewsSource

    sources: list[RealtimeSource] = [HackerNewsSource()]
    gnews = GNewsSource(env_path=env_path) if env_path else GNewsSource()
    if gnews.validate_config():
        sources.append(gnews)
    sources.append(RedditSource())
    return MultiSourceFetcher(sources)

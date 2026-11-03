"""TechCompass — 实时公开数据源抽象接口。

所有实时来源（新闻 API / 讨论 / RSS）统一实现这个接口：
- name(): 来源标识
- search(query, limit): 按关键词取最近内容，返回 RealtimeItem
- 不做网页爬取，只用官方公开 API。

RealtimeItem 可通过 to_document() 转成统一 TechCompassDocument。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from .unified_schema import TechCompassDocument, make_document_id


@dataclass
class RealtimeItem:
    """统一的实时内容条目（新闻/讨论/RSS 都归一成这个）。"""

    id: str
    source: str
    title: str
    content: str
    url: str
    published_at: Optional[datetime] = None
    metadata: dict[str, Any] = None

    def to_document(self) -> TechCompassDocument:
        """转换为统一 TechCompassDocument。"""
        source_type = "news"
        if self.source in {"reddit", "reddit_rss", "rss", "forum", "hackernews"}:
            source_type = "discussion"
        return TechCompassDocument(
            id=make_document_id(self.source, self.id or self.url or self.title),
            source=self.source,
            source_type=source_type,
            title=self.title,
            content=self.content,
            url=self.url or None,
            published_at=self.published_at,
            language=(self.metadata or {}).get("language"),
            metadata=self.metadata or {},
            raw=self.to_dict(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "title": self.title,
            "content": self.content,
            "url": self.url,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "metadata": self.metadata or {},
        }


class RealtimeSource:
    """实时公开数据源的统一接口。"""

    name = "base"

    def search(self, query: str, limit: int = 10) -> list[RealtimeItem]:
        raise NotImplementedError

    def validate_config(self) -> bool:
        return False

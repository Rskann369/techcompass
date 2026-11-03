"""TechCompass — 统一数据模型（Item）。

Week 2 的核心地基之一：所有数据源（评价 / 新闻 / 工单）最终都会被归一化
成同一种 `Item` 结构，后续 Workflow 各节点只认这一个 Schema。

设计要点：
- 字段命名统一，避免不同来源字段名不一致导致下游出错。
- 用 datetime 统一时间（默认 UTC）。
- `raw` 字段保留原始数据，方便回溯和评测。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from .unified_schema import TechCompassDocument, make_document_id

# 数据来源类型
SOURCE_REVIEW = "review"
SOURCE_NEWS = "news"
SOURCE_TICKET = "ticket"
VALID_SOURCES = {SOURCE_REVIEW, SOURCE_NEWS, SOURCE_TICKET}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Item:
    """统一舆情条目。"""

    id: str
    source_type: str                # review / news / ticket
    product_category: str           # headphones / charger / watch ...
    product_name: str
    title: str
    content: str
    timestamp: datetime             # 原始时间
    rating: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)
    ingested_at: datetime = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        """序列化为可落盘的 dict（时间统一转 ISO）。"""
        return {
            "id": self.id,
            "source_type": self.source_type,
            "product_category": self.product_category,
            "product_name": self.product_name,
            "title": self.title,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "rating": self.rating,
            "metadata": self.metadata,
            "raw": self.raw,
            "ingested_at": self.ingested_at.isoformat(),
        }

    def to_document(self) -> TechCompassDocument:
        """转换为统一 TechCompassDocument。"""
        source = self.metadata.get("source", "local_review")
        return TechCompassDocument(
            id=make_document_id(source, self.id),
            source=source,
            source_type="review" if self.source_type == "review" else self.source_type,
            title=self.title,
            content=self.content,
            url=self.metadata.get("url"),
            published_at=self.timestamp,
            language=self.metadata.get("language"),
            metadata={
                "product_category": self.product_category,
                "product_name": self.product_name,
                "rating": self.rating,
                **self.metadata,
            },
            raw=self.to_dict(),
            ingested_at=self.ingested_at,
        )


def parse_timestamp(value: Any) -> datetime:
    """把各种来源的时间转成 UTC-aware datetime。"""
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, (int, float)):
        # 兼容 Unix 时间戳（秒或毫秒）
        if value > 10_000_000_000:  # 毫秒
            value = value / 1000
        dt = datetime.fromtimestamp(value, tz=timezone.utc)
    else:
        text = str(value).strip()
        # 处理常见格式：YYYY-MM-DD 或 YYYY-MM-DD HH:MM:SS
        text = text.replace("Z", "+00:00")
        for fmt in ("%Y-%m-%d %H:%M:%S%z", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(text, fmt)
                break
            except ValueError:
                continue
        else:
            # 解析失败时兜底为 now
            dt = _utcnow()

    # 统一转 UTC
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


if __name__ == "__main__":
    # 简单自检
    it = Item(
        id="x1",
        source_type=SOURCE_REVIEW,
        product_category="headphones",
        product_name="WireX",
        title="充电发热",
        content="充电烫",
        timestamp=parse_timestamp("2026-10-05"),
        rating=2,
    )
    print(it.to_dict())

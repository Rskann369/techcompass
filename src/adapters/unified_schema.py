"""TechCompass — 统一跨源数据 Schema。

目标：让 GNews / Reddit / 本地评论 / RAG document 使用同一套核心字段，
避免“每个适配器一套结构、RAG 再一套 dict”的碎片化状态。

统一模型：TechCompassDocument

核心字段：
- id：全局稳定 ID（source + external_id 推导）
- source：数据源标识
- source_type：news / discussion / review / ticket
- title / content / url / published_at / language
- metadata：源特有字段
- raw：原始响应保留
- ingested_at：入库时间
- content_hash：内容去重 hash
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

VALID_SOURCE_TYPES = {"news", "discussion", "review", "ticket", "unknown"}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def make_document_id(source: str, external_id: str) -> str:
    raw = f"{source}:{external_id}".encode("utf-8")
    return f"{source}_{hashlib.sha256(raw).hexdigest()[:20]}"


@dataclass
class TechCompassDocument:
    """TechCompass 全链路统一文档模型。"""

    id: str
    source: str
    source_type: str
    title: str
    content: str
    url: Optional[str] = None
    published_at: Optional[datetime] = None
    language: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)
    ingested_at: datetime = field(default_factory=utcnow)

    def __post_init__(self) -> None:
        if self.source_type not in VALID_SOURCE_TYPES:
            self.source_type = "unknown"
        if self.language is None:
            # 极简语言判断：中文/英文，其他置 unknown
            text = f"{self.title} {self.content}"
            if any("\u4e00" <= ch <= "\u9fff" for ch in text):
                self.language = "zh"
            elif any(ch.isascii() and ch.isalpha() for ch in text):
                self.language = "en"
            else:
                self.language = "unknown"

    @property
    def content_hash(self) -> str:
        text = f"{self.title.strip()}\n{self.content.strip()}"
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "source_type": self.source_type,
            "title": self.title,
            "content": self.content,
            "url": self.url or "",
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "language": self.language,
            "metadata": self.metadata,
            "raw": self.raw,
            "ingested_at": self.ingested_at.isoformat(),
            "content_hash": self.content_hash,
        }

    def to_rag_document(self) -> dict[str, Any]:
        """SQLite RAG documents 表使用的结构。"""
        return {
            "id": self.id,
            "source": self.source,
            "source_type": self.source_type,
            "title": self.title,
            "content": self.content,
            "url": self.url or "",
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "language": self.language,
            "metadata": self.metadata,
        }


def document_from_dict(data: dict[str, Any]) -> TechCompassDocument:
    """从统一 dict 构造 Document。"""
    published = data.get("published_at")
    if isinstance(published, str):
        try:
            published = datetime.fromisoformat(published.replace("Z", "+00:00"))
        except ValueError:
            published = None
    ingested = data.get("ingested_at")
    if isinstance(ingested, str):
        try:
            ingested = datetime.fromisoformat(ingested.replace("Z", "+00:00"))
        except ValueError:
            ingested = utcnow()

    source = str(data.get("source", "unknown"))
    external_id = str(data.get("id", ""))
    doc_id = str(data.get("id", "")) if source == "unknown" else make_document_id(source, external_id)

    return TechCompassDocument(
        id=doc_id,
        source=source,
        source_type=str(data.get("source_type", "unknown")),
        title=str(data.get("title", "")),
        content=str(data.get("content", "")),
        url=data.get("url"),
        published_at=published,
        language=data.get("language"),
        metadata=dict(data.get("metadata") or {}),
        raw=dict(data.get("raw") or {}),
        ingested_at=ingested if isinstance(ingested, datetime) else utcnow(),
    )

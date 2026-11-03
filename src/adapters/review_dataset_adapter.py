"""TechCompass — 评价数据集适配器。

职责：把本地 jsonl / csv 的原始评价字段归一化成统一 `Item`。
Week 2 先用本地样例跑通；后续可扩展到 Amazon 评论等公开数据集。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .schema import Item, SOURCE_REVIEW, parse_timestamp
from .base import BaseAdapter, APIResult


class ReviewDatasetAdapter(BaseAdapter):
    name = "review_dataset"

    def __init__(self, path: str):
        self.path = Path(path)

    def validate_config(self) -> bool:
        return self.path.exists()

    def fetch(self) -> APIResult:
        if not self.path.exists():
            return APIResult(
                ok=False,
                error_code="FILE_NOT_FOUND",
                retryable=False,
                fallback_action="use_empty",
            )

        try:
            items = self._load()
        except Exception as e:  # noqa: BLE001
            return APIResult(
                ok=False,
                error_code="LOAD_ERROR",
                retryable=False,
                fallback_action="use_empty",
                raw_data=str(e),
            )

        return APIResult(ok=True, items=items, raw_data={"file": str(self.path)})

    def _load(self) -> list[Item]:
        if self.path.suffix == ".jsonl":
            return self._load_jsonl()
        if self.path.suffix == ".csv":
            return self._load_csv()
        raise ValueError(f"不支持的文件类型: {self.path.suffix}")

    def _load_jsonl(self) -> list[Item]:
        items = []
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                items.append(self._normalize(raw))
        return items

    def _load_csv(self) -> list[Item]:
        items = []
        with self.path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                items.append(self._normalize(row))
        return items

    def _normalize(self, raw: dict[str, Any]) -> Item:
        """字段名容忍大小写/常见别名，统一成 Item。"""
        def get(*names: str, default=None):
            for n in names:
                if n in raw and raw[n] not in (None, ""):
                    return raw[n]
            return default

        rating = get("rating", "star", "stars")
        try:
            rating = float(rating) if rating is not None else None
        except (TypeError, ValueError):
            rating = None

        return Item(
            id=str(get("id", "review_id", default="")),
            source_type=SOURCE_REVIEW,
            product_category=str(get("product_category", "category", default="")),
            product_name=str(get("product_name", "product", "name", default="")),
            title=str(get("title", default="")),
            content=str(get("content", "text", "review", "body", default="")),
            timestamp=parse_timestamp(get("timestamp", "date", "time", default=None)),
            rating=rating,
            raw=raw,
        )

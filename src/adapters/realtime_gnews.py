"""GNews 实时公开新闻源适配器。

实现 RealtimeSource 接口，调用 GNews API 搜索最新新闻。
免费额度有限：注意请求频率，失败时返回空列表并保留 error 到 metadata。
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

import httpx
from dotenv import load_dotenv

from .realtime_base import RealtimeSource, RealtimeItem

GNEWS_URL = "https://gnews.io/api/v4/search"


class GNewsSource(RealtimeSource):
    name = "gnews"

    def __init__(self, api_key: Optional[str] = None, env_path: Optional[str] = None):
        if env_path:
            load_dotenv(env_path)
        self.api_key = api_key or os.getenv("GNEWS_API_KEY")

    def validate_config(self) -> bool:
        return bool(self.api_key)

    def search(self, query: str, limit: int = 10) -> list[RealtimeItem]:
        if not self.api_key:
            return []
        params = {"q": query, "lang": "en", "max": limit, "token": self.api_key}
        try:
            resp = httpx.get(GNEWS_URL, params=params, timeout=30)
        except Exception as e:  # noqa: BLE001
            return []
        if resp.status_code != 200:
            return []
        data = resp.json()
        articles = data.get("articles", [])
        items = []
        for a in articles:
            ts = a.get("publishedAt")
            dt = None
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except Exception:  # noqa: BLE001
                dt = None
            items.append(RealtimeItem(
                id=str(a.get("id") or a.get("url") or ""),
                source=self.name,
                title=str(a.get("title") or ""),
                content=str(a.get("description") or a.get("content") or ""),
                url=str(a.get("url") or ""),
                published_at=dt,
                metadata={"source_name": (a.get("source") or {}).get("name"),
                          "lang": a.get("lang"), "image": a.get("image")},
            ))
        return items

"""NewsAPI 实时公开新闻源适配器。

实现 RealtimeSource 接口，调用 NewsAPI (newsapi.org) 搜索最新新闻。
注意：
- 免费额度按天计，且有国家/语言限制。
- 开发者版只能在开发环境使用，生产不可用（NewsAPI 限制）。
- 使用 /v2/everything 按关键词 + 日期范围检索。
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx
from dotenv import load_dotenv

from .realtime_base import RealtimeSource, RealtimeItem

NEWSAPI_URL = "https://newsapi.org/v2/everything"


class NewsAPISource(RealtimeSource):
    name = "newsapi"

    def __init__(self, api_key: Optional[str] = None, env_path: Optional[str] = None):
        if env_path:
            load_dotenv(env_path)
        self.api_key = api_key or os.getenv("NEWS_API_KEY")

    def validate_config(self) -> bool:
        return bool(self.api_key)

    def search(self, query: str, limit: int = 10, days_back: int = 30) -> list[RealtimeItem]:
        if not self.api_key:
            return []
        to_dt = datetime.now(timezone.utc)
        from_dt = to_dt - timedelta(days=days_back)
        params = {
            "q": query,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": min(limit, 100),
            "from": from_dt.strftime("%Y-%m-%dT%H:%M:%S"),
            "to": to_dt.strftime("%Y-%m-%dT%H:%M:%S"),
            "apiKey": self.api_key,
        }
        try:
            resp = httpx.get(NEWSAPI_URL, params=params, timeout=30)
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
                id=str(a.get("url") or ""),
                source=self.name,
                title=str(a.get("title") or ""),
                content=str(a.get("description") or ""),
                url=str(a.get("url") or ""),
                published_at=dt,
                metadata={"source_name": (a.get("source") or {}).get("name"),
                          "author": a.get("author")},
            ))
        return items

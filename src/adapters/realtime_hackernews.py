"""TechCompass — Hacker News 实时讨论源适配器。

使用 Algolia HN Search API（免费、无需 key）：
    https://hn.algolia.com/api/v1/search_by_date?query=<q>&hitsPerPage=<n>

特点：
- 无需凭据，公开接口，稳定性好。
- 覆盖数码消费品相关的 stories（含热度 points / comments）。
- 失败或限流时返回空列表，不抛异常。
"""

from __future__ import annotations

from datetime import datetime

import httpx

from .realtime_base import RealtimeSource, RealtimeItem

HN_SEARCH_URL = "https://hn.algolia.com/api/v1/search_by_date"


class HackerNewsSource(RealtimeSource):
    name = "hackernews"

    def __init__(self, timeout: float = 15.0):
        self.timeout = timeout

    def validate_config(self) -> bool:
        return True

    def search(self, query: str, limit: int = 10) -> list[RealtimeItem]:
        params = {"query": query, "hitsPerPage": min(limit, 50), "tags": "story"}
        try:
            resp = httpx.get(HN_SEARCH_URL, params=params, timeout=self.timeout)
        except Exception:  # noqa: BLE001
            return []
        if resp.status_code != 200:
            return []
        try:
            data = resp.json()
        except Exception:  # noqa: BLE001
            return []

        items: list[RealtimeItem] = []
        for h in data.get("hits", []):
            title = str(h.get("title") or "").strip()
            if not title:
                continue
            object_id = str(h.get("objectID") or "")
            url = str(h.get("url") or "") or f"https://news.ycombinator.com/item?id={object_id}"
            points = h.get("points")
            num_comments = h.get("num_comments")
            story_text = (h.get("story_text") or "").strip()
            # content：优先正文，否则用 title + 热度作为轻量语义信号
            content = story_text or f"{title}. points={points}, comments={num_comments}"

            created = h.get("created_at")
            dt = None
            if created:
                try:
                    dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                except ValueError:
                    dt = None

            items.append(RealtimeItem(
                id=object_id,
                source=self.name,
                title=title,
                content=content[:500],
                url=url,
                published_at=dt,
                metadata={
                    "author": h.get("author"),
                    "points": points,
                    "num_comments": num_comments,
                    "hn_url": f"https://news.ycombinator.com/item?id={object_id}",
                },
            ))
            if len(items) >= limit:
                break
        return items

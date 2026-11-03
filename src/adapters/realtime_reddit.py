"""TechCompass — Reddit 实时公开讨论源适配器。

由于当前无法创建 Reddit App，使用 Reddit 公开的 Atom/RSS 搜索端点：
    https://www.reddit.com/search.rss?q=<query>&limit=<limit>

特点：
- 不需要 client_id / client_secret。
- 可获取公开搜索结果。
- 有限流风险：HTTP 429 时返回空列表，不抛异常。
- 只请求公开 RSS，不做网页爬取。
"""

from __future__ import annotations

import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Optional

from .realtime_base import RealtimeSource, RealtimeItem
from .text_cleaner import strip_html

REDDIT_SEARCH_RSS = "https://www.reddit.com/search.rss"
ATOM_NS = {"a": "http://www.w3.org/2005/Atom"}


class RedditSource(RealtimeSource):
    name = "reddit_rss"

    def __init__(self, user_agent: str = "TechCompass/0.1 research prototype"):
        self.user_agent = user_agent

    def validate_config(self) -> bool:
        return True

    def search(self, query: str, limit: int = 10) -> list[RealtimeItem]:
        params = urllib.parse.urlencode({"q": query, "limit": min(limit, 25)})
        url = f"{REDDIT_SEARCH_RSS}?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})

        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                if resp.status != 200:
                    return []
                data = resp.read()
        except Exception:  # noqa: BLE001
            # 常见错误：429 Too Many Requests、网络不可达、超时
            return []

        try:
            root = ET.fromstring(data)
        except ET.ParseError:
            return []

        items: list[RealtimeItem] = []
        for entry in root.findall("a:entry", ATOM_NS):
            title = entry.findtext("a:title", default="", namespaces=ATOM_NS) or ""
            link_el = entry.find("a:link", ATOM_NS)
            url_out = link_el.get("href", "") if link_el is not None else ""
            updated = entry.findtext("a:updated", default="", namespaces=ATOM_NS) or ""
            content_el = entry.find("a:content", ATOM_NS)
            raw_content = (content_el.text or "") if content_el is not None and content_el.text else ""
            content = strip_html(raw_content)

            dt = None
            try:
                dt = datetime.fromisoformat(updated.replace("Z", "+00:00"))
            except ValueError:
                dt = None

            # RSS entry 的 id 通常包含 subreddit 和帖子 id
            raw_id = entry.findtext("a:id", default="", namespaces=ATOM_NS) or ""
            items.append(RealtimeItem(
                id=raw_id,
                source=self.name,
                title=title,
                content=content[:500],
                url=url_out,
                published_at=dt,
                metadata={
                    "raw_id": raw_id,
                    "endpoint": "search.rss",
                },
            ))
            if len(items) >= limit:
                break
        return items

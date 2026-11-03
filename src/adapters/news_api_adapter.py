"""TechCompass — GNews 新闻 API 适配器。

职责：调用 GNews API 搜索新闻，并把返回的新闻条目归一化成统一 `Item`
（source_type=news）。仅供研究/原型验证使用，遵守 API 免费额度限制。

流程：
1. 读 .env 里的 GNEWS_API_KEY。
2. 用 httpx 请求 https://gnews.io/api/v4/search。
3. 将每篇文章映射成 Item（title/content/timestamp）。
4. 网络失败 / key 缺失 / 额度超限时返回 APIResult(ok=False)，由 runner 降级。
"""

from __future__ import annotations

import os
from typing import Any, Optional

import httpx
from dotenv import load_dotenv

from .schema import Item, SOURCE_NEWS, parse_timestamp
from .base import BaseAdapter, APIResult

GNEWS_URL = "https://gnews.io/api/v4/search"


class NewsAPIAdapter(BaseAdapter):
    name = "gnews"

    def __init__(
        self,
        query: str,
        lang: str = "en",
        max_results: int = 5,
        api_key: Optional[str] = None,
        env_path: Optional[str] = None,
    ):
        self.query = query
        self.lang = lang
        self.max_results = max_results
        if env_path:
            load_dotenv(env_path)
        self.api_key = api_key or os.getenv("GNEWS_API_KEY")

    def validate_config(self) -> bool:
        return bool(self.api_key)

    def fetch(self) -> APIResult:
        if not self.api_key:
            return APIResult(
                ok=False,
                error_code="MISSING_API_KEY",
                retryable=False,
                fallback_action="use_local_samples",
            )

        params = {
            "q": self.query,
            "lang": self.lang,
            "max": self.max_results,
            "token": self.api_key,
        }

        try:
            resp = httpx.get(GNEWS_URL, params=params, timeout=20)
        except Exception as e:  # noqa: BLE001
            return APIResult(
                ok=False,
                error_code="NETWORK_ERROR",
                retryable=True,
                fallback_action="use_local_samples",
                raw_data=str(e),
            )

        if resp.status_code != 200:
            code = "HTTP_" + str(resp.status_code)
            # GNews 免费版达到每日上限时通常返回 4xx/429
            retryable = resp.status_code in (429, 500, 502, 503, 504)
            return APIResult(
                ok=False,
                error_code=code,
                retryable=retryable,
                fallback_action="use_local_samples",
                raw_data=str(resp.text)[:300],
            )

        try:
            data = resp.json()
        except Exception as e:  # noqa: BLE001
            return APIResult(
                ok=False,
                error_code="BAD_JSON",
                retryable=False,
                fallback_action="use_local_samples",
                raw_data=str(e),
            )

        articles = data.get("articles", [])
        items = [self._normalize(a) for a in articles]
        return APIResult(
            ok=True,
            items=items,
            raw_data={"totalArticles": data.get("totalArticles")},
        )

    def _normalize(self, a: dict[str, Any]) -> Item:
        source = a.get("source") or {}
        source_name = source.get("name") if isinstance(source, dict) else str(source)

        return Item(
            id=str(a.get("id") or a.get("url") or ""),
            source_type=SOURCE_NEWS,
            product_category=self._guess_category(a.get("title", "") + a.get("description", "")),
            product_name=str(source_name or ""),
            title=str(a.get("title") or ""),
            content=str(a.get("description") or a.get("content") or ""),
            timestamp=parse_timestamp(a.get("publishedAt")),
            rating=None,
            metadata={
                "url": a.get("url"),
                "image": a.get("image"),
                "lang": a.get("lang"),
                "source": source_name,
            },
            raw=a,
        )

    @staticmethod
    def _guess_category(text: str) -> str:
        """粗糙品类猜测，Week3 会换成 LLM 主题分类。"""
        t = text.lower()
        if any(k in t for k in ["earbud", "headphone", "earphone"]):
            return "headphones"
        if any(k in t for k in ["charger", "charging", "power"]):
            return "charger"
        if any(k in t for k in ["watch", "wearable", "smartwatch"]):
            return "watch"
        return "electronics"

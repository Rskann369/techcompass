"""TechCompass — 领域工具（检索/处理/输出三层）。

检索层 search_reviews：采用「分片关键词命中 + 计分排序」，比整段子串匹配
在中文场景下召回更好（例如「发烫 安全隐患」也能命中「充电发热」）。

处理层 classify_review：封装结构化分类。
"""

from __future__ import annotations

import re
from typing import Any

from adapters.review_dataset_adapter import ReviewDatasetAdapter
from adapters.news_api_adapter import NewsAPIAdapter
from project_paths import DEFAULT_ENV_PATH
from tools.registry import Tool, ToolRegistry


# 同义/相关词映射：用于召回时做轻量扩展
SYNONYMS = {
    "发热": ["发热", "发烫", "烫", "过热"],
    "发烫": ["发热", "发烫", "烫", "过热"],
    "续航": ["续航", "电池", "半天"],
    "电池": ["续航", "电池", "半天"],
    "充不进": ["充不进", "接触不良", "断充"],
    "断充": ["断充", "接触不良", "充不进"],
    "无声": ["无声", "单侧", "没声音"],
    "松动": ["松动", "接触不良", "断开"],
}


def _tokenize(text: str) -> list[str]:
    """把 query 拆成多个可检索 token（空格 + 常见标点分隔）。"""
    text = re.sub(r"[，。！？、,!.?;；:：]+", " ", text)
    parts = [p for p in text.split() if p]
    # 对每个 part 再按 2-4 字做滑动切分，提升中文命中能力
    tokens = set()
    for p in parts:
        p = p.strip()
        tokens.add(p)
        if len(p) >= 2:
            # 扩同义词
            for k, vals in SYNONYMS.items():
                if k in p:
                    tokens.update(vals)
        # 滑动 2 字 & 3 字
        for w in (2, 3):
            for i in range(0, len(p) - w + 1):
                sub = p[i:i + w]
                if len(sub) >= 2:
                    tokens.add(sub)
    return {t for t in tokens if t}


def make_search_reviews(dataset_path: str) -> Tool:
    """检索本地评价：关键词分片命中 + 计分排序。"""
    def run(args: dict[str, Any]) -> dict[str, Any]:
        kw = str(args.get("query", ""))
        max_r = int(args.get("max", 3))
        res = ReviewDatasetAdapter(dataset_path).fetch()
        items = res.items if res.ok else []

        tokens = _tokenize(kw)
        scored = []
        for it in items:
            text = it.title + it.content
            score = 0
            hit_tokens = []
            for t in tokens:
                if t in text:
                    score += 1
                    hit_tokens.append(t)
            if score > 0:
                scored.append((score, len(hit_tokens), it))

        # 按分数降序（分数相同按命中 token 种类数降序）取 top max_r
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        hits = [x[2] for x in scored[:max_r]]

        return {
            "count": len(hits),
            "reviews": [
                {"id": it.id, "product": it.product_name,
                 "title": it.title, "content": it.content, "rating": it.rating}
                for it in hits
            ],
        }
    return Tool(
        name="search_reviews",
        description="在本地已知评价库中按关键词检索相关用户评论，用于取证用户反馈。支持多关键词。",
        args_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索关键词（可多个，空格分隔）"},
                "max": {"type": "integer", "description": "最多返回条数"},
            },
            "required": ["query"],
        },
        run=run,
        fallback=lambda: {"count": 0, "reviews": []},
    )


def make_search_news(env_path: str = str(DEFAULT_ENV_PATH)) -> Tool:
    """检索新闻：调用公开实时新闻源。

    当前主源：GNews（稳定可用、公开 API）。
    NewsAPI 因当前环境连接超时，暂不接入主链路，保留为后续候选。
    """
    from adapters.realtime_gnews import GNewsSource

    source = GNewsSource(env_path=env_path)

    def run(args: dict[str, Any]) -> dict[str, Any]:
        q = str(args.get("query", ""))
        max_r = int(args.get("max", 5))
        items = source.search(q, limit=max_r)
        return {
            "count": len(items),
            "news": [
                {"title": it.title, "source": it.metadata.get("source_name"),
                 "published": it.published_at.isoformat() if it.published_at else None,
                 "url": it.url}
                for it in items
            ],
        }
    return Tool(
        name="search_news",
        description="通过公开实时新闻源（当前为 GNews）检索最新新闻，用于补充舆情背景。可能因额度/网络失败。",
        args_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "新闻搜索关键词"},
                "max": {"type": "integer", "description": "最多返回条数"},
            },
            "required": ["query"],
        },
        run=run,
        fallback=lambda: {"count": 0, "news": [], "error": "fallback"},
    )


def make_search_discussions(env_path: str = str(DEFAULT_ENV_PATH)) -> Tool:
    """检索用户讨论：Reddit 公开 RSS 搜索，无需 App 凭据。"""
    from adapters.realtime_reddit import RedditSource

    source = RedditSource()

    def run(args: dict[str, Any]) -> dict[str, Any]:
        q = str(args.get("query", ""))
        max_r = int(args.get("max", 5))
        items = source.search(q, limit=max_r)
        return {
            "count": len(items),
            "discussions": [
                {"title": it.title, "content": it.content,
                 "url": it.url,
                 "subreddit": it.metadata.get("subreddit"),
                 "score": it.metadata.get("score"),
                 "num_comments": it.metadata.get("num_comments")}
                for it in items
            ],
        }
    return Tool(
        name="search_discussions",
        description="通过 Reddit 公开 RSS 检索真实用户讨论。无需凭据；被限流或网络失败时返回空。",
        args_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "讨论搜索关键词"},
                "max": {"type": "integer", "description": "最多返回条数"},
            },
            "required": ["query"],
        },
        run=run,
        fallback=lambda: {"count": 0, "discussions": []},
    )


def make_classify_review() -> Tool:
    """处理层：分类结果强制结构化（LLM 用 Tool Calling，这里供本地规则或校验）。"""
    def run(args: dict[str, Any]) -> dict[str, Any]:
        cat = str(args.get("category", "其他"))
        sent = str(args.get("sentiment", "neutral"))
        return {"category": cat, "sentiment": sent}
    return Tool(
        name="classify_review",
        description="对一条用户评论做主题分类和情绪判断。category 取值：产品问题/物流问题/宣传问题/使用问题/其他；sentiment 取值：positive/neutral/negative/risk。",
        args_schema={
            "type": "object",
            "properties": {
                "category": {"type": "string"},
                "sentiment": {"type": "string"},
            },
            "required": ["category", "sentiment"],
        },
        run=run,
    )


def build_registry(dataset_path: str, env_path: str = str(DEFAULT_ENV_PATH), callbacks=None) -> ToolRegistry:
    from rag.tools import register_rag_tools
    from tools.analysis_tools import register_analysis_tools
    from tools.llm_analysis_tools import make_analyze_consumer_sentiment
    from tools.live_search_tools import register_live_search_tools
    from tools.output_tools import register_output_tools

    reg = ToolRegistry()
    # 统一检索入口
    register_rag_tools(reg, db_path="data/rag/techcompass.db")
    # 实时多源检索（混合检索：索引不足时现场补充）
    register_live_search_tools(reg, env_path=env_path)
    # 开放域分析：rewrite + model analysis
    register_analysis_tools(reg)
    reg.register(make_analyze_consumer_sentiment(callbacks=callbacks))
    # 输出与归档
    register_output_tools(reg)
    return reg

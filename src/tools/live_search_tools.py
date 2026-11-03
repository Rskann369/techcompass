"""TechCompass — 实时多源搜索工具（混合检索的现场拉取部分）。

search_live_sources：按 query 现场从多源（HackerNews / GNews / Reddit）拉取，
语义重排序后返回 top N（含链接与相似度）。

用途：
- 当本地索引证据不足或与问题无关时，由 Agent 现场补充最新数据。
- 只返回内存结果，不写回 SQLite。

merge_evidence：把现场拉取结果合并进索引证据（去重 + 按相似度重排 + 截断 top_k）。
"""

from __future__ import annotations

from typing import Any

from project_paths import DEFAULT_ENV_PATH
from tools.registry import Tool


def merge_evidence(
    index_docs: list[dict[str, Any]],
    live_results: list[dict[str, Any]],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """把现场结果合并进索引证据：按 url 去重、按相似度降序、截断 top_k。"""
    merged = list(index_docs)
    seen_urls = {d.get("url") for d in merged if d.get("url")}
    for d in live_results:
        url = d.get("url")
        if url and url in seen_urls:
            continue
        merged.append(d)
        if url:
            seen_urls.add(url)
    merged.sort(key=lambda d: d.get("_semantic_similarity", 0) or 0.0, reverse=True)
    return merged[:top_k]


def index_evidence_is_weak(
    index_docs: list[dict[str, Any]],
    min_count: int = 2,
    quality_threshold: float = 0.62,
) -> bool:
    """判断索引证据是否「不足或偏弱」，需要现场拉取补充。

    两种情况算弱：
    - 相关证据数量少于 min_count；
    - 最相关的证据相似度低于 quality_threshold（说明索引里的匹配本身不可信）。
    """
    if len(index_docs) < min_count:
        return True
    best_sim = max((d.get("_semantic_similarity", 0) or 0.0) for d in index_docs)
    return best_sim < quality_threshold


def make_search_live_sources(
    env_path: str | None = str(DEFAULT_ENV_PATH),
    default_top_k: int = 5,
    default_threshold: float = 0.55,
) -> Tool:
    def run(args: dict[str, Any]) -> dict[str, Any]:
        from adapters.multi_source import build_default_fetcher
        from rag.semantic_rerank import SemanticReranker

        query = str(args.get("query", ""))
        top_k = int(args.get("top_k", default_top_k))
        threshold = float(args.get("threshold", default_threshold))

        fetcher = build_default_fetcher(env_path=env_path)
        result = fetcher.search(query, limit_per_source=10)
        reranker = SemanticReranker()
        top = reranker.rerank(
            query,
            [d.to_dict() for d in result.documents],
            top_k=top_k,
            threshold=threshold,
        )
        return {
            "query": query,
            "count": len(top),
            "total_raw": result.total_raw,
            "per_source": result.per_source,
            "results": [
                {
                    "title": d.get("title"),
                    "content": d.get("content"),
                    "url": d.get("url"),
                    "source": d.get("source"),
                    "source_type": d.get("source_type"),
                    "published_at": d.get("published_at"),
                    "_semantic_similarity": d.get("_semantic_similarity"),
                }
                for d in top
            ],
        }

    return Tool(
        name="search_live_sources",
        description=(
            "实时多源检索：按 query 现场从 HackerNews/GNews/Reddit 拉取最新样本，"
            "语义重排序后返回 top N（含链接与相似度）。"
            "当本地索引证据不足或需要最新舆情时使用，不写回数据库。"
        ),
        args_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer"},
                "threshold": {"type": "number"},
            },
            "required": ["query"],
        },
        run=run,
        fallback=lambda: {"query": "", "count": 0, "total_raw": 0, "per_source": {}, "results": []},
    )


def register_live_search_tools(registry, env_path: str | None = str(DEFAULT_ENV_PATH)) -> None:
    registry.register(make_search_live_sources(env_path=env_path))

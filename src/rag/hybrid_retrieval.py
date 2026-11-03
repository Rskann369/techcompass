"""TechCompass — 多查询融合检索（Hybrid Retrieval）。

流程：
1. 接收多个 rewrite queries
2. 每个查询分别调用 LlamaIndex retriever
3. 按 document/chunk 去重
4. 使用 Reciprocal Rank Fusion (RRF) 融合排序
5. 返回 top_k 证据

RRF 公式：
    score(d) = Σ 1 / (k + rank_i(d))
其中 k 通常取 60。
"""

from __future__ import annotations

from typing import Any

from rag.llamaindex_store import build_llamaindex_index, llamaindex_search

RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict[str, Any]]],
    top_k: int = 5,
    k: int = RRF_K,
) -> list[dict[str, Any]]:
    """对多个排序结果做 RRF 融合。"""
    scores: dict[str, float] = {}
    items: dict[str, dict[str, Any]] = {}

    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            key = _result_key(item)
            if key not in items:
                items[key] = item
                scores[key] = 0.0
            scores[key] += 1.0 / (k + rank)

    ranked_keys = sorted(scores, key=lambda x: scores[x], reverse=True)[:top_k]
    results = []
    for key in ranked_keys:
        item = dict(items[key])
        item["rrf_score"] = round(scores[key], 6)
        results.append(item)
    return results


def _result_key(item: dict[str, Any]) -> str:
    """生成 document-level 去重 key，避免同一文档多个 chunk 重复返回。"""
    url = item.get("url")
    title = item.get("title")
    if url:
        return f"url:{url}"
    if title:
        return f"title:{title}"
    return f"text:{item.get('text', '')[:200]}"


def hybrid_search(
    query: str,
    queries: list[str] | None = None,
    top_k: int = 5,
    per_query_k: int = 5,
    db_path: str = "data/rag/techcompass.db",
) -> dict[str, Any]:
    """多查询融合检索。"""
    effective_queries = list(dict.fromkeys([query, *(queries or [])]))
    index = build_llamaindex_index(db_path)

    ranked_lists = []
    for q in effective_queries:
        results = llamaindex_search(index, q, top_k=per_query_k)
        ranked_lists.append(results)

    fused = reciprocal_rank_fusion(ranked_lists, top_k=top_k)
    return {
        "query": query,
        "queries": effective_queries,
        "engine": "llamaindex_rrf",
        "count": len(fused),
        "results": fused,
    }

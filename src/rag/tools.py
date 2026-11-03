"""TechCompass — RAG 工具层。

把 SQLite RAG 暴露为 Agent 可调用工具：
- rag_search：本地 SQLite 向量检索
- rag_stats：查看语料库规模
- rag_rebuild：重新拉取公开实时数据并构建索引
"""

from __future__ import annotations

from typing import Any

from tools.registry import Tool, ToolRegistry
from rag.store import RagStore, search_store, chunk_text


def make_rag_search(db_path: str = "data/rag/techcompass.db") -> Tool:
    def run(args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query", ""))
        top_k = int(args.get("top_k", 5))
        store = RagStore(db_path)
        try:
            results = search_store(store, query, top_k)
            return {
                "query": query,
                "count": len(results),
                "results": [r.to_dict() for r in results],
            }
        finally:
            store.close()
    return Tool(
        name="rag_search",
        description="在本地 SQLite RAG 语料库中检索真实新闻/讨论证据。返回文本、来源、URL、时间与相似度。",
        args_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["query"],
        },
        run=run,
        fallback=lambda: {"query": "", "count": 0, "results": []},
    )


def make_rag_stats(db_path: str = "data/rag/techcompass.db") -> Tool:
    def run(args: dict[str, Any]) -> dict[str, Any]:
        store = RagStore(db_path)
        try:
            return {"stats": store.stats()}
        finally:
            store.close()
    return Tool(
        name="rag_stats",
        description="查看本地 RAG 语料库规模，包括 documents/chunks/embeddings/retrievals/runs。",
        args_schema={"type": "object", "properties": {}, "required": []},
        run=run,
        fallback=lambda: {"stats": {}},
    )


def register_rag_tools(registry: ToolRegistry, db_path: str = "data/rag/techcompass.db") -> None:
    """只暴露统一检索入口，避免 Agent 重复选择多个 RAG 工具。"""
    registry.register(make_hybrid_rag_search(db_path))


def make_llamaindex_search(db_path: str = "data/rag/techcompass.db") -> Tool:
    """LlamaIndex VectorStoreIndex 检索工具。"""
    from rag.llamaindex_store import build_llamaindex_index, llamaindex_search

    # 每次调用构建索引，MVP 规模下可接受；后续可加进程内缓存。
    def run(args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query", ""))
        top_k = int(args.get("top_k", 5))
        index = build_llamaindex_index(db_path)
        results = llamaindex_search(index, query, top_k=top_k)
        return {"query": query, "engine": "llamaindex", "count": len(results), "results": results}
    return Tool(
        name="llamaindex_search",
        description="使用 LlamaIndex VectorStoreIndex 检索本地真实公开数据，适合语义/关键词混合证据检索。",
        args_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["query"],
        },
        run=run,
        fallback=lambda: {"query": "", "engine": "llamaindex", "count": 0, "results": []},
    )


def make_hybrid_rag_search(db_path: str = "data/rag/techcompass.db") -> Tool:
    """多查询融合检索工具：LLamaIndex + RRF。"""
    from rag.hybrid_retrieval import hybrid_search

    def run(args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query", ""))
        queries = args.get("queries", [])
        top_k = int(args.get("top_k", 5))
        return hybrid_search(
            query=query,
            queries=queries if isinstance(queries, list) else [],
            top_k=top_k,
            db_path=db_path,
        )
    return Tool(
        name="hybrid_rag_search",
        description="多查询融合检索：对多个改写查询分别检索并用 RRF 融合排序，提高召回稳定性。",
        args_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "queries": {"type": "array", "items": {"type": "string"}},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["query"],
        },
        run=run,
        fallback=lambda: {"query": "", "queries": [], "count": 0, "results": []},
    )

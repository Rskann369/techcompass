"""TechCompass — HTML 清洗与多查询融合检索测试。"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adapters.text_cleaner import is_html_like, strip_html  # noqa: E402
from rag.hybrid_retrieval import hybrid_search, reciprocal_rank_fusion  # noqa: E402
from rag.store import RagStore  # noqa: E402
from tools.domain_tools import build_registry  # noqa: E402


def test_strip_html():
    raw = '<div>Hello <a href="x">world</a></div><p>Battery &amp; safety</p>'
    clean = strip_html(raw)
    assert "<div>" not in clean
    assert "<a" not in clean
    assert "Hello" in clean
    assert "world" in clean
    assert "Battery & safety" in clean
    assert is_html_like(raw)
    assert not is_html_like(clean)


def test_rrf_document_level_dedup():
    item = {"url": "https://example.com/a", "title": "A", "text": "x"}
    ranked_lists = [[item, item]]
    fused = reciprocal_rank_fusion(ranked_lists, top_k=5)
    assert len(fused) == 1
    assert fused[0]["rrf_score"] > 0


@pytest.mark.local
def test_hybrid_search_returns_multiple_unique_documents():
    result = hybrid_search(
        query="wireless earbuds safety",
        queries=["safety hazard", "headphones"],
        top_k=5,
    )
    assert result["count"] > 0
    keys = [r.get("url") or r.get("title") for r in result["results"]]
    assert len(keys) == len(set(keys))


@pytest.mark.local
def test_hybrid_rag_tool_registered_and_runs():
    registry = build_registry("data/samples/nonexistent.jsonl")
    assert "hybrid_rag_search" in registry.names()
    tool = registry.get("hybrid_rag_search")
    result = tool.run({
        "query": "smartwatch battery",
        "queries": ["battery life", "solar smartwatch"],
        "top_k": 5,
    })
    assert result["engine"] == "llamaindex_rrf"
    assert result["count"] > 0


@pytest.mark.local
def test_no_orphan_embeddings():
    store = RagStore()
    try:
        orphan = store.conn.execute(
            "SELECT COUNT(*) FROM embeddings WHERE chunk_id NOT IN (SELECT id FROM chunks)"
        ).fetchone()[0]
        assert orphan == 0
    finally:
        store.close()


def test_tool_count_is_8():
    registry = build_registry("data/samples/nonexistent.jsonl")
    assert len(registry.names()) == 8


if __name__ == "__main__":
    tests = [
        test_strip_html,
        test_rrf_document_level_dedup,
        test_hybrid_search_returns_multiple_unique_documents,
        test_hybrid_rag_tool_registered_and_runs,
        test_no_orphan_embeddings,
        test_tool_count_is_8,
    ]
    passed = 0
    for test in tests:
        try:
            test()
            print("PASS", test.__name__)
            passed += 1
        except Exception as e:
            print("FAIL", test.__name__, "->", e)
    print(f"{passed}/{len(tests)} passed")

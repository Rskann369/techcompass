"""混合检索：search_live_sources 工具 + 证据合并逻辑测试。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tools.live_search_tools import (  # noqa: E402
    merge_evidence,
    make_search_live_sources,
    index_evidence_is_weak,
)
from tools.domain_tools import build_registry  # noqa: E402


def test_index_evidence_is_weak():
    assert index_evidence_is_weak([], min_count=2, quality_threshold=0.62) is True
    assert index_evidence_is_weak([{"_semantic_similarity": 0.56}], 2, 0.62) is True
    assert index_evidence_is_weak(
        [{"_semantic_similarity": 0.7}, {"_semantic_similarity": 0.8}], 2, 0.62
    ) is False


def test_search_live_sources_registered():
    reg = build_registry(dataset_path="nonexistent", env_path=None)
    assert "search_live_sources" in reg.names()


def test_merge_evidence_dedups_and_ranks():
    index = [
        {"title": "a", "url": "https://x.com/1", "_semantic_similarity": 0.6},
        {"title": "b", "url": "https://x.com/2", "_semantic_similarity": 0.55},
    ]
    live = [
        {"title": "dup", "url": "https://x.com/1", "_semantic_similarity": 0.9},
        {"title": "c", "url": "https://x.com/3", "_semantic_similarity": 0.8},
        {"title": "d", "url": "https://x.com/4", "_semantic_similarity": 0.7},
    ]
    merged = merge_evidence(index, live, top_k=5)
    urls = [d["url"] for d in merged]
    assert urls == ["https://x.com/3", "https://x.com/4", "https://x.com/1", "https://x.com/2"]
    assert len(merged) == 4  # dup url 去重


def test_merge_evidence_respects_top_k():
    docs = [
        {"title": str(i), "url": f"https://x.com/{i}", "_semantic_similarity": 1.0 - i * 0.1}
        for i in range(10)
    ]
    assert len(merge_evidence([], docs, top_k=3)) == 3


def test_search_live_sources_tool_runs_with_mock():
    import unittest.mock as mock
    from adapters.multi_source import FetchResult

    class _FakeDoc:
        def __init__(self, title, url, source):
            self.title = title
            self.content = title
            self.url = url
            self.source = source
            self.source_type = "discussion"
            self.published_at = None

        def to_dict(self):
            return {
                "title": self.title, "content": self.content, "url": self.url,
                "source": self.source, "source_type": self.source_type,
                "published_at": None,
            }

    fetcher = mock.MagicMock()
    fetcher.search.return_value = FetchResult(
        query="q",
        documents=[_FakeDoc("earbuds overheat", "https://x.com/1", "hackernews")],
        per_source={"hackernews": 1},
    )
    ranked = [{
        "title": "earbuds overheat", "content": "earbuds overheat",
        "url": "https://x.com/1", "source": "hackernews",
        "source_type": "discussion", "published_at": None, "_semantic_similarity": 0.9,
    }]
    reranker = mock.MagicMock()
    reranker.rerank.return_value = ranked

    with mock.patch("adapters.multi_source.build_default_fetcher", return_value=fetcher), \
         mock.patch("rag.semantic_rerank.SemanticReranker", return_value=reranker):
        tool = make_search_live_sources(env_path=None)
        res = tool.run({"query": "earbuds"})

    assert res["count"] == 1
    assert res["results"][0]["url"] == "https://x.com/1"


if __name__ == "__main__":
    tests = [
        test_index_evidence_is_weak,
        test_search_live_sources_registered,
        test_merge_evidence_dedups_and_ranks,
        test_merge_evidence_respects_top_k,
        test_search_live_sources_tool_runs_with_mock,
    ]
    passed = 0
    for test in tests:
        try:
            test()
            print("PASS", test.__name__)
            passed += 1
        except Exception as e:  # noqa: BLE001
            print("FAIL", test.__name__, "->", e)
    print(f"{passed}/{len(tests)} passed")
    raise SystemExit(0 if passed == len(tests) else 1)

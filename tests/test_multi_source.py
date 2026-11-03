"""多源聚合 + 语义重排序测试（离线，mock 数据源）。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from adapters.multi_source import MultiSourceFetcher  # noqa: E402
from adapters.realtime_base import RealtimeSource, RealtimeItem  # noqa: E402
from rag.semantic_rerank import SemanticReranker  # noqa: E402


class _FakeSource(RealtimeSource):
    def __init__(self, name, items):
        self.name = name
        self._items = items

    def search(self, query, limit=10):
        return self._items


def _item(source, iid, title, url, content=""):
    return RealtimeItem(
        id=iid, source=source, title=title, content=content or title,
        url=url, metadata={},
    )


class _FakeEmbedder:
    def embed(self, text):
        vec = np.zeros(4, dtype=np.float32)
        for i, kw in enumerate(["charging", "battery", "wireless", "earbuds"]):
            if kw in text.lower():
                vec[i] = 1.0
        norm = float(np.linalg.norm(vec))
        return (vec / norm) if norm > 0 else vec

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]


def test_multi_source_aggregates_and_dedups_by_url():
    a = _FakeSource("a", [
        _item("a", "1", "charging issue", "https://x.com/1", "earbuds charging"),
        _item("a", "2", "dupe url", "https://x.com/dup"),
    ])
    b = _FakeSource("b", [
        _item("b", "3", "dupe url again", "https://x.com/dup"),
        _item("b", "4", "battery drain", "https://x.com/4", "battery"),
    ])
    fetcher = MultiSourceFetcher([a, b])
    result = fetcher.search("test", limit_per_source=10)
    assert result.total_raw == 4
    assert result.total_deduped == 3  # dup url 去重一条
    urls = {d.url for d in result.documents}
    assert len(urls) == 3


def test_reranker_orders_and_attaches_similarity():
    docs = [
        {"title": "wireless earbuds charging problem", "content": "overheat while charging", "url": "https://x.com/1"},
        {"title": "random article about food", "content": "nothing to do with earbuds", "url": "https://x.com/2"},
        {"title": "earbuds battery life review", "content": "battery drain fast", "url": "https://x.com/3"},
    ]
    reranker = SemanticReranker(embedder=_FakeEmbedder())
    top = reranker.rerank("wireless earbuds charging", docs, top_k=2, threshold=0.0)
    assert len(top) == 2
    assert top[0]["url"] == "https://x.com/1"  # 最相关
    assert all("_semantic_similarity" in d for d in top)
    assert top[0]["_semantic_similarity"] >= top[1]["_semantic_similarity"]


def test_reranker_threshold_filters_weak_docs():
    docs = [
        {"title": "wireless earbuds charging problem", "content": "overheat", "url": "https://x.com/1"},
        {"title": "unrelated", "content": "gardening tips", "url": "https://x.com/2"},
    ]
    reranker = SemanticReranker(embedder=_FakeEmbedder())
    top = reranker.rerank("wireless earbuds charging", docs, top_k=5, threshold=0.3)
    assert len(top) == 1
    assert top[0]["url"] == "https://x.com/1"


def test_hackernews_parse_without_network():
    import unittest.mock as mock
    import adapters.realtime_hackernews as hn

    class _FakeResp:
        status_code = 200
        def json(self):
            return {"hits": [
                {"objectID": "1", "title": "Earbuds overheat while charging",
                 "url": "https://x.com/1", "points": 10, "num_comments": 3,
                 "created_at": "2026-10-01T00:00:00Z", "story_text": ""},
            ]}

    with mock.patch.object(hn.httpx, "get", return_value=_FakeResp()):
        items = hn.HackerNewsSource().search("earbuds", limit=5)
    assert len(items) == 1
    assert items[0].title == "Earbuds overheat while charging"
    assert items[0].source == "hackernews"
    assert items[0].url == "https://x.com/1"


if __name__ == "__main__":
    tests = [
        test_multi_source_aggregates_and_dedups_by_url,
        test_reranker_orders_and_attaches_similarity,
        test_reranker_threshold_filters_weak_docs,
        test_hackernews_parse_without_network,
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

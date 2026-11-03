"""TechCompass — 统一跨源 Schema 测试。"""

import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adapters.realtime_base import RealtimeItem  # noqa: E402
from adapters.schema import Item, parse_timestamp  # noqa: E402
from adapters.unified_schema import TechCompassDocument, document_from_dict, make_document_id  # noqa: E402
from rag.store import RagStore  # noqa: E402


def test_gnews_realtime_to_document():
    item = RealtimeItem(
        id="abc",
        source="gnews",
        title="Wireless earbuds review",
        content="Battery life is good",
        url="https://example.com",
        published_at=datetime(2026, 10, 10, tzinfo=timezone.utc),
        metadata={"source_name": "Example"},
    )
    doc = item.to_document()
    assert isinstance(doc, TechCompassDocument)
    assert doc.source_type == "news"
    assert doc.language == "en"
    assert doc.content_hash
    assert doc.to_rag_document()["source_type"] == "news"


def test_reddit_realtime_to_document():
    item = RealtimeItem(
        id="t3_xyz",
        source="reddit_rss",
        title="耳机发热",
        content="充电时发热",
        url="https://reddit.com/example",
        metadata={"subreddit": "WirelessEarbuds"},
    )
    doc = item.to_document()
    assert doc.source_type == "discussion"
    assert doc.language == "zh"


def test_local_item_to_document():
    item = Item(
        id="r001",
        source_type="review",
        product_category="headphones",
        product_name="TestBuds",
        title="发热",
        content="充电发烫",
        timestamp=parse_timestamp("2026-10-01"),
        rating=2,
        metadata={"source": "local_review"},
    )
    doc = item.to_document()
    assert doc.source_type == "review"
    assert doc.metadata["product_name"] == "TestBuds"
    assert doc.published_at is not None


def test_document_id_is_stable():
    a = make_document_id("gnews", "abc")
    b = make_document_id("gnews", "abc")
    c = make_document_id("gnews", "xyz")
    assert a == b
    assert a != c


def test_document_roundtrip():
    doc = TechCompassDocument(
        id="x", source="gnews", source_type="news",
        title="Hello", content="World",
    )
    restored = document_from_dict(doc.to_dict())
    assert restored.source == "gnews"
    assert restored.title == "Hello"
    assert restored.content_hash == doc.content_hash


def test_rag_upsert_uses_unified_schema():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "rag.db"
        store = RagStore(str(db))
        item = RealtimeItem(
            id="abc", source="gnews", title="Title", content="Content",
            url="https://example.com", metadata={"source_name": "Example"},
        )
        doc = item.to_document()
        store.upsert_document(doc.to_rag_document())
        row = store.conn.execute("SELECT source, source_type, language, ingested_at FROM documents").fetchone()
        assert row["source"] == "gnews"
        assert row["source_type"] == "news"
        assert row["language"] == "en"
        assert row["ingested_at"]
        store.close()


if __name__ == "__main__":
    tests = [
        test_gnews_realtime_to_document,
        test_reddit_realtime_to_document,
        test_local_item_to_document,
        test_document_id_is_stable,
        test_document_roundtrip,
        test_rag_upsert_uses_unified_schema,
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

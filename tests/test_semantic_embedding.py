"""TechCompass — 本地语义 Embedding 测试。"""

import sys
from pathlib import Path

import pytest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

pytestmark = pytest.mark.local

from rag.llamaindex_store import TechCompassEmbedding  # noqa: E402
from rag.semantic_embedding import LOCAL_MODEL_PATH, SemanticEmbedder  # noqa: E402
from rag.store import RagStore  # noqa: E402


def test_local_model_exists():
    assert LOCAL_MODEL_PATH.exists()
    assert (LOCAL_MODEL_PATH / "model.safetensors").exists()


def test_embedding_dimension_and_norm():
    embedder = SemanticEmbedder()
    vectors = embedder.embed_batch([
        "battery swelling and fire risk",
        "shipping delay and packaging damage",
    ])
    for vector in vectors:
        assert len(vector) == 384
        assert abs(float(np.linalg.norm(vector)) - 1.0) < 1e-4


def test_semantic_similarity():
    embedder = SemanticEmbedder()
    vectors = embedder.embed_batch([
        "battery swelling and fire risk",
        "battery expansion and safety hazard",
        "shipping delay and packaging damage",
    ])
    similar = float(np.dot(vectors[0], vectors[1]))
    different = float(np.dot(vectors[0], vectors[2]))
    assert similar > different


def test_llamaindex_semantic_backend():
    embedding = TechCompassEmbedding(backend="semantic")
    vector = embedding._get_text_embedding("wireless earbuds safety")
    assert len(vector) == 384
    assert embedding.model_name == "BAAI/bge-small-en-v1.5"


def test_sqlite_embeddings_use_semantic_model():
    store = RagStore()
    try:
        rows = store.conn.execute(
            "SELECT DISTINCT model, dim FROM embeddings"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["model"] == "BAAI/bge-small-en-v1.5"
        assert rows[0]["dim"] == 384
    finally:
        store.close()


if __name__ == "__main__":
    tests = [
        test_local_model_exists,
        test_embedding_dimension_and_norm,
        test_semantic_similarity,
        test_llamaindex_semantic_backend,
        test_sqlite_embeddings_use_semantic_model,
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

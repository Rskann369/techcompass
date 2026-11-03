"""TechCompass — 重建 SQLite embeddings 为 BGE 语义向量。

流程：
1. 读取 chunks
2. 使用 BAAI/bge-small-en-v1.5 批量生成 384 维向量
3. 删除旧向量
4. 写入新向量
5. 记录 run

用法：
    PYTHONPATH=src ./.venv311/bin/python scripts/rebuild_semantic_embeddings.py
"""

import json
import time

from rag.semantic_embedding import SemanticEmbedder
from rag.store import RagStore


def rebuild(db_path: str = "data/rag/techcompass.db") -> dict:
    store = RagStore(db_path)
    rows = store.conn.execute("SELECT id, text FROM chunks ORDER BY id").fetchall()
    if not rows:
        store.close()
        return {"chunks": 0, "model": SemanticEmbedder.model_name, "dim": SemanticEmbedder.dim}

    embedder = SemanticEmbedder()
    run_id = store.start_run("rebuild_semantic_embeddings")
    started = time.perf_counter()

    texts = [row["text"] for row in rows]
    vectors = embedder.embed_batch(texts)

    store.conn.execute("DELETE FROM embeddings")
    for row, vector in zip(rows, vectors):
        store.conn.execute(
            "INSERT INTO embeddings(chunk_id, dim, vector, model, created_at) VALUES (?, ?, ?, ?, ?)",
            (row["id"], len(vector), vector.tobytes(), embedder.model_name, time.strftime("%Y-%m-%dT%H:%M:%S%z")),
        )
    store.conn.commit()

    elapsed = round((time.perf_counter() - started) * 1000, 3)
    metrics = {
        "chunks": len(rows),
        "model": embedder.model_name,
        "dim": embedder.dim,
        "elapsed_ms": elapsed,
    }
    store.finish_run(run_id, "success", metrics)
    stats = store.stats()
    store.close()
    return {**metrics, "stats": stats}


if __name__ == "__main__":
    print(json.dumps(rebuild(), ensure_ascii=False, indent=2))

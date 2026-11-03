"""修复 SQLite RAG 一致性：
1. 删除 orphan chunks / embeddings
2. 检查每个 document 是否有 chunks
3. 对缺失 chunks 的 documents 重新切分并生成 semantic embeddings
"""

import json

from rag.semantic_embedding import SemanticEmbedder
from rag.store import RagStore, chunk_text


def repair(db_path: str = "data/rag/techcompass.db") -> dict:
    store = RagStore(db_path)

    # 1) 删除 orphan chunks 和对应 embeddings
    orphan_chunks = store.conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE document_id NOT IN (SELECT id FROM documents)"
    ).fetchone()[0]
    store.conn.execute(
        "DELETE FROM embeddings WHERE chunk_id IN (SELECT id FROM chunks WHERE document_id NOT IN (SELECT id FROM documents))"
    )
    store.conn.execute(
        "DELETE FROM chunks WHERE document_id NOT IN (SELECT id FROM documents)"
    )
    store.conn.commit()

    # 2) 找没有 chunks 的 documents
    missing_docs = store.conn.execute(
        """
        SELECT d.id, d.title, d.content
        FROM documents d
        LEFT JOIN chunks c ON c.document_id=d.id
        WHERE c.id IS NULL
        """
    ).fetchall()

    embedder = SemanticEmbedder()
    rebuilt = 0
    for doc in missing_docs:
        text = f"{doc['title']}\n{doc['content']}"
        chunks = chunk_text(text)
        if not chunks:
            continue
        store.replace_chunks(doc["id"], chunks)
        rebuilt += 1

    # 3) 最终一致性检查
    orphan_after = store.conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE document_id NOT IN (SELECT id FROM documents)"
    ).fetchone()[0]
    orphan_embeddings = store.conn.execute(
        "SELECT COUNT(*) FROM embeddings WHERE chunk_id NOT IN (SELECT id FROM chunks)"
    ).fetchone()[0]
    docs_without_chunks = store.conn.execute(
        "SELECT COUNT(*) FROM documents d LEFT JOIN chunks c ON c.document_id=d.id WHERE c.id IS NULL"
    ).fetchone()[0]

    stats = store.stats()
    store.close()
    return {
        "orphan_chunks_deleted": orphan_chunks,
        "documents_rebuilt": rebuilt,
        "orphan_chunks_remaining": orphan_after,
        "orphan_embeddings_remaining": orphan_embeddings,
        "documents_without_chunks": docs_without_chunks,
        "stats": stats,
    }


if __name__ == "__main__":
    print(json.dumps(repair(), ensure_ascii=False, indent=2))

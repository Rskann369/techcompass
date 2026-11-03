"""清洗 SQLite RAG 中遗留的 HTML，并重建受影响文档的 chunks/embeddings。"""

import json
from adapters.text_cleaner import is_html_like, strip_html
from rag.store import RagStore, chunk_text


def clean_rag_html(db_path: str = "data/rag/techcompass.db") -> dict:
    store = RagStore(db_path)
    rows = store.conn.execute("SELECT id, title, content FROM documents").fetchall()
    cleaned_documents = 0
    cleaned_chunks = 0

    for row in rows:
        title = row["title"] or ""
        content = row["content"] or ""
        if not (is_html_like(title) or is_html_like(content)):
            continue

        new_title = strip_html(title)
        new_content = strip_html(content)
        store.conn.execute(
            "UPDATE documents SET title=?, content=? WHERE id=?",
            (new_title, new_content, row["id"]),
        )
        store.conn.commit()
        store.replace_chunks(row["id"], chunk_text(f"{new_title}\n{new_content}"))
        cleaned_documents += 1
        cleaned_chunks += len(chunk_text(f"{new_title}\n{new_content}"))

    remaining = store.conn.execute(
        "SELECT COUNT(*) FROM documents WHERE title LIKE '%<div>%' OR content LIKE '%<div>%'"
    ).fetchone()[0]
    stats = store.stats()
    store.close()
    return {
        "documents_scanned": len(rows),
        "documents_cleaned": cleaned_documents,
        "chunks_rebuilt": cleaned_chunks,
        "remaining_html_like_documents": remaining,
        "stats": stats,
    }


if __name__ == "__main__":
    print(json.dumps(clean_rag_html(), ensure_ascii=False, indent=2))

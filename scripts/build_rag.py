"""TechCompass — 构建 SQLite RAG 语料库（多源）。

统一数据流程：
1. MultiSourceFetcher 多源拉取（HackerNews / GNews / Reddit）
2. TechCompassDocument（统一 Schema）+ 去重
3. SQLite documents
4. chunks + embeddings
5. runs 记录

用法：
    PYTHONPATH=src ./.venv311/bin/python scripts/build_rag.py
"""

import time
from pathlib import Path

from dotenv import load_dotenv

from adapters.multi_source import build_default_fetcher
from rag.store import RagStore, chunk_text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

QUERIES = [
    "wireless earbuds",
    "smartwatch battery",
    "consumer electronics safety",
    "headphones charging issue",
    "gaming mouse review",
    "bluetooth speaker review",
    "laptop battery problem",
    "smartphone overheating",
    "usb c charger safety",
    "noise cancelling headphones",
    "smart home device privacy",
    "wireless charger recall",
    "camera firmware update",
    "tablet screen issue",
    "drone battery safety",
    "electric scooter recall",
    "wireless earbuds review",
    "smartwatch review",
    "bluetooth speaker battery",
    "gaming keyboard review",
    "smart home camera privacy",
    "product recall safety hazard",
    "battery fire risk",
    "laptop battery recall",
    "consumer electronics data breach",
    "smart home device security vulnerability",
    "electric blanket fire recall",
    "power bank explosion risk",
    "child safety product recall",
    "hoverboard fire hazard",
    "USB charger electric shock",
]


def main():
    fetcher = build_default_fetcher(env_path=str(PROJECT_ROOT / ".env"))
    store = RagStore()
    run_id = store.start_run("build_rag")
    inserted = 0
    chunks_created = 0
    source_counts = {}
    errors = []

    for query in QUERIES:
        print(f"[fetch] {query}", flush=True)
        result = fetcher.search(query, limit_per_source=5)
        source_counts[query] = result.per_source
        for name, err in result.errors.items():
            errors.append({"source": name, "query": query, "error": err})

        for document in result.documents:
            doc_id = store.upsert_document(document.to_rag_document())
            if doc_id:
                doc_text = f"{document.title}\n{document.content}"
                chunks = chunk_text(doc_text)
                store.replace_chunks(doc_id, chunks)
                inserted += 1
                chunks_created += len(chunks)
        time.sleep(1)

    stats = store.stats()
    metrics = {
        "documents_inserted": inserted,
        "chunks_created": chunks_created,
        "source_counts": source_counts,
        "errors": errors,
        "stats": stats,
    }
    store.finish_run(run_id, "success" if not errors else "partial", metrics)
    print("\n=== RAG build summary ===")
    print(f"documents inserted/updated: {inserted}")
    print(f"chunks created: {chunks_created}")
    print(f"stats: {stats}")
    print(f"errors: {errors}")
    store.close()


if __name__ == "__main__":
    main()

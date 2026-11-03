"""TechCompass — 自动生成 RAG 评测集评估入口。"""
import json
from rag.llamaindex_store import build_llamaindex_index, llamaindex_search
from rag.store import RagStore

DB = "data/rag/techcompass.db"
EVALSET = "data/evalsets/generated_rag_queries.jsonl"


def main():
    with open(EVALSET, encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    index = build_llamaindex_index(DB)
    store = RagStore(DB)
    rows = store.conn.execute("SELECT id, title FROM documents").fetchall()
    title_to_id = {r["title"]: r["id"] for r in rows}
    store.close()

    top_k = 5
    hit = 0
    recalls, precisions, mrrs, details = [], [], [], []

    for case in cases:
        query = case["query"]
        expected = {title_to_id[t] for t in case.get("relevant_titles", []) if t in title_to_id}
        if not expected:
            continue
        results = llamaindex_search(index, query, top_k=top_k)
        returned_raw = [title_to_id[r["title"]] for r in results if r.get("title") in title_to_id]
        returned = list(dict.fromkeys(returned_raw))[:top_k]
        relevant_returned = [d for d in returned if d in expected]
        recall = len(relevant_returned) / len(expected) if expected else 0
        precision = len(relevant_returned) / len(returned) if returned else 0
        mrr = 0.0
        for rank, d in enumerate(returned, 1):
            if d in expected:
                mrr = 1 / rank
                break
        hit += bool(relevant_returned)
        recalls.append(recall)
        precisions.append(precision)
        mrrs.append(mrr)
        details.append({"query": query, "expected": sorted(expected), "returned": returned,
                        "recall": round(recall, 4), "precision": round(precision, 4), "mrr": round(mrr, 4)})

    n = len(details)
    print(json.dumps({
        "n": n,
        "top_k": top_k,
        "hit_rate": round(hit/n, 4) if n else 0,
        "avg_recall": round(sum(recalls)/n, 4) if n else 0,
        "avg_precision": round(sum(precisions)/n, 4) if n else 0,
        "mrr": round(sum(mrrs)/n, 4) if n else 0,
        "details": details,
    }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()

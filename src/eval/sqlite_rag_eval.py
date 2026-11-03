"""TechCompass — SQLite RAG 检索评估器。

评估对象：本地 SQLite RAG（真实 GNews/Reddit 文档）。

指标：
- hit_rate      ：至少命中一个相关 chunk 的 query 比例
- recall@k      ：相关 chunk 被召回比例
- precision@k   ：返回结果中相关比例
- mrr@k         ：第一个相关结果的倒数排名
- ndcg@k        ：排序质量
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from rag.store import RagStore, search_store


def load_evalset(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def dcg(relevances: list[int]) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))


def evaluate(
    db_path: str,
    evalset_path: str,
    top_k: int = 5,
) -> dict[str, Any]:
    store = RagStore(db_path)
    queries = load_evalset(evalset_path)

    hit = 0
    recalls = []
    precisions = []
    mrrs = []
    ndcgs = []
    details = []

    for q in queries:
        query = q["query"]
        relevant = set(q["relevant_document_ids"])
        results = search_store(store, query, top_k=top_k)
        result_doc_ids = [r.document_id for r in results]

        relevant_returned = [doc_id for doc_id in result_doc_ids if doc_id in relevant]
        recall = len(relevant_returned) / len(relevant) if relevant else 0.0
        precision = len(relevant_returned) / len(result_doc_ids) if result_doc_ids else 0.0

        mrr = 0.0
        for rank, doc_id in enumerate(result_doc_ids, start=1):
            if doc_id in relevant:
                mrr = 1 / rank
                break

        ideal = [1] * min(len(relevant), top_k) + [0] * max(0, top_k - len(relevant))
        actual = [1 if doc_id in relevant else 0 for doc_id in result_doc_ids]
        actual += [0] * max(0, top_k - len(actual))
        ndcg = dcg(actual) / dcg(ideal) if dcg(ideal) else 0.0

        recalls.append(recall)
        precisions.append(precision)
        mrrs.append(mrr)
        ndcgs.append(ndcg)
        hit += bool(relevant_returned)

        details.append({
            "query": query,
            "relevant_document_ids": sorted(relevant),
            "returned_document_ids": result_doc_ids,
            "relevant_returned": relevant_returned,
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "mrr": round(mrr, 4),
            "ndcg": round(ndcg, 4),
        })

    n = len(queries)
    store.close()
    return {
        "n": n,
        "top_k": top_k,
        "hit_rate": round(hit / n, 4) if n else 0,
        "avg_recall": round(sum(recalls) / n, 4) if n else 0,
        "avg_precision": round(sum(precisions) / n, 4) if n else 0,
        "mrr": round(sum(mrrs) / n, 4) if n else 0,
        "ndcg": round(sum(ndcgs) / n, 4) if n else 0,
        "details": details,
    }


def report(summary: dict[str, Any]) -> str:
    lines = [
        f"Query 数: {summary['n']}  top_k={summary['top_k']}",
        f"Hit rate: {summary['hit_rate']:.2%}",
        f"Recall@{summary['top_k']}: {summary['avg_recall']:.2%}",
        f"Precision@{summary['top_k']}: {summary['avg_precision']:.2%}",
        f"MRR@{summary['top_k']}: {summary['mrr']:.2%}",
        f"NDCG@{summary['top_k']}: {summary['ndcg']:.2%}",
        "",
        "明细：",
    ]
    for d in summary["details"]:
        lines.append(
            f"  [{d['query']}] recall={d['recall']:.2f}, precision={d['precision']:.2f}, "
            f"mrr={d['mrr']:.2f}, ndcg={d['ndcg']:.2f}"
        )
        lines.append(f"    expected={d['relevant_document_ids']}")
        lines.append(f"    returned={d['returned_document_ids']}")
    return "\n".join(lines)

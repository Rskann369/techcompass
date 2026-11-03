"""TechCompass — 检索召回评估器。

指标:
- hit_rate      ：query 是否至少命中一条相关文档
- avg_recall    ：平均召回率
- avg_precision ：平均精确率
"""

from __future__ import annotations

import json
from typing import Any

from tools.domain_tools import make_search_reviews


def load_queries(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_retrieval_eval(
    corpus_path: str,
    queries_path: str,
    top_k: int = 3,
) -> dict[str, Any]:
    tool = make_search_reviews(corpus_path)
    queries = load_queries(queries_path)

    hit_count = 0
    recall_sum = 0.0
    precision_sum = 0.0
    details = []

    for q in queries:
        query = q["query"]
        true_ids = set(q["relevant_ids"])
        result = tool.run({"query": query, "max": top_k})
        result_ids = {r.get("id") for r in result.get("reviews", [])}

        relevant_returned = result_ids & true_ids
        recall = len(relevant_returned) / len(true_ids) if true_ids else 0.0
        precision = len(relevant_returned) / len(result_ids) if result_ids else 0.0
        if relevant_returned:
            hit_count += 1
        recall_sum += recall
        precision_sum += precision

        details.append({
            "query": query,
            "true_count": len(true_ids),
            "returned_ids": list(result_ids),
            "relevant_returned_count": len(relevant_returned),
            "recall": round(recall, 4),
            "precision": round(precision, 4),
        })

    n = len(queries)
    return {
        "n": n,
        "hit_rate": round(hit_count / n, 4) if n else 0,
        "avg_recall": round(recall_sum / n, 4) if n else 0,
        "avg_precision": round(precision_sum / n, 4) if n else 0,
        "details": details,
    }


def report(summary: dict[str, Any]) -> str:
    lines = [
        f"查询数: {summary['n']}",
        f"命中率(hit_rate): {summary['hit_rate']:.2%}",
        f"平均召回率(avg_recall): {summary['avg_recall']:.2%}",
        f"平均精确率(avg_precision): {summary['avg_precision']:.2%}",
        "",
        "查询明细：",
    ]
    for d in summary["details"]:
        lines.append(
            f"  [{d['query']}] 应命中 {d['true_count']} 条，"
            f"实际返回 {len(d['returned_ids'])} 条，"
            f"命中 {d['relevant_returned_count']} 条 "
            f"(recall={d['recall']}, precision={d['precision']})"
        )
    return "\n".join(lines)

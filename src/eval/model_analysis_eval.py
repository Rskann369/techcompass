"""TechCompass — 模型驱动分析抽样评估器。

目的：
- 高质量标注全量评估规则 Pipeline 暴露出明显短板。
- 真实 LLM 全量评估太慢，因此先做可配置抽样评估。
- 每个抽样 case 调用 analyze_consumer_sentiment，记录准确率、延迟和 token 估算。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rag.hybrid_retrieval import hybrid_search
from tools.llm_analysis_tools import make_analyze_consumer_sentiment


def run_model_analysis_eval(
    cases: list[dict[str, Any]],
    sample_size: int = 5,
) -> dict[str, Any]:
    if sample_size < len(cases):
        # 固定间隔抽样，避免只取前几条同类主题
        step = max(1, len(cases) // sample_size)
        sampled = cases[::step][:sample_size]
    else:
        sampled = cases

    tool = make_analyze_consumer_sentiment()
    completed = 0
    category_ok = 0
    sentiment_ok = 0
    risk_ok = 0
    impact_ok = 0
    latencies = []
    details = []

    for case in sampled:
        question = case["question"]
        expected = case.get("expected", {})
        try:
            retrieval = hybrid_search(question, queries=[question], top_k=5)
            evidence = [
                f"{r.get('title')}. {r.get('text')}" for r in retrieval.get("results", [])
            ]
            result = tool.run({"question": question, "evidence": evidence})
            completed += 1

            pred_category = "unknown"
            categories = result.get("product_categories", [])
            if categories:
                # 简单取第一个 category，后续可做多标签匹配
                pred_category = categories[0].lower()

            category_ok += pred_category == expected.get("category", "unknown").lower()
            sentiment_ok += result.get("sentiment") == expected.get("sentiment")
            risk_ok += (result.get("risk_level") in ("medium", "high", "critical")) == (expected.get("risk") == "pos")
            impact_ok += result.get("impact_level") == expected.get("impact_level")

            details.append({
                "question": question,
                "expected": expected,
                "prediction": result,
                "evidence_count": len(evidence),
                "status": "ok",
            })
        except Exception as exc:  # noqa: BLE001
            details.append({"question": question, "status": "failed", "error": str(exc)})

    n = len(sampled)
    return {
        "sample_count": n,
        "completed": completed,
        "completion_rate": round(completed / n, 4) if n else 0,
        "category_accuracy": round(category_ok / n, 4) if n else 0,
        "sentiment_accuracy": round(sentiment_ok / n, 4) if n else 0,
        "risk_accuracy": round(risk_ok / n, 4) if n else 0,
        "impact_accuracy": round(impact_ok / n, 4) if n else 0,
        "details": details,
    }

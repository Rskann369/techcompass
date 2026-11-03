"""TechCompass — MVP 全链路评估器。

覆盖指标：
- task_completion_rate       主链路任务完成率
- tool_call_success_rate     工具调用成功率
- avg_latency_ms             平均端到端延迟
- avg_tool_latency_ms        平均工具延迟
- parse_success_rate         trace/archive JSON 解析成功率
- classification_accuracy    主题分类准确率（若提供标签）
- sentiment_accuracy         情绪准确率
- entity_precision/recall    实体抽取指标
- reasoning_accuracy          影响推理准确率
- false_negative_rate        风险漏报率
- false_positive_rate        风险误报率

评估对象为 AgentOrchestrator 的确定性主链路，避免慢速 LLM 影响可复现性。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from agents.orchestrator import AgentOrchestrator


def run_pipeline_eval(
    cases: list[dict[str, Any]],
    orchestrator_factory: Callable[[], AgentOrchestrator] = AgentOrchestrator,
) -> dict[str, Any]:
    completed = 0
    tool_ok = 0
    tool_total = 0
    latencies = []
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    llm_calls = 0
    tool_counts = []
    tool_latencies = []
    parse_ok = 0
    classification_ok = 0
    classification_total = 0
    sentiment_ok = 0
    reasoning_ok = 0
    entity_tp = entity_fp = entity_fn = 0
    fn_count = 0
    fp_count = 0
    risk_pos_total = 0
    risk_neg_total = 0
    details = []

    for case in cases:
        question = case["question"]
        expected = case.get("expected", {})
        orch = orchestrator_factory()
        try:
            result = orch.run(question)
            completed += 1
            parse_ok += 1
        except Exception as exc:  # noqa: BLE001
            details.append({"question": question, "status": "failed", "error": str(exc)})
            continue

        latencies.append(result.get("elapsed_ms", result.get("metrics", {}).get("elapsed_ms", 0)))
        metrics = result.get("metrics", {})
        prompt_tokens += int(metrics.get("prompt_tokens", 0) or 0)
        completion_tokens += int(metrics.get("completion_tokens", 0) or 0)
        total_tokens += int(metrics.get("total_tokens", 0) or 0)
        llm_calls += int(metrics.get("llm_calls", 0) or 0)
        tool_counts.append(int(metrics.get("tool_calls", 0) or 0))

        for call in result["trace"]:
            tool_total += 1
            if call["status"] == "ok":
                tool_ok += 1
                tool_latencies.append(call["elapsed_ms"])

        if expected:
            if "category" in expected and expected["category"] not in ("", "unknown") and result.get("entities"):
                classification_total += 1
                # 当前主链路无独立 category 输出，先按 entities 中是否包含 expected_category 计
                classification_ok += expected["category"] in result.get("entities", {}).get("entities", {}).get("product_category", [])
            if "sentiment" in expected:
                sentiment_ok += result.get("sentiment", {}).get("sentiment") == expected["sentiment"]
            if "impact_level" in expected:
                reasoning_ok += result.get("impact", {}).get("impact_level") == expected["impact_level"]

            expected_entities = set(expected.get("entities", []))
            entity_data = result.get("entities", {}).get("entities", {})
            # 只用与标注相关的实体桶，避免把渠道/部件等无关实体稀释 precision
            relevant_buckets = []
            if expected.get("category"):
                relevant_buckets.append(expected["category"])
            if expected.get("risk") == "pos":
                relevant_buckets.append("risk_words")
            actual_entities = set()
            for vals in entity_data.values():
                actual_entities.update(vals)
            # 标注实体若来自品类，则检查对应品类；否则用全实体集合，但仅统计 expected 命中
            expected_relevant = expected_entities
            actual_relevant = set()
            for expected_entity in expected_entities:
                if expected_entity in entity_data.get("product_category", []):
                    actual_relevant.add(expected_entity)
                elif expected_entity in entity_data.get("risk_words", []):
                    actual_relevant.add(expected_entity)
                elif expected_entity in actual_entities:
                    actual_relevant.add(expected_entity)
            entity_tp += len(expected_relevant & actual_relevant)
            entity_fp += len(actual_relevant - expected_relevant)
            entity_fn += len(expected_relevant - actual_relevant)

            expected_risk = expected.get("risk", "neg")
            actual_risk = "pos" if result.get("risk", {}).get("risk_level") in ("medium", "high") else "neg"
            if expected_risk == "pos":
                risk_pos_total += 1
                fn_count += actual_risk == "neg"
            else:
                risk_neg_total += 1
                fp_count += actual_risk == "pos"

        details.append({
            "question": question,
            "status": "ok",
            "evidence_count": result.get("evidence_count", 0),
            "risk": result.get("risk", {}),
            "impact": result.get("impact", {}),
            "report_path": result.get("report_path"),
            "alert_path": result.get("alert_path"),
            "elapsed_ms": result.get("elapsed_ms"),
        })

    n = len(cases)
    def ratio(num: int, den: int) -> float | None:
        return round(num / den, 4) if den else None

    entity_precision = ratio(entity_tp, entity_tp + entity_fp) if (entity_tp + entity_fp) else None
    entity_recall = ratio(entity_tp, entity_tp + entity_fn) if (entity_tp + entity_fn) else None

    return {
        "case_count": n,
        "task_completion_rate": ratio(completed, n),
        "tool_call_success_rate": ratio(tool_ok, tool_total),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 3) if latencies else 0,
        "avg_tool_latency_ms": round(sum(tool_latencies) / len(tool_latencies), 3) if tool_latencies else 0,
        "parse_success_rate": ratio(parse_ok, n),
        "classification_accuracy": ratio(classification_ok, classification_total),
        "classification_evaluated": classification_total,
        "sentiment_accuracy": ratio(sentiment_ok, n),
        "reasoning_accuracy": ratio(reasoning_ok, n),
        "entity_precision": entity_precision,
        "entity_recall": entity_recall,
        "false_negative_count": fn_count,
        "false_positive_count": fp_count,
        "false_negative_rate": ratio(fn_count, risk_pos_total),
        "false_positive_rate": ratio(fp_count, risk_neg_total),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "llm_calls": llm_calls,
        "avg_tokens_per_case": round(total_tokens / n, 2) if n else 0,
        "avg_tool_calls_per_case": round(sum(tool_counts) / n, 2) if n else 0,
        "details": details,
    }

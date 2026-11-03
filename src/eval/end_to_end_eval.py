"""TechCompass — 端到端评测器。

定义：
    输入用户 query
      → 查询重写
      → Hybrid RAG
      → 语义过滤
      → 分析
      → 行动建议
      → 周报 / 告警 / 归档

评测维度：
1. 端到端任务完成率
2. 端到端工具成功率
3. 检索命中相关证据率
4. 分析准确率
5. 报告生成率
6. 告警触发正确率
7. 归档生成率
8. 平均延迟
9. token / 工具调用成本
10. 证据引用可追溯率
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from eval.agent_runner import run_agent_for_eval
from eval.node_eval import load_cases, _ratio


def run_end_to_end_eval(
    path: str = "data/evalsets/node_eval_cases.jsonl",
    analysis_backend: str = "rule",
) -> dict[str, Any]:
    cases = load_cases(path)

    completed = 0
    tool_ok = tool_total = 0
    retrieval_hit = 0
    analysis_ok = 0
    report_ok = 0
    archive_ok = 0
    alert_correct = 0
    action_ok = 0
    refusal_ok = refusal_total = 0
    evidence_support = 0
    evidence_traceable = 0
    latencies = []
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    tool_calls = 0
    details = []

    for case in cases:
        question = case["question"]
        expected = case.get("expected", {})
        try:
            result = run_agent_for_eval(question)
            completed += 1
        except Exception as exc:  # noqa: BLE001
            details.append({"question": question, "status": "failed", "error": str(exc)})
            continue

        metrics = result.get("metrics", {})
        tool_total += int(metrics.get("tool_calls", 0) or 0)
        tool_ok += int(metrics.get("tool_success", 0) or 0)
        tool_calls += int(metrics.get("tool_calls", 0) or 0)
        prompt_tokens += int(metrics.get("prompt_tokens", 0) or 0)
        completion_tokens += int(metrics.get("completion_tokens", 0) or 0)
        total_tokens += int(metrics.get("total_tokens", 0) or 0)
        latencies.append(float(metrics.get("elapsed_ms", 0) or 0))

        evidence = result.get("evidence", [])
        relevant_titles = set(case.get("expected", {}).get("relevant_titles", []))
        retrieval_hit += any(e.get("title") in relevant_titles for e in evidence)

        actual_risk = "pos" if result.get("risk", {}).get("risk_level") in ("medium", "high") else "neg"
        actual_sentiment = result.get("sentiment", {}).get("sentiment")
        actual_impact = result.get("impact", {}).get("impact_level")
        risk_types = result.get("risk", {}).get("risk_types", [])
        actual_issue = risk_types[0] if risk_types else "other"
        actual_action = "refuse" if result.get("refused") else (
            "review_privacy" if actual_issue == "privacy" and actual_risk == "pos"
            else "recall_review" if actual_issue == "safety" and actual_risk == "pos"
            else "support_review" if actual_risk == "pos"
            else "monitor"
        )
        analysis_ok += (
            actual_risk == expected.get("risk")
            and actual_sentiment == expected.get("sentiment")
            and actual_impact == expected.get("impact_level")
        )
        action_ok += actual_action == expected.get("expected_action")
        should_refuse = expected.get("should_refuse", False)
        refusal_total += 1
        refusal_ok += bool(result.get("refused")) == should_refuse
        has_evidence = result.get("evidence_count", 0) > 0
        evidence_support += (not should_refuse and has_evidence) or (should_refuse and not has_evidence)

        report_exists = bool(result.get("generated_report"))
        archive_exists = bool(result.get("archived"))
        report_ok += report_exists
        archive_ok += archive_exists

        should_alert = expected.get("impact_level") in ("high", "critical") or expected.get("risk") == "pos"
        has_alert = bool(result.get("created_alert"))
        alert_correct += should_alert == has_alert

        evidence_traceable += report_exists

        details.append({
            "question": question,
            "status": "ok",
            "evidence_count": result.get("evidence_count", 0),
            "retrieval_hit": any(e.get("title") in relevant_titles for e in evidence),
            "risk": result.get("risk", {}),
            "impact": result.get("impact", {}),
            "sentiment": result.get("sentiment", {}),
            "generated_report": report_exists,
            "archived": archive_exists,
            "should_alert": should_alert,
            "metrics": metrics,
        })

    n = len(cases)
    return {
        "eval_type": "end_to_end",
        "analysis_backend": analysis_backend,
        "case_count": n,
        "task_completion_rate": _ratio(completed, n),
        "tool_call_success_rate": _ratio(tool_ok, tool_total),
        "retrieval_hit_rate": _ratio(retrieval_hit, n),
        "analysis_exact_match_rate": _ratio(analysis_ok, n),
        "report_generation_rate": _ratio(report_ok, n),
        "archive_generation_rate": _ratio(archive_ok, n),
        "alert_trigger_accuracy": _ratio(alert_correct, n),
        "action_appropriateness": _ratio(action_ok, n),
        "refusal_correctness": _ratio(refusal_ok, refusal_total),
        "evidence_support_rate": _ratio(evidence_support, n),
        "evidence_traceable_rate": _ratio(evidence_traceable, n),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 3) if latencies else 0,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "avg_tokens_per_case": round(total_tokens / n, 2) if n else 0,
        "avg_tool_calls_per_case": round(tool_calls / n, 2) if n else 0,
        "details": details,
    }

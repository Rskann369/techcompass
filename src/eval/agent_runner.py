"""TechCompass — 真实 LLM Agent 评估运行器。

把 LangGraphReactAgent（真实 LLM）的结果归一成评测器能消费的字段形状，
使 node_eval / end_to_end_eval 无需理解 Agent 内部消息结构。
"""

from __future__ import annotations

from typing import Any


def run_agent_for_eval(question: str) -> dict[str, Any]:
    from agents.langgraph_react_agent import LangGraphReactAgent

    result = LangGraphReactAgent().run(question)
    analysis = result.get("analysis") or {}
    tool_names = {c.get("tool") for c in result.get("tool_calls", [])}

    return {
        "refused": (not analysis) or (not analysis.get("evidence_sufficient", True)),
        "entities": {"entities": {"product_category": analysis.get("product_categories", [])}},
        "sentiment": {"sentiment": analysis.get("sentiment", "neutral")},
        "risk": {
            "risk_level": analysis.get("risk_level", "none"),
            "risk_types": [analysis.get("risk_type")] if analysis.get("risk_type") else [],
        },
        "impact": {"impact_level": analysis.get("impact_level", "none")},
        "evidence_count": len(result.get("relevant_evidence", [])),
        "evidence": result.get("relevant_evidence", []),
        "metrics": result.get("metrics", {}),
        "generated_report": "generate_weekly_report" in tool_names,
        "created_alert": "create_alert" in tool_names,
        "archived": "archive_run" in tool_names,
    }

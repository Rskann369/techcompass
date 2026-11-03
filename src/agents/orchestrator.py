"""TechCompass — Agent 主链路编排器。

主链路：
    rewrite_query
      -> rag_search
      -> deduplicate_docs
      -> extract_entities
      -> analyze_sentiment
      -> score_risk
      -> reason_impact
      -> suggest_action
      -> generate_weekly_report
      -> create_alert
      -> archive_run

设计原则：
- 先提供确定性可测试的 orchestrator，保证主链路可运行、可评估。
- 每步记录 tool、输入摘要、输出摘要、耗时、错误。
- 外部 RAG 检索失败时允许降级为空证据，后续节点仍能输出“证据不足”。
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any

from tools.domain_tools import build_registry


class AgentOrchestrator:
    def __init__(self, db_path: str = "data/rag/techcompass.db", dataset_path: str = "data/samples/nonexistent.jsonl"):
        self.registry = build_registry(dataset_path=dataset_path)
        self.db_path = db_path
        self.trace: list[dict[str, Any]] = []

    def _call(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        tool = self.registry.get(tool_name)
        if tool is None:
            raise KeyError(f"unknown tool: {tool_name}")
        started = time.perf_counter()
        try:
            result = tool.execute(args)
            elapsed = round((time.perf_counter() - started) * 1000, 3)
            record = {
                "tool": tool_name,
                "status": "ok",
                "elapsed_ms": elapsed,
                "args": args,
                "result_keys": list(result.keys()) if isinstance(result, dict) else [],
            }
            self.trace.append(record)
            return result
        except Exception as exc:  # noqa: BLE001
            elapsed = round((time.perf_counter() - started) * 1000, 3)
            self.trace.append({
                "tool": tool_name,
                "status": "failed",
                "elapsed_ms": elapsed,
                "args": args,
                "error": str(exc),
            })
            raise

    def run(self, question: str) -> dict[str, Any]:
        started_at = datetime.now(timezone.utc).isoformat()
        overall_start = time.perf_counter()

        # 1. Query rewrite
        rewrite = self._call("rewrite_query", {"query": question})
        queries = rewrite.get("rewrites", [question]) or [question]

        # 2. RAG retrieval
        rag = self._call("rag_search", {"query": queries[0], "top_k": 5})
        results = rag.get("results", [])

        # 3. Deduplicate documents
        docs = [
            {
                "document_id": r.get("document_id"),
                "title": r.get("title"),
                "content": r.get("text"),
                "url": r.get("url"),
                "published_at": r.get("published_at"),
            }
            for r in results
        ]
        dedup = self._call("deduplicate_docs", {"documents": docs})
        unique_docs = dedup.get("unique_documents", [])

        # 4-6. Evidence analysis
        evidence_text = "\n".join(
            f"{d.get('title', '')}. {d.get('content', '')}" for d in unique_docs
        ) or question
        entities = self._call("extract_entities", {"text": evidence_text})
        sentiment = self._call("analyze_sentiment", {"text": evidence_text})
        risk = self._call("score_risk", {"text": evidence_text})

        # 7. Impact reasoning
        impact = self._call("reason_impact", {
            "risk_level": risk.get("risk_level", "low"),
            "evidence_count": len(unique_docs),
            "spread": "multiple" if len(unique_docs) > 1 else "single",
        })
        impact_level = impact.get("impact_level", "low")

        # 8. Action suggestion
        action = self._call("suggest_action", {"impact_level": impact_level})

        # 9. Weekly report
        findings = [
            f"{d.get('title', '')} | {d.get('published_at', '')} | {d.get('url', '')}"
            for d in unique_docs
        ]
        report = self._call("generate_weekly_report", {
            "topic": question,
            "findings": findings,
            "impact_level": impact_level,
            "actions": action.get("actions", {}),
        })

        # 10. Alert if needed
        alert_path = None
        if impact_level in ("high", "critical") or risk.get("risk_level") in ("medium", "high"):
            alert = self._call("create_alert", {
                "title": f"TechCompass 舆情告警：{question}",
                "level": "critical" if impact_level == "critical" else "high" if impact_level == "high" else "medium",
                "evidence": findings,
            })
            alert_path = alert.get("alert_path")

        # 11. Archive run
        archive = self._call("archive_run", {
            "status": "success",
            "summary": {
                "question": question,
                "impact_level": impact_level,
                "risk_level": risk.get("risk_level"),
                "evidence_count": len(unique_docs),
                "report_path": report.get("report_path"),
                "alert_path": alert_path,
            },
        })

        total_elapsed_ms = round((time.perf_counter() - overall_start) * 1000, 3)
        return {
            "ok": True,
            "started_at": started_at,
            "question": question,
            "queries": queries,
            "evidence_count": len(unique_docs),
            "evidence": unique_docs,
            "entities": entities,
            "sentiment": sentiment,
            "risk": risk,
            "impact": impact,
            "actions": action,
            "report_path": report.get("report_path"),
            "alert_path": alert_path,
            "archive_path": archive.get("archive_path"),
            "trace": self.trace,
            "elapsed_ms": total_elapsed_ms,
        }


if __name__ == "__main__":
    agent = AgentOrchestrator()
    print(json.dumps(agent.run("wireless earbuds safety"), ensure_ascii=False, indent=2))

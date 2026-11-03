"""TechCompass — 输出与数据操作工具层。

覆盖目标中的：
- 周报生成
- 告警生成
- CSV 导出
- 历史查询
- 文档去重
- 运行归档

这些工具负责把分析结果沉淀为可交付产物和可追溯数据。
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
import uuid
from pathlib import Path
from typing import Any

from tools.registry import Tool

OUTPUT_ROOT = Path("outputs")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_generate_weekly_report() -> Tool:
    def run(args: dict[str, Any]) -> dict[str, Any]:
        topic = str(args.get("topic", "消费品舆情"))
        findings = args.get("findings", [])
        impact = str(args.get("impact_level", "low"))
        actions = args.get("actions", {})
        timestamp = _now()
        filename = f"weekly_report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.md"
        path = OUTPUT_ROOT / "weekly_reports" / filename
        path.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            f"# TechCompass 周报 — {topic}",
            "",
            f"- 生成时间：{timestamp}",
            f"- 影响等级：{impact}",
            "",
            "## 核心发现",
        ]
        if findings:
            for i, f in enumerate(findings, 1):
                lines.append(f"{i}. {f}")
        else:
            lines.append("暂无明确发现。")

        lines += ["", "## 行动建议"]
        for key, label in (("customer_support", "客服"), ("product", "产品"), ("communication", "传播")):
            if key in actions:
                lines.append(f"- {label}: {actions[key]}")
        if not actions:
            lines.append("- 常规监测。")

        content = "\n".join(lines) + "\n"
        path.write_text(content, encoding="utf-8")
        return {"report_path": str(path), "topic": topic, "impact_level": impact}
    return Tool(
        name="generate_weekly_report",
        description="生成 Markdown 周报并保存到 outputs/weekly_reports。",
        args_schema={
            "type": "object",
            "properties": {
                "topic": {"type": "string"},
                "findings": {"type": "array", "items": {"type": "string"}},
                "impact_level": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                "actions": {"type": "object"},
            },
            "required": ["topic"],
        },
        run=run,
        fallback=lambda: {"report_path": "", "error": "report generation failed"},
    )


def make_create_alert() -> Tool:
    def run(args: dict[str, Any]) -> dict[str, Any]:
        title = str(args.get("title", "未命名告警"))
        level = str(args.get("level", "medium"))
        evidence = args.get("evidence", [])
        timestamp = _now()
        alert = {
            "title": title,
            "level": level,
            "evidence": evidence,
            "created_at": timestamp,
        }
        path = OUTPUT_ROOT / "alerts" / f"alert_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(alert, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"alert_path": str(path), "alert": alert}
    return Tool(
        name="create_alert",
        description="创建舆情告警 JSON 文件，用于高风险或需人工确认事件。",
        args_schema={
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "level": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                "evidence": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["title", "level"],
        },
        run=run,
        fallback=lambda: {"alert_path": "", "error": "alert failed"},
    )


def make_export_csv() -> Tool:
    def run(args: dict[str, Any]) -> dict[str, Any]:
        rows = args.get("rows", [])
        path_arg = str(args.get("path", "export.csv"))
        root = (Path.cwd() / "outputs").resolve()
        path = (root / path_arg).resolve()
        # 路径白名单：只允许写入 outputs/ 内
        if root not in path.parents:
            raise ValueError("export path must stay inside outputs/")
        path.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            path.write_text("", encoding="utf-8")
            return {"path": str(path), "row_count": 0}
        fields = list(rows[0].keys())
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return {"path": str(path), "row_count": len(rows), "fields": fields}
    return Tool(
        name="export_csv",
        description="将结构化结果导出为 CSV 文件。",
        args_schema={
            "type": "object",
            "properties": {
                "rows": {"type": "array", "items": {"type": "object"}},
                "path": {"type": "string"},
            },
            "required": ["rows"],
        },
        run=run,
        fallback=lambda: {"path": "", "row_count": 0},
    )


def make_query_history() -> Tool:
    def run(args: dict[str, Any]) -> dict[str, Any]:
        from rag.store import RagStore
        query = str(args.get("query", ""))
        store = RagStore()
        try:
            rows = store.conn.execute(
                "SELECT query, top_k, chunk_ids, scores, created_at FROM retrievals WHERE query LIKE ? ORDER BY id DESC LIMIT 20",
                (f"%{query}%",),
            ).fetchall()
            return {
                "count": len(rows),
                "history": [dict(r) for r in rows],
            }
        finally:
            store.close()
    return Tool(
        name="query_history",
        description="查询历史 RAG 检索记录，支持追溯和复盘。",
        args_schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        run=run,
        fallback=lambda: {"count": 0, "history": []},
    )


def make_deduplicate_docs() -> Tool:
    def run(args: dict[str, Any]) -> dict[str, Any]:
        docs = args.get("documents", [])
        seen = set()
        unique = []
        duplicates = []
        for d in docs:
            key = str(d.get("title", "")) + "\n" + str(d.get("content", ""))
            if key in seen:
                duplicates.append(d)
            else:
                seen.add(key)
                unique.append(d)
        return {"unique_count": len(unique), "duplicate_count": len(duplicates), "unique_documents": unique}
    return Tool(
        name="deduplicate_docs",
        description="按 title+content 去重文档列表，返回唯一文档和重复数量。",
        args_schema={
            "type": "object",
            "properties": {"documents": {"type": "array", "items": {"type": "object"}}},
            "required": ["documents"],
        },
        run=run,
        fallback=lambda: {"unique_count": 0, "duplicate_count": 0, "unique_documents": []},
    )


def make_archive_run() -> Tool:
    def run(args: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "status": args.get("status", "success"),
            "summary": args.get("summary", {}),
            "archived_at": _now(),
        }
        path = OUTPUT_ROOT / "runs" / f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"archive_path": str(path), "status": payload["status"]}
    return Tool(
        name="archive_run",
        description="将一次 Agent/Workflow 运行结果归档为 JSON，便于评估与追溯。",
        args_schema={
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["success", "partial", "failed"]},
                "summary": {"type": "object"},
            },
            "required": ["status"],
        },
        run=run,
        fallback=lambda: {"archive_path": "", "status": "failed"},
    )


def register_output_tools(registry) -> None:
    """只注册主链路必需的输出工具。

    export_csv / query_history / agent_history / agent_trace 属于人用管理功能，
    不暴露给 LLM Agent。
    """
    registry.register(make_deduplicate_docs())
    registry.register(make_generate_weekly_report())
    registry.register(make_create_alert())
    registry.register(make_archive_run())


def make_agent_history() -> Tool:
    def run(args: dict[str, Any]) -> dict[str, Any]:
        from rag.store import RagStore
        limit = int(args.get("limit", 10))
        store = RagStore()
        try:
            rows = store.list_agent_runs(limit=limit)
            return {"count": len(rows), "runs": rows}
        finally:
            store.close()
    return Tool(
        name="agent_history",
        description="查询最近的 Agent 运行记录，包括问题、状态、指标和输出路径。",
        args_schema={
            "type": "object",
            "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 100}},
            "required": [],
        },
        run=run,
        fallback=lambda: {"count": 0, "runs": []},
    )


def make_agent_trace() -> Tool:
    def run(args: dict[str, Any]) -> dict[str, Any]:
        from rag.store import RagStore
        run_id = int(args["run_id"])
        store = RagStore()
        try:
            trace = store.get_agent_run_trace(run_id)
            return {"run_id": run_id, "count": len(trace), "trace": trace}
        finally:
            store.close()
    return Tool(
        name="agent_trace",
        description="按 run_id 查询某次 Agent 运行的完整工具调用链和 LLM 调用记录。",
        args_schema={
            "type": "object",
            "properties": {"run_id": {"type": "integer"}},
            "required": ["run_id"],
        },
        run=run,
        fallback=lambda: {"run_id": None, "count": 0, "trace": []},
    )

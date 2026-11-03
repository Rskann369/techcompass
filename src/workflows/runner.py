"""TechCompass — Workflow Runner。

能力：
- YAML 配置化节点
- 节点级重试 / 指数退避
- Workflow checkpoint
- SQLite 持久化 workflow_runs / workflow_node_runs
- 从失败节点恢复执行
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

from adapters.schema import Item
from adapters.review_dataset_adapter import ReviewDatasetAdapter
from adapters.news_api_adapter import NewsAPIAdapter
from workflows.config_loader import load_workflow_config
from workflows.nodes import WorkflowState, node_archive
from workflows.scheduler import WorkflowScheduler, NodeSpec
from project_paths import DEFAULT_ENV_PATH
from rag.store import RagStore


@dataclass
class RunConfig:
    query: str = "有没有充电发热或安全隐患的负面反馈？"
    dataset_path: str = "data/samples/sample_reviews_large.jsonl"
    archive_path: str = "outputs/weekly_reports/latest_run.json"
    top_k: int = 5
    use_news: bool = False
    news_query: Optional[str] = None
    news_max: int = 5
    env_path: str = str(DEFAULT_ENV_PATH)
    workflow_config_path: str = "config/workflow.yml"
    analysis_backend: str = "rule"


def _fallback_items() -> list[Item]:
    from adapters.schema import parse_timestamp, SOURCE_REVIEW
    return [
        Item(
            id="fb1", source_type=SOURCE_REVIEW, product_category="headphones",
            product_name="WireX Buds Pro", title="充电发热",
            content="充电后发热，担心安全。", timestamp=parse_timestamp("2026-10-05"), rating=2,
        )
    ]


def _collect_items(config: RunConfig, state: WorkflowState) -> list[Item]:
    items: list[Item] = []
    review = ReviewDatasetAdapter(config.dataset_path).fetch()
    if review.ok and review.items:
        items.extend(review.items)
        state.record(f"source review_dataset: {len(review.items)} items")
    else:
        state.record(f"source review_dataset failed ({review.error_code})")

    if config.use_news:
        nq = config.news_query or config.query
        news = NewsAPIAdapter(query=nq, max_results=config.news_max, env_path=config.env_path).fetch()
        if news.ok and news.items:
            items.extend(news.items)
            state.record(f"source gnews: {len(news.items)} items")
        else:
            state.record(f"source gnews failed ({news.error_code}) -> fallback")

    if not items:
        state.record("all sources empty -> use builtin fallback samples")
        items = _fallback_items()
    return items


def _build_specs(config: RunConfig, items: list[Item]) -> list[NodeSpec]:
    workflow_config = load_workflow_config(config.workflow_config_path)
    specs = []
    for spec in workflow_config.specs:
        kwargs = dict(spec.kwargs)
        if spec.name == "ingest":
            kwargs["items"] = items
        elif spec.name == "index_retrieve":
            kwargs["query"] = config.query
            kwargs["top_k"] = config.top_k
        elif spec.name == "analyze":
            kwargs["analysis_backend"] = config.analysis_backend
            kwargs["query"] = config.query
        elif spec.name == "archive":
            kwargs["path"] = config.archive_path
        specs.append(NodeSpec(
            name=spec.name,
            fn=spec.fn,
            kwargs=kwargs,
            failure_policy=spec.failure_policy,
            max_retries=spec.max_retries,
            backoff_seconds=spec.backoff_seconds,
            timeout_seconds=spec.timeout_seconds,
        ))
    return specs


def run_workflow(
    config: Optional[RunConfig] = None,
    resume_run_id: Optional[int] = None,
) -> dict[str, Any]:
    config = config or RunConfig()
    store = RagStore()

    workflow_config = load_workflow_config(config.workflow_config_path)

    if resume_run_id:
        run_row = store.get_workflow_run(resume_run_id)
        if not run_row:
            raise ValueError(f"workflow run not found: {resume_run_id}")
        checkpoint = json.loads(run_row["checkpoint"] or "{}")
        state = WorkflowState.from_checkpoint(checkpoint)
        items = [Item(**x) for x in checkpoint.get("raw_items", [])] if checkpoint.get("raw_items") else _collect_items(config, state)
        run_id = resume_run_id
    else:
        state = WorkflowState()
        items = _collect_items(config, state)
        run_id = store.start_workflow_run(workflow_config.name, config.query)

    specs = _build_specs(config, items)

    # Resume: skip nodes already ok
    completed_nodes = set()
    if resume_run_id:
        node_rows = store.list_workflow_nodes(resume_run_id)
        completed_nodes = {r["node_name"] for r in node_rows if r["status"] == "ok"}
        specs = [s for s in specs if s.name not in completed_nodes]

    # 每个节点执行后立即持久化，避免进程中断丢失已完成节点
    scheduler = WorkflowScheduler(specs, on_node_finished=lambda record: (
        store.record_workflow_node(
            run_id=run_id,
            node_name=record["node"],
            status=record["status"],
            attempts=record.get("attempts", 0),
            elapsed_ms=record.get("elapsed_ms"),
            error=record.get("error"),
            checkpoint=state.to_checkpoint(),
        ),
        store.checkpoint_workflow_run(
            run_id=run_id,
            checkpoint=state.to_checkpoint(),
        ),
    ))
    state, summary = scheduler.run(state)

    final_status = "success" if not state.errors else "failed"
    store.finish_workflow_run(
        run_id=run_id,
        status=final_status,
        checkpoint=state.to_checkpoint(),
        metrics={
            "summary": summary,
            "ingested": len(items),
            "errors": state.errors,
        },
    )
    store.close()

    return {
        "workflow_run_id": run_id,
        "workflow_name": workflow_config.name,
        "ok": not state.errors,
        "summary": summary,
        "output": state.output,
        "log": state.log,
        "errors": state.errors,
        "ingested": len(items),
        "checkpoint": state.to_checkpoint(),
    }


if __name__ == "__main__":
    res = run_workflow(RunConfig(use_news=False))
    print(json.dumps(res, ensure_ascii=False, indent=2))

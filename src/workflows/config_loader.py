"""TechCompass — Workflow YAML 配置加载器。

从 config/workflow.yml 加载节点顺序、失败策略、重试次数和退避参数。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from workflows.nodes import (
    node_ingest,
    node_normalize,
    node_index_retrieve,
    node_analyze,
    node_reason,
    node_output,
    node_archive,
)
from workflows.scheduler import NodeSpec

NODE_REGISTRY = {
    "node_ingest": node_ingest,
    "node_normalize": node_normalize,
    "node_index_retrieve": node_index_retrieve,
    "node_analyze": node_analyze,
    "node_reason": node_reason,
    "node_output": node_output,
    "node_archive": node_archive,
}


@dataclass
class WorkflowConfig:
    name: str
    specs: list[NodeSpec]


def load_workflow_config(path: str = "config/workflow.yml") -> WorkflowConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"workflow config not found: {config_path}")

    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    name = data.get("name", "workflow")
    raw_nodes = data.get("nodes", [])
    specs: list[NodeSpec] = []

    for raw in raw_nodes:
        node_name = raw.get("name")
        fn_name = raw.get("fn")
        if not node_name or not fn_name:
            raise ValueError("workflow node must contain name and fn")
        fn = NODE_REGISTRY.get(fn_name)
        if fn is None:
            raise ValueError(f"unknown workflow node function: {fn_name}")

        timeout_raw = raw.get("timeout_seconds")
        specs.append(NodeSpec(
            name=node_name,
            fn=fn,
            kwargs={},
            failure_policy=raw.get("failure_policy", "strict"),
            max_retries=int(raw.get("max_retries", 0)),
            backoff_seconds=float(raw.get("backoff_seconds", 0.1)),
            timeout_seconds=float(timeout_raw) if timeout_raw is not None else None,
        ))

    return WorkflowConfig(name=name, specs=specs)

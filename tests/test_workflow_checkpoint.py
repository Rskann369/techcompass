"""TechCompass — Workflow YAML 配置与 checkpoint/resume 测试。"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.store import RagStore  # noqa: E402
from workflows.config_loader import load_workflow_config  # noqa: E402
from workflows.nodes import WorkflowState  # noqa: E402
from workflows.runner import RunConfig, run_workflow  # noqa: E402


def test_load_workflow_config():
    config = load_workflow_config("config/workflow.yml")
    assert config.name == "techcompass_sentiment_workflow"
    assert len(config.specs) == 7
    assert config.specs[0].name == "ingest"
    assert config.specs[2].name == "index_retrieve"
    assert config.specs[2].failure_policy == "continue"
    assert config.specs[2].max_retries == 2


def test_workflow_state_checkpoint_roundtrip():
    state = WorkflowState()
    state.raw_items = [{
        "id": "x", "source_type": "review", "product_category": "headphones",
        "product_name": "Test", "title": "title", "content": "content",
        "timestamp": "2026-01-01T00:00:00+00:00", "rating": 1,
    }]
    state.log.append("hello")
    state.errors.append("none")
    checkpoint = state.to_checkpoint()
    restored = WorkflowState.from_checkpoint(checkpoint)
    assert restored.raw_items[0]["id"] == "x"
    assert restored.log == ["hello"]
    assert restored.errors == ["none"]


@pytest.mark.local
def test_workflow_run_persists_nodes_and_checkpoint():
    cfg = RunConfig(archive_path="/tmp/test_workflow_archive.json")
    result = run_workflow(cfg)
    assert result["ok"] is True
    run_id = result["workflow_run_id"]

    store = RagStore()
    try:
        run = store.get_workflow_run(run_id)
        assert run["workflow_name"] == "techcompass_sentiment_workflow"
        assert run["status"] == "success"
        assert run["checkpoint"]

        nodes = store.list_workflow_nodes(run_id)
        assert len(nodes) >= 7
        assert all(n["status"] == "ok" for n in nodes)
        assert {n["node_name"] for n in nodes} == {
            "ingest", "normalize", "index_retrieve", "analyze", "reason", "output", "archive"
        }
    finally:
        store.close()


@pytest.mark.local
def test_resume_skips_completed_nodes():
    cfg = RunConfig(archive_path="/tmp/test_workflow_resume.json")
    first = run_workflow(cfg)
    run_id = first["workflow_run_id"]
    resumed = run_workflow(cfg, resume_run_id=run_id)
    assert resumed["workflow_run_id"] == run_id
    # 已完成节点会被跳过，summary 应为空
    assert resumed["summary"] == []
    assert resumed["ok"] is True


if __name__ == "__main__":
    tests = [
        test_load_workflow_config,
        test_workflow_state_checkpoint_roundtrip,
        test_workflow_run_persists_nodes_and_checkpoint,
        test_resume_skips_completed_nodes,
    ]
    passed = 0
    for test in tests:
        try:
            test()
            print("PASS", test.__name__)
            passed += 1
        except Exception as e:
            print("FAIL", test.__name__, "->", e)
    print(f"{passed}/{len(tests)} passed")

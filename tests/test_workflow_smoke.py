"""Week 2 冒烟测试：验证 Workflow 能跑通、降级、幂等。

运行（Python 3.13 环境）：
    PYTHONPATH=src ./.venv311/bin/python -m pytest tests/test_workflow_smoke.py -q
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

pytestmark = pytest.mark.local

from workflows.runner import run_workflow, RunConfig  # noqa: E402


def test_end_to_end():
    res = run_workflow(RunConfig(archive_path="/tmp/tc_smoke.json"))
    assert res["ok"] is True
    assert "summary" in res["output"]
    assert "priority" in res["output"]
    assert res["ingested"] >= 1


def test_fallback_when_source_missing():
    cfg = RunConfig(dataset_path="data/samples/missing.jsonl", archive_path="/tmp/tc_fb.json")
    res = run_workflow(cfg)
    assert res["ok"] is True
    assert any("fallback" in line for line in res["log"])


def test_idempotent():
    cfg = RunConfig(archive_path="/tmp/tc_idem.json")
    r1 = run_workflow(cfg)
    r2 = run_workflow(cfg)
    assert r1["output"]["summary"]["total_reviewed"] == r2["output"]["summary"]["total_reviewed"]


if __name__ == "__main__":
    print("smoke tests: run with pytest instead")

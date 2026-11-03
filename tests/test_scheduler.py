"""WorkflowScheduler 调度能力测试。

覆盖：
- 正常执行
- 节点重试后成功
- 严格失败终止后续节点
- continue 失败继续执行
- 指数退避行为
- 兼容 allow_fail=True
"""

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from workflows.scheduler import NodeSpec, WorkflowScheduler  # noqa: E402


@dataclass
class State:
    calls: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def ok_node(state, name: str = "ok"):
    state.calls.append(name)
    return state


def flaky_node(state, fail_first: int, name: str = "flaky"):
    state.calls.append(name)
    if len([c for c in state.calls if c == name]) <= fail_first:
        raise RuntimeError("temporary failure")
    return state


def always_fail(state, name: str = "fail"):
    state.calls.append(name)
    raise RuntimeError("permanent failure")


def test_success():
    state = State()
    sched = WorkflowScheduler([
        NodeSpec("a", ok_node, {"name": "a"}),
        NodeSpec("b", ok_node, {"name": "b"}),
    ])
    state, summary = sched.run(state)
    assert state.calls == ["a", "b"]
    assert [s["status"] for s in summary] == ["ok", "ok"]


def test_retry_then_success():
    state = State()
    sched = WorkflowScheduler([
        NodeSpec("flaky", flaky_node, {"fail_first": 1}, max_retries=2, backoff_seconds=0),
        NodeSpec("after", ok_node, {"name": "after"}),
    ])
    state, summary = sched.run(state)
    assert state.calls.count("flaky") == 2
    assert summary[0]["status"] == "ok"
    assert summary[0]["attempts"] == 2
    assert summary[1]["status"] == "ok"


def test_strict_failure_skips_remaining():
    state = State()
    sched = WorkflowScheduler([
        NodeSpec("fail", always_fail, {}, max_retries=1, backoff_seconds=0),
        NodeSpec("next", ok_node, {"name": "next"}),
    ])
    state, summary = sched.run(state)
    assert "next" not in state.calls
    assert summary[0]["status"] == "failed"
    assert summary[1]["status"] == "skipped"
    assert "fail: permanent failure" in state.errors


def test_continue_failure_executes_remaining():
    state = State()
    sched = WorkflowScheduler([
        NodeSpec("fail", always_fail, {}, failure_policy="continue", max_retries=0),
        NodeSpec("next", ok_node, {"name": "next"}),
    ])
    state, summary = sched.run(state)
    assert "next" in state.calls
    assert summary[0]["status"] == "failed"
    assert summary[1]["status"] == "ok"
    assert "fail: permanent failure (continued)" in state.errors


def test_allow_fail_compatibility():
    state = State()
    sched = WorkflowScheduler([
        NodeSpec("fail", always_fail, {}, allow_fail=True),
        NodeSpec("next", ok_node, {"name": "next"}),
    ])
    state, summary = sched.run(state)
    assert "next" in state.calls
    assert summary[0]["failure_policy"] == "continue"


def test_exponential_backoff_timing():
    state = State()
    sched = WorkflowScheduler([
        NodeSpec(
            "flaky",
            flaky_node,
            {"fail_first": 2},
            max_retries=2,
            backoff_seconds=0.01,
        ),
    ])
    start = time.perf_counter()
    state, summary = sched.run(state)
    elapsed = time.perf_counter() - start
    # backoff: 0.01 + 0.02 = 0.03
    assert elapsed >= 0.03
    assert summary[0]["attempts"] == 3


if __name__ == "__main__":
    tests = [
        test_success,
        test_retry_then_success,
        test_strict_failure_skips_remaining,
        test_continue_failure_executes_remaining,
        test_allow_fail_compatibility,
        test_exponential_backoff_timing,
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

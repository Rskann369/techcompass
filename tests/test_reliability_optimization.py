"""可靠性优化测试：调度重试、超时、节点级 checkpoint、路径安全。"""
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tools.registry import Tool, ToolRegistry  # noqa: E402
from workflows.scheduler import NodeSpec, WorkflowScheduler  # noqa: E402


@dataclass
class S:
    errors: list[str] = field(default_factory=list)


def test_node_exception_triggers_retry():
    calls = {"n": 0}

    def flaky(state):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("fail")
        return state

    sched = WorkflowScheduler([NodeSpec("n", flaky, {}, max_retries=1, backoff_seconds=0)])
    _, summary = sched.run(S())
    assert summary[0]["status"] == "ok"
    assert summary[0]["attempts"] == 2


def test_timeout_is_enforced():
    def slow(state):
        time.sleep(0.3)
        return state

    sched = WorkflowScheduler([
        NodeSpec("slow", slow, {}, max_retries=0, timeout_seconds=0.05)
    ])
    _, summary = sched.run(S())
    assert summary[0]["status"] == "failed"
    assert "timed out" in summary[0]["error"]


def test_callback_persists_each_node_record():
    records = []

    def ok(state):
        return state

    def cb(record):
        records.append(record)

    sched = WorkflowScheduler(
        [NodeSpec("a", ok, {}), NodeSpec("b", ok, {})],
        on_node_finished=cb,
    )
    sched.run(S())
    assert len(records) == 2
    assert records[0]["node"] == "a"
    assert records[1]["node"] == "b"


def test_tool_args_validation():
    tool = Tool(
        name="t", description="t",
        args_schema={
            "type": "object",
            "properties": {"q": {"type": "string"}, "level": {"type": "string", "enum": ["low", "high"]}},
            "required": ["q"],
        },
        run=lambda args: {"ok": True},
    )
    assert tool.execute({"q": "x"}) == {"ok": True}
    try:
        tool.execute({})
    except ValueError as e:
        assert "missing required argument" in str(e)
    else:
        raise AssertionError("expected validation error")

    try:
        tool.execute({"q": "x", "level": "bad"})
    except ValueError:
        pass
    else:
        raise AssertionError("expected enum error")


def test_export_csv_is_path_restricted(tmp_path=None):
    from tools.output_tools import make_export_csv
    tool = make_export_csv()
    result = tool.execute({"rows": [], "path": "../outside.csv"})
    assert result["_error"] == "export path must stay inside outputs/"
    assert result["_fallback"] is True

    result = tool.execute({"rows": [{"a": 1}], "path": "safe.csv"})
    assert Path(result["path"]).resolve().is_relative_to((Path.cwd() / "outputs").resolve())


if __name__ == "__main__":
    tests = [
        test_node_exception_triggers_retry,
        test_timeout_is_enforced,
        test_callback_persists_each_node_record,
        test_tool_args_validation,
        test_export_csv_is_path_restricted,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print("PASS", t.__name__)
            passed += 1
        except Exception as e:
            print("FAIL", t.__name__, "->", e)
    print(f"{passed}/{len(tests)} passed")

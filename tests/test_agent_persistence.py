"""TechCompass — 真实 LLM Agent trace SQLite 持久化测试（集成）。"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

pytestmark = pytest.mark.local

from agents.langgraph_react_agent import LangGraphReactAgent  # noqa: E402
from rag.store import RagStore  # noqa: E402


def test_agent_run_persists_run_and_trace():
    store = RagStore()
    before_runs = store.stats()["agent_runs"]
    store.close()

    agent = LangGraphReactAgent()
    result = agent.run("wireless earbuds safety")

    assert result["ok"] is True
    assert result["run_id"] is not None

    store = RagStore()
    try:
        stats = store.stats()
        assert stats["agent_runs"] == before_runs + 1
        run = store.list_agent_runs(limit=1)[0]
        assert run["id"] == result["run_id"]
        assert run["pipeline"] == "langgraph_react_agent"
        assert run["question"] == "wireless earbuds safety"
        trace = store.get_agent_run_trace(result["run_id"])
        assert len(trace) >= 1
    finally:
        store.close()


if __name__ == "__main__":
    tests = [test_agent_run_persists_run_and_trace]
    passed = 0
    for test in tests:
        try:
            test()
            print("PASS", test.__name__)
            passed += 1
        except Exception as e:  # noqa: BLE001
            print("FAIL", test.__name__, "->", e)
    print(f"{passed}/{len(tests)} passed")
    raise SystemExit(0 if passed == len(tests) else 1)

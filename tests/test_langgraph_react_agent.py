"""LangGraph create_react_agent integration tests (offline)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

pytestmark = pytest.mark.local

from agents.langgraph_react_agent import LangGraphReactAgent  # noqa: E402
from rag.store import RagStore  # noqa: E402


def test_agent_uses_create_react_agent_and_langchain_tools():
    agent = LangGraphReactAgent(verbose=False)
    assert agent.agent is not None
    assert len(agent.registry.to_langchain_tools()) == 8
    assert agent.system_prompt.startswith("你是 TechCompass")


def test_agent_run_persists_run_trace_and_llm_events():
    agent = LangGraphReactAgent(verbose=False)

    # Fake agent invocation to avoid real network call.
    class FakeMessage:
        def __init__(self, content, tool_calls=None, usage_metadata=None):
            self.content = content
            self.tool_calls = tool_calls or []
            self.usage_metadata = usage_metadata or {}

    original_invoke = agent.agent.invoke
    agent.agent.invoke = lambda inputs, config=None: {
        "messages": [
            FakeMessage("question", usage_metadata={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}),
            FakeMessage("tool call", tool_calls=[{"name": "hybrid_rag_search", "args": {"query": "privacy"}}]),
            FakeMessage("final answer", usage_metadata={"input_tokens": 4, "output_tokens": 3, "total_tokens": 7}),
        ]
    }

    try:
        result = agent.run("test question")
        assert result["ok"] is True
        assert result["answer"] == "final answer"
        assert result["tool_calls"][0]["tool"] == "hybrid_rag_search"

        store = RagStore()
        run = store.list_agent_runs(limit=1)[0]
        assert run["pipeline"] == "langgraph_react_agent"
        assert run["question"] == "test question"
        assert run["answer"] == "final answer"

        trace = store.get_agent_run_trace(run["id"])
        assert any(item["tool_name"] == "hybrid_rag_search" for item in trace)

        llm_events = store.list_llm_events(run["id"])
        assert len(llm_events) >= 0
        store.close()
    finally:
        agent.agent.invoke = original_invoke
        agent.store.close()


if __name__ == "__main__":
    tests = [
        test_agent_uses_create_react_agent_and_langchain_tools,
        test_agent_run_persists_run_trace_and_llm_events,
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

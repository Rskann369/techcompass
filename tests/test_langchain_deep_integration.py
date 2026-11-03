"""LangChain 深度集成测试：callback 持久化、StructuredTool 输出、分析工具 callback。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agents.langchain_callbacks import TechCompassCallbackHandler  # noqa: E402
from rag.store import RagStore  # noqa: E402
from tools.domain_tools import build_registry  # noqa: E402
from tools.llm_analysis_tools import make_analyze_consumer_sentiment  # noqa: E402


def test_llm_events_persistence():
    handler = TechCompassCallbackHandler()
    handler.on_llm_start({}, ["hello"], run_id="run-1")
    handler.on_llm_end(
        type("R", (), {"usage_metadata": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}})(),
        run_id="run-1",
    )

    store = RagStore()
    agent_run_id = store.start_agent_run("test_callbacks", "callback persistence test")
    store.insert_llm_events(agent_run_id, handler.events)
    events = store.list_llm_events(agent_run_id)
    assert len(events) == 2
    assert events[0]["event_type"] == "llm_start"
    assert events[1]["event_type"] == "llm_end"
    assert events[1]["prompt_tokens"] == 3
    assert events[1]["completion_tokens"] == 2
    assert events[1]["total_tokens"] == 5
    store.close()


def test_registry_to_langchain_tools():
    registry = build_registry("data/samples/nonexistent.jsonl")
    tools = registry.to_langchain_tools()
    assert len(tools) == 8
    names = {tool.name for tool in tools}
    assert "hybrid_rag_search" in names
    assert "analyze_consumer_sentiment" in names
    assert "generate_weekly_report" in names


def test_analyze_tool_accepts_callbacks():
    handler = TechCompassCallbackHandler()
    tool = make_analyze_consumer_sentiment(callbacks=[handler])
    assert tool.name == "analyze_consumer_sentiment"


if __name__ == "__main__":
    tests = [
        test_llm_events_persistence,
        test_registry_to_langchain_tools,
        test_analyze_tool_accepts_callbacks,
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

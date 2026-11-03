"""LangChain integration tests: PromptTemplate, Pydantic schema, callbacks."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agents.langchain_callbacks import TechCompassCallbackHandler  # noqa: E402
from tools.llm_analysis_tools import (  # noqa: E402
    ANALYSIS_PROMPT,
    ConsumerSentimentAnalysis,
)


def test_analysis_prompt_template():
    prompt_value = ANALYSIS_PROMPT.invoke({
        "question": "smart home camera privacy",
        "evidence": "[1] Apple HomeKit privacy concern",
    })
    text = prompt_value.to_string()
    assert "smart home camera privacy" in text
    assert "Apple HomeKit privacy concern" in text


def test_structured_output_schema():
    model = ConsumerSentimentAnalysis(
        issue_type="privacy",
        product_categories=["smart_home"],
        sentiment="negative",
        risk_level="high",
        impact_level="medium",
        evidence_sufficient=True,
        evidence_refs=["[1]"],
        confidence=0.8,
    )
    data = model.model_dump()
    assert data["issue_type"] == "privacy"
    assert data["risk_level"] == "high"
    assert data["evidence_refs"] == ["[1]"]


def test_structured_output_schema_rejects_invalid_enum():
    try:
        ConsumerSentimentAnalysis(
            issue_type="privacy",
            sentiment="invalid",
            risk_level="high",
            impact_level="medium",
            evidence_sufficient=True,
        )
    except Exception:
        return
    raise AssertionError("expected validation error")


def test_callback_handler_llm_events():
    handler = TechCompassCallbackHandler()
    run_id = "test-run"
    handler.on_llm_start({}, ["hello"], run_id=run_id)
    handler.on_llm_end("response", run_id=run_id)
    assert len(handler.events) == 2
    assert handler.events[0]["type"] == "llm_start"
    assert handler.events[1]["type"] == "llm_end"


def test_callback_handler_tool_events():
    handler = TechCompassCallbackHandler()
    run_id = "tool-run"
    handler.on_tool_start({"name": "hybrid_rag_search"}, {"query": "privacy"}, run_id=run_id)
    handler.on_tool_end({"count": 1}, run_id=run_id)
    assert len(handler.events) == 2
    assert handler.events[0]["tool"] == "hybrid_rag_search"
    assert handler.events[1]["output"]["count"] == 1


if __name__ == "__main__":
    tests = [
        test_analysis_prompt_template,
        test_structured_output_schema,
        test_structured_output_schema_rejects_invalid_enum,
        test_callback_handler_llm_events,
        test_callback_handler_tool_events,
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

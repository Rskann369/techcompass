"""TechCompass — 统一风险词表与 Workflow Hybrid RAG 测试。"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analysis.risk_lexicon import detect_risk, detect_sentiment  # noqa: E402
from tools.domain_tools import build_registry  # noqa: E402
from tools.analysis_tools import make_score_risk, make_analyze_sentiment  # noqa: E402
from workflows.nodes import WorkflowState, node_analyze, node_index_retrieve  # noqa: E402


def test_chinese_safety_risk():
    result = detect_risk("耳机充电时出现安全隐患，可能起火")
    assert result["risk_level"] == "high"
    assert "safety" in result["risk_types"]


def test_english_privacy_risk():
    result = detect_risk("Smart home camera has unauthorized access and privacy leak")
    assert result["risk_level"] == "high"
    assert "privacy" in result["risk_types"]
    assert result["requires_human_review"] is True


def test_english_battery_risk():
    result = detect_risk("The wireless earbuds battery is swollen and fails to charge")
    assert result["risk_level"] == "medium"
    assert "battery" in result["risk_types"]


def test_english_logistics_low_risk():
    result = detect_risk("Shipping delay and packaging damaged")
    assert result["risk_level"] == "low"
    assert "logistics" in result["risk_types"]


def test_no_risk():
    result = detect_risk("This is a great product review")
    assert result["risk_level"] == "none"
    assert result["requires_human_review"] is False


def test_unified_sentiment():
    assert detect_sentiment("very bad and disappointed")["sentiment"] == "negative"
    assert detect_sentiment("great product, recommend")["sentiment"] == "positive"
    assert detect_sentiment("firmware update")["sentiment"] == "neutral"


def test_agent_rule_tools_use_unified_lexicon():
    registry = build_registry("data/samples/nonexistent.jsonl")
    risk = make_score_risk().run({"text": "privacy leak and data breach"})
    assert risk["risk_level"] == "high"
    assert "privacy" in risk["risk_types"]

    sentiment = make_analyze_sentiment().run({"text": "very disappointed"})
    assert sentiment["sentiment"] == "negative"


@pytest.mark.local
def test_workflow_index_retrieve_uses_hybrid_rag():
    state = WorkflowState()
    result = node_index_retrieve(state, query="smart home camera privacy", top_k=5)
    assert result is state
    assert len(state.retrieved) > 0
    assert any("hybrid RAG returned" in line for line in state.log)
    # Hybrid RAG 返回结果包含 rrf_score
    assert any("rrf_score" in item for item in state.retrieved)


def test_workflow_analyze_uses_unified_lexicon():
    state = WorkflowState()
    state.retrieved = [
        {"title": "Privacy leak", "content": "Smart home camera unauthorized access and privacy leak"},
        {"title": "Positive review", "content": "great product and recommend"},
    ]
    result = node_analyze(state)
    assert result.analysis["risk_count"] == 1
    assert result.analysis["negative_count"] == 1
    assert result.analysis["mode"] == "unified-rule-based"


if __name__ == "__main__":
    tests = [
        test_chinese_safety_risk,
        test_english_privacy_risk,
        test_english_battery_risk,
        test_english_logistics_low_risk,
        test_no_risk,
        test_unified_sentiment,
        test_agent_rule_tools_use_unified_lexicon,
        test_workflow_index_retrieve_uses_hybrid_rag,
        test_workflow_analyze_uses_unified_lexicon,
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

"""TechCompass — 模型驱动分析工具测试。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tools.domain_tools import build_registry  # noqa: E402
from tools.llm_analysis_tools import (  # noqa: E402
    ANALYSIS_TOOL_DEF,
    JUDGE_TOOL_DEF,
    make_analyze_consumer_sentiment,
    make_judge_evidence,
)


def test_tools_registered():
    registry = build_registry("data/samples/nonexistent.jsonl")
    assert "analyze_consumer_sentiment" in registry.names()
    assert len(registry.names()) == 8


def test_tool_schemas_are_open_domain():
    # analysis schema 不包含任何硬编码品类或风险词
    props = ANALYSIS_TOOL_DEF["function"]["parameters"]["properties"]
    assert "product_categories" in props
    assert props["product_categories"]["type"] == "array"
    assert "issue_type" in props
    assert props["issue_type"]["type"] == "string"
    assert "risk_type" in props
    assert props["risk_type"]["type"] == "string"
    assert "evidence_sufficient" in props


def test_judge_schema_dimensions():
    props = JUDGE_TOOL_DEF["function"]["parameters"]["properties"]
    for key in ("relevance", "sufficiency", "credibility", "coverage", "reason"):
        assert key in props


def test_judge_evidence_rejects_missing_question():
    tool = make_judge_evidence()
    try:
        tool.execute({"evidence": []})
    except ValueError as e:
        assert "missing required argument" in str(e)
    else:
        raise AssertionError("expected validation error")


def test_analyze_rejects_missing_question():
    tool = make_analyze_consumer_sentiment()
    try:
        tool.execute({"evidence": []})
    except ValueError as e:
        assert "missing required argument" in str(e)
    else:
        raise AssertionError("expected validation error")


if __name__ == "__main__":
    tests = [
        test_tools_registered,
        test_tool_schemas_are_open_domain,
        test_judge_schema_dimensions,
        test_judge_evidence_rejects_missing_question,
        test_analyze_rejects_missing_question,
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

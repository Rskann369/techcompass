"""TechCompass — Workflow 分析层 backend 切换测试。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from workflows.nodes import WorkflowState, node_analyze  # noqa: E402
from workflows.runner import RunConfig  # noqa: E402


def test_workflow_config_backend_default():
    cfg = RunConfig()
    assert cfg.analysis_backend == "rule"


def test_workflow_rule_analyze():
    state = WorkflowState()
    state.retrieved = [{"title": "Shipping delay", "content": "shipping delay"}]
    result = node_analyze(state, analysis_backend="rule")
    assert result.analysis["mode"] == "unified-rule-based"


def test_workflow_llm_analyze():
    state = WorkflowState()
    state.retrieved = [{"title": "Privacy risk", "content": "smart home camera privacy risk"}]

    import tools.llm_analysis_tools as mod
    original = mod.make_analyze_consumer_sentiment

    def fake_tool():
        class T:
            def run(self, args):
                return {
                    "sentiment": "negative",
                    "risk_level": "high",
                    "impact_level": "high",
                }
        return T()

    mod.make_analyze_consumer_sentiment = fake_tool
    try:
        result = node_analyze(state, analysis_backend="llm", query="smart home camera privacy")
        assert result.analysis["mode"] == "llm"
        assert result.analysis["negative_count"] == 1
        assert result.analysis["risk_count"] == 1
    finally:
        mod.make_analyze_consumer_sentiment = original


if __name__ == "__main__":
    tests = [
        test_workflow_config_backend_default,
        test_workflow_rule_analyze,
        test_workflow_llm_analyze,
    ]
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

"""TechCompass — 工具链测试。

验证：
1. 工具数量达到目标（12+）
2. 每个工具有 name/description/args_schema/run
3. 分析工具输出正确
4. 输出工具能生成文件
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tools.domain_tools import build_registry  # noqa: E402
from tools.analysis_tools import make_extract_entities, make_analyze_sentiment, make_score_risk, make_reason_impact, make_suggest_action  # noqa: E402


def make_registry():
    return build_registry("data/samples/nonexistent.jsonl")


def test_tool_count():
    reg = make_registry()
    names = set(reg.names())
    assert {
        "hybrid_rag_search", "rewrite_query", "analyze_consumer_sentiment",
        "search_live_sources", "deduplicate_docs", "generate_weekly_report",
        "create_alert", "archive_run",
    }.issubset(names)


def test_tool_schema():
    reg = make_registry()
    for name in reg.names():
        tool = reg.get(name)
        assert tool.name == name
        assert tool.description
        assert isinstance(tool.args_schema, dict)
        assert callable(tool.run)


def test_rewrite_query():
    reg = make_registry()
    res = reg.get("rewrite_query").run({"query": "smartwatch battery"})
    assert "smartwatch" in res["rewrites"]
    assert "battery" in res["rewrites"]
    assert "smartwatch battery" in res["rewrites"]
    # 开放域重写不应自动引入外部主题
    assert "wearable" not in res["rewrites"]


def test_extract_entities():
    reg = make_registry()
    res = make_extract_entities().run({"text": "AirPods battery safety issue"})
    assert "battery" in res["entities"]["product_category"]
    assert "airpods" in res["entities"]["product_model"]
    assert "safety" in res["entities"]["risk_words"]


def test_sentiment():
    reg = make_registry()
    assert make_analyze_sentiment().run({"text": "很好，推荐"})["sentiment"] == "positive"
    assert make_analyze_sentiment().run({"text": "发热，有隐患"})["sentiment"] == "negative"


def test_risk():
    reg = make_registry()
    res = make_score_risk().run({"text": "充电时发热并存在安全隐患"})
    assert res["risk_level"] == "high"
    assert res["requires_human_review"] is True


def test_impact_and_action():
    reg = make_registry()
    impact = make_reason_impact().run({"risk_level": "high", "evidence_count": 3, "spread": "broad"})
    assert impact["impact_level"] == "critical"
    action = make_suggest_action().run({"impact_level": "critical"})
    assert "暂停" in action["actions"]["customer_support"]


def test_weekly_report():
    reg = make_registry()
    res = reg.get("generate_weekly_report").run({
        "topic": "wireless earbuds",
        "findings": ["充电发热反馈增加"],
        "impact_level": "high",
        "actions": {"customer_support": "联系用户", "product": "排查批次"},
    })
    assert Path(res["report_path"]).exists()


def test_alert():
    reg = make_registry()
    res = reg.get("create_alert").run({"title": "充电安全", "level": "high", "evidence": ["doc1"]})
    assert Path(res["alert_path"]).exists()


def test_dedup():
    docs = [
        {"title": "a", "content": "x"},
        {"title": "a", "content": "x"},
        {"title": "b", "content": "y"},
    ]
    reg = make_registry()
    res = reg.get("deduplicate_docs").run({"documents": docs})
    assert res["unique_count"] == 2
    assert res["duplicate_count"] == 1


if __name__ == "__main__":
    tests = [
        test_tool_count, test_tool_schema, test_rewrite_query, test_extract_entities,
        test_sentiment, test_risk, test_impact_and_action, test_weekly_report,
        test_alert, test_dedup,
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

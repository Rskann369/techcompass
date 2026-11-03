"""结论解读层（trace → 自然语言结论）测试。

验证：
- 结论依据逐条给出具体链接。
- 置信度等级正确计算。
- 证据不足时拒答。
- LLM 后端失败时自动回退确定性模板。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agents.conclusion import ConclusionBuilder  # noqa: E402


def _sample_result():
    return {
        "question": "wireless earbuds charging overheating",
        "refused": False,
        "sentiment": {"sentiment": "negative"},
        "risk": {
            "risk_level": "medium",
            "risk_types": ["battery"],
            "requires_human_review": True,
            "hard_safety_signal": False,
        },
        "impact": {"impact_level": "medium"},
        "actions": {
            "actions": {
                "customer_support": "48小时内回复",
                "product": "排查充电发热",
                "communication": "发布安全说明",
            }
        },
        "analysis": {
            "issue_type": "safety",
            "root_cause": "充电过程发热",
            "affected_scope": "部分批次",
            "confidence": 0.8,
            "evidence_sufficient": True,
        },
        "relevant_evidence": [
            {
                "title": "Earbuds overheat while charging",
                "url": "https://example.com/1",
                "source": "gnews",
                "source_type": "news",
                "_semantic_similarity": 0.82,
                "content": "hot during charge",
            },
            {
                "title": "Battery swelling thread",
                "url": "https://reddit.com/2",
                "source": "reddit_rss",
                "source_type": "discussion",
                "_semantic_similarity": 0.71,
                "content": "battery swelling",
            },
        ],
    }


def test_evidence_basis_has_specific_links():
    conclusion = ConclusionBuilder().build(_sample_result())
    basis = conclusion["evidence_basis"]
    assert len(basis) == 2
    assert basis[0]["url"] == "https://example.com/1"
    assert basis[1]["url"] == "https://reddit.com/2"
    assert basis[0]["source_label"] == "GNews 新闻"
    assert basis[1]["source_label"] == "Reddit 讨论"


def test_confidence_high_when_evidence_sufficient():
    conclusion = ConclusionBuilder().build(_sample_result())
    assert conclusion["confidence"]["level"] == "high"
    assert conclusion["confidence"]["evidence_count"] == 2


def test_refused_case_low_confidence_and_no_evidence():
    result = {
        "question": "write a poem about autumn",
        "refused": True,
        "sentiment": {"sentiment": "neutral"},
        "risk": {"risk_level": "none"},
        "impact": {"impact_level": "none"},
        "actions": {"actions": {}},
        "relevant_evidence": [],
    }
    conclusion = ConclusionBuilder().build(result)
    assert conclusion["refused"] is True
    assert conclusion["confidence"]["level"] == "low"
    assert conclusion["evidence_basis"] == []
    assert "证据不足" in conclusion["summary"]
    assert "无法给出舆情结论" in conclusion["headline"]


def test_llm_backend_falls_back_to_template_on_failure():
    builder = ConclusionBuilder(use_llm=True, request_timeout=0.1)
    conclusion = builder.build(_sample_result())
    # 无可用网关 / 超时时应回退到确定性模板，仍产出 headline/summary
    assert conclusion["headline"]
    assert conclusion["summary"]
    assert conclusion["evidence_basis"][0]["url"] == "https://example.com/1"


if __name__ == "__main__":
    tests = [
        test_evidence_basis_has_specific_links,
        test_confidence_high_when_evidence_sufficient,
        test_refused_case_low_confidence_and_no_evidence,
        test_llm_backend_falls_back_to_template_on_failure,
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

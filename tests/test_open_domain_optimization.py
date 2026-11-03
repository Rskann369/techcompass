"""开放域 RAG / 分析优化测试（去掉离线 pipeline 依赖）。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tools.analysis_tools import make_rewrite_query  # noqa: E402
from tools.llm_analysis_tools import _format_evidence  # noqa: E402


def test_rewrite_query_does_not_add_unrelated_topics():
    result = make_rewrite_query().run({"query": "baby monitor audio leak"})
    terms = set(result["rewrites"])
    assert "smart home" not in terms
    assert "camera" not in terms
    assert "privacy" not in terms
    assert "baby monitor audio leak" in terms


def test_rewrite_query_preserves_original():
    result = make_rewrite_query().run({"query": "drone firmware vulnerability"})
    assert result["original"] == "drone firmware vulnerability"
    assert result["original"] in result["rewrites"]


def test_structured_evidence_has_numbered_refs():
    evidence = [
        {"title": "A", "source": "gnews", "url": "https://a", "content": "x"},
        {"title": "B", "source": "reddit", "url": "https://b", "content": "y"},
    ]
    text, refs = _format_evidence(evidence)
    assert "[1] title: A" in text
    assert "[2] title: B" in text
    assert refs == ["[1]", "[2]"]


if __name__ == "__main__":
    tests = [
        test_rewrite_query_does_not_add_unrelated_topics,
        test_rewrite_query_preserves_original,
        test_structured_evidence_has_numbered_refs,
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

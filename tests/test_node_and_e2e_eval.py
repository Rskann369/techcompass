"""TechCompass — 节点级评测器测试（仅离线可跑的部分）。

analysis / output / e2e 节点现在走真实 LLM，需要网关，不在快速单测里跑；
这里只保留 retrieval（离线 RAG）与 evalset 合法性。
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eval.node_eval import evaluate_retrieval_node, load_cases  # noqa: E402

CASES = load_cases("data/evalsets/node_eval_cases.jsonl")


def test_evalset_valid():
    assert len(CASES) >= 5
    for case in CASES:
        assert "question" in case
        assert "expected" in case
        assert "relevant_titles" in case["expected"]


@pytest.mark.local
def test_retrieval_node_metrics():
    result = evaluate_retrieval_node(CASES)
    for key in ("hit_rate", "recall_at_k", "precision_at_k", "mrr_at_k", "evidence_relevance_rate"):
        assert key in result
    assert result["hit_rate"] >= 0.7
    assert result["case_count"] == len(CASES)


if __name__ == "__main__":
    tests = [test_evalset_valid, test_retrieval_node_metrics]
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

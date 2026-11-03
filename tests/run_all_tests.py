"""TechCompass — 一键运行全部测试。"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = [
    "tests/test_reliability_optimization.py",
    "tests/test_scheduler.py",
    "tests/test_tools.py",
    "tests/test_agent_persistence.py",
    "tests/test_hybrid_retrieval.py",
    "tests/test_analysis_backend.py",
    "tests/test_open_domain_optimization.py",
    "tests/test_llm_analysis_tools.py",
    "tests/test_unified_risk_and_workflow.py",
    "tests/test_workflow_checkpoint.py",
    "tests/test_semantic_embedding.py",
    "tests/test_node_and_e2e_eval.py",
    "tests/test_prompt_gate_timeout.py",
    "tests/test_evidence_and_risk_gate.py",
    "tests/test_langchain_integration.py",
    "tests/test_langchain_deep_integration.py",
    "tests/test_langgraph_react_agent.py",
    "tests/test_conclusion.py",
    "tests/test_multi_source.py",
    "tests/test_live_search.py",
]


def main():
    passed_files = 0
    failed_files = []
    for test in TESTS:
        print(f"\n===== {test} =====", flush=True)
        result = subprocess.run(
            [sys.executable, str(ROOT / test)],
            cwd=ROOT,
            env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin:/usr/sbin:/sbin"},
        )
        if result.returncode == 0:
            passed_files += 1
        else:
            failed_files.append(test)

    print("\n===== SUMMARY =====")
    print(f"Passed files: {passed_files}/{len(TESTS)}")
    if failed_files:
        print("Failed files:")
        for f in failed_files:
            print(f" - {f}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

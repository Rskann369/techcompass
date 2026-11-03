"""Prompt 统一 + YAML timeout 测试。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tools.llm_analysis_tools import ANALYSIS_SYSTEM_PROMPT  # noqa: E402
from workflows.config_loader import load_workflow_config  # noqa: E402


def test_prompts_are_composed_into_analysis():
    from prompts.classify_prompt import CLASSIFY_SYSTEM
    from prompts.entity_prompt import ENTITY_SYSTEM
    from prompts.reason_prompt import REASON_SYSTEM
    from prompts.impact_prompt import IMPACT_SYSTEM
    from prompts.action_prompt import ACTION_SYSTEM
    assert CLASSIFY_SYSTEM in ANALYSIS_SYSTEM_PROMPT
    assert ENTITY_SYSTEM in ANALYSIS_SYSTEM_PROMPT
    assert REASON_SYSTEM in ANALYSIS_SYSTEM_PROMPT
    assert IMPACT_SYSTEM in ANALYSIS_SYSTEM_PROMPT
    assert ACTION_SYSTEM in ANALYSIS_SYSTEM_PROMPT


def test_yaml_timeout_loaded():
    config = load_workflow_config("config/workflow.yml")
    assert all(spec.timeout_seconds is not None for spec in config.specs)
    retrieve = next(spec for spec in config.specs if spec.name == "index_retrieve")
    assert retrieve.timeout_seconds == 30


if __name__ == "__main__":
    tests = [test_prompts_are_composed_into_analysis, test_yaml_timeout_loaded]
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

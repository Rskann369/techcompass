"""硬安全信号 / 风险优先情绪 测试（纯风险词表单测）。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analysis.risk_lexicon import detect_risk, detect_sentiment, has_hard_safety_signal  # noqa: E402


def test_hard_safety_signal_detection():
    assert has_hard_safety_signal("battery fire risk") is True
    assert has_hard_safety_signal("privacy leak reported") is True
    assert has_hard_safety_signal("great product review") is False


def test_medium_risk_forces_negative_sentiment():
    result = detect_sentiment("Volkswagen recalls electric vehicles over fire risk")
    assert result["sentiment"] == "negative"
    assert result["risk_level"] in ("medium", "high")


def test_positive_review_without_risk():
    result = detect_sentiment("This is a great product and I recommend it")
    assert result["sentiment"] == "positive"


def test_neutral_informational_text():
    result = detect_sentiment("Garmin smartwatch has 320-hour battery life")
    assert result["sentiment"] == "neutral"


def test_hard_safety_requires_review_even_if_level_low():
    result = detect_risk("product recall announced")
    assert result["hard_safety_signal"] is True
    assert result["risk_level"] in ("medium", "high")
    assert result["requires_human_review"] is True


if __name__ == "__main__":
    tests = [
        test_hard_safety_signal_detection,
        test_medium_risk_forces_negative_sentiment,
        test_positive_review_without_risk,
        test_neutral_informational_text,
        test_hard_safety_requires_review_even_if_level_low,
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

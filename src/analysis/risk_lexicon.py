"""TechCompass — 统一中英文风险词表。

目标：
- Workflow 与 Agent 使用同一套风险判断。
- 同时覆盖中文和英文消费品舆情风险表达。
- 支持风险等级、风险类型和是否需要人工审核。
"""

from __future__ import annotations

from typing import Any

# 风险等级词表：按 high → medium → low 判断，命中即停止。
RISK_WORDS: dict[str, tuple[str, ...]] = {
    "high": (
        # 安全/事故
        "爆炸", "起火", "烧毁", "漏电", "触电", "融化", "致伤", "安全隐患", "安全风险",
        "explosion", "fire", "burn", "electric shock", "electrocution", "melt",
        "injury", "safety hazard", "safety risk", "dangerous",

        # 隐私/安全
        "隐私泄露", "数据泄露", "账号被盗", "未授权访问",         "privacy leak", "data leak", "data breach", "unauthorized access",
        "collecting and sharing information", "privacy leak", "privacy risk", "privacy concerns", "privacy breach",
        "account compromised", "account takeover", "hacked", "data collection without consent", "unauthorized data collection",

        # 召回/监管
        "召回", "监管处罚", "禁售",
        "recall", "regulatory action", "sales ban", "banned by regulator",
    ),
    "medium": (
        # 充电/电池
        "发热", "发烫", "过热", "鼓包", "电池膨胀", "续航骤降", "断充", "接触不良", "无法充电",
        "overheat", "overheating", "hot to touch", "battery swelling", "swollen battery",
        "battery drain", "fails to charge", "not charging", "loose connection",

        # 功能失效
        "无法开机", "单侧无声", "黑屏", "死机", "固件变砖", "失去功能",
        "won't turn on", "will not turn on", "no sound", "black screen",
        "freezes", "firmware bricked", "bricked after update", "bricked after firmware", "stops working",

        # 质量/宣传
        "虚假宣传", "质量问题", "批次问题",
        "false advertising", "misleading", "defect", "defective", "quality issue",
    ),
    "low": (
        "卡顿", "不适", "看不清", "说明书", "配对困难", "包装破损", "物流延误",
        "uncomfortable", "unclear", "manual", "pairing issue",
        "packaging damaged", "shipping delay", "delayed delivery",
    ),
}

# 风险类型词表，用于识别 risk_type。
RISK_TYPE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "safety": (
        "爆炸", "起火", "烧毁", "漏电", "触电", "融化", "安全隐患",
        "explosion", "fire", "burn", "electric shock", "melt", "safety", "hazard", "dangerous",
    ),
    "privacy": (
        "隐私", "数据泄露", "未授权访问", "摄像头", "监控",
        "privacy", "data leak", "data breach", "unauthorized access", "camera",
    ),
    "security": (
        "账号被盗", "漏洞", "攻击", "入侵",
        "account compromised", "vulnerability", "attack", "hacked",
    ),
    "battery": (
        "电池", "续航", "充电", "鼓包", "发热", "发烫", "过热",
        "battery", "charging", "overheat", "swollen", "drain",
    ),
    "quality": (
        "质量", "故障", "损坏", " defective", "坏",
        "quality", "defect", "broken", "failure",
    ),
    "firmware": (
        "固件", "升级", "更新", "变砖",
        "firmware", "update", "bricked",
    ),
    "logistics": (
        "物流", "快递", "发货", "配送", "延误",
        "shipping", "delivery", "delay",
    ),
    "marketing": (
        "虚假宣传", "误导", "广告", "参数不符",
        "false advertising", "misleading", "advertising",
    ),
}


HARD_SAFETY_SIGNALS = (
    "fire", "explosion", "recall", "privacy leak", "overheating",
    "electric shock", "injury", "safety hazard",
    "起火", "爆炸", "召回", "隐私泄露", "过热", "触电", "安全事故",
)

def has_hard_safety_signal(text: str) -> bool:
    """明确安全/召回/隐私泄露信号，不允许被低风险判断覆盖。"""
    lowered = (text or "").lower()
    return any(signal.lower() in lowered for signal in HARD_SAFETY_SIGNALS)


def detect_risk(text: str) -> dict[str, Any]:
    """统一风险判断：Workflow 与 Agent 均调用此函数。"""
    text = text or ""
    lowered = text.lower()

    risk_level = "none"
    reasons: list[str] = []

    for level in ("high", "medium", "low"):
        hits = []
        for word in RISK_WORDS[level]:
            if word.lower() in lowered:
                hits.append(word)
        if hits:
            risk_level = level
            reasons.extend(hits)
            break

    hard_safety = has_hard_safety_signal(text)
    if hard_safety and risk_level not in ("medium", "high"):
        risk_level = "medium"
        reasons.append("hard safety signal")

    risk_types = []
    for risk_type, keywords in RISK_TYPE_KEYWORDS.items():
        if any(keyword.lower() in lowered for keyword in keywords):
            risk_types.append(risk_type)

    return {
        "risk_level": risk_level,
        "risk_score": {"none": 0, "low": 30, "medium": 60, "high": 90}[risk_level],
        "risk_types": sorted(set(risk_types)),
        "reasons": sorted(set(reasons)),
        "hard_safety_signal": hard_safety,
        "requires_human_review": hard_safety or risk_level in ("high", "medium"),
    }


def detect_sentiment(text: str) -> dict[str, Any]:
    """统一基础情绪判断：风险优先，避免安全事件被 neutral 稀释。"""
    text = text or ""
    lowered = text.lower()

    positive = (
        "满意", "喜欢", "不错", "推荐", "超出预期", "好评", "稳定", "舒服", "顺利",
        "satisfied", "love", "great", "excellent", "recommend", "stable", "comfortable",
    )
    negative = (
        "差", "失望", "退货", "换货", "问题", "隐患", "发热", "发烫", "松动", "断充", "无法", "危险", "担心",
        "bad", "terrible", "disappointed", "return", "refund", "issue", "problem",
        "overheat", "broken", "dangerous", "concern",
    )

    pos_hits = [w for w in positive if w.lower() in lowered]
    neg_hits = [w for w in negative if w.lower() in lowered]

    risk_result = detect_risk(text)
    risk_level = risk_result["risk_level"]

    # 1. 风险优先：中高风险事件必须是 negative
    if risk_level in ("medium", "high") or risk_result.get("hard_safety_signal"):
        sentiment = "negative"
    # 2. 正负信号都明显
    elif pos_hits and neg_hits:
        sentiment = "mixed"
    # 3. 正面为主
    elif pos_hits:
        sentiment = "positive"
    # 4. 负面为主
    elif neg_hits:
        sentiment = "negative"
    # 5. 无风险且无信号
    else:
        sentiment = "neutral"

    return {
        "sentiment": sentiment,
        "positive_signals": pos_hits,
        "negative_signals": neg_hits,
        "risk_level": risk_level,
        "hard_safety_signal": risk_result.get("hard_safety_signal", False),
    }

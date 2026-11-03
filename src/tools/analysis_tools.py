"""TechCompass — 分析与处理工具层。

覆盖目标中的：
- 查询重写
- 实体抽取
- 情绪判断
- 风险评分
- 影响推理
- 行动建议

这些工具先提供确定性/规则版实现，保证 Agent MVP 可运行、可测试；
后续可通过 LLM Tool Calling 或语义模型增强。
"""

from __future__ import annotations

import re
from typing import Any

from tools.registry import Tool
from analysis.risk_lexicon import detect_risk, detect_sentiment

RISK_WORDS = {
    "high": ["爆炸", "起火", "烧", "漏电", "触电", "融化", "安全", "隐患", "致伤", "safety", "hazard", "recall"],
    "medium": ["发热", "发烫", "过热", "断充", "无法开机", "单侧无声", "续航骤降", "接触不良", "overheat", "fire", "fails to charge"],
    "low": ["卡顿", "不适", "看不清", "说明书", "配对困难", "包装破损"],
}

SENTIMENT_POS = ["满意", "喜欢", "不错", "推荐", "超出预期", "好评", "稳定", "舒服"]
SENTIMENT_NEG = ["差", "失望", "退货", "换货", "问题", "隐患", "发热", "发烫", "松动", "断充", "无法", "危险"]

CATEGORY_RULES = [
    ("产品问题", ["发热", "发烫", "过热", "安全", "隐患", "松动", "断充", "无法", "无声", "电池", "续航", "损坏", "故障"]),
    ("物流问题", ["快递", "物流", "发货", "包装", "运输", "延误", "破损"]),
    ("宣传问题", ["宣传", "虚假", "误导", "介绍", "广告", "实际"]),
    ("使用问题", ["说明书", "配对", "连接", "蓝牙", "不会用", "佩戴", "操作"]),
]


def make_rewrite_query() -> Tool:
    """开放域、去主题化的保守查询扩展。

    规则：
    - 不预设品类、品牌、风险主题。
    - 不引入用户未提到的新主题。
    - 只保留原 query、关键短语和词汇组合。
    """
    def run(args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query", "")).strip()
        if not query:
            return {"original": "", "rewrites": [], "count": 0}

        # 基础分词：中英文混合，保留字母数字、中文和连字符
        terms = [x for x in re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*|[\u4e00-\u9fff]+", query) if x]
        expanded = {query}

        # 保守短语组合：相邻 2-3 个词组合，不引入外部同义词
        for n in (2, 3):
            for i in range(len(terms) - n + 1):
                expanded.add(" ".join(terms[i:i+n]).lower())

        # 保留原词
        for term in terms:
            expanded.add(term.lower())

        # 限制数量，避免无意义扩展
        rewrites = sorted(expanded)[:8]
        return {
            "original": query,
            "rewrites": rewrites,
            "count": len(rewrites),
        }
    return Tool(
        name="rewrite_query",
        description="开放域查询重写：只做保守扩展，不预设主题、品类或风险词。",
        args_schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        run=run,
        fallback=lambda: {"original": "", "rewrites": [], "count": 0},
    )

def make_extract_entities() -> Tool:
    def run(args: dict[str, Any]) -> dict[str, Any]:
        text = str(args.get("text", ""))
        lower = text.lower()
        entities = {
            "product_category": [],
            "product_model": [],
            "component": [],
            "channel": [],
            "risk_words": [],
        }
        categories = {
            "headphones": ["earbud", "headphone", "耳机", "airpods", "linkbuds"],
            "smartwatch": ["smartwatch", "smart watch", "腕表", "garmin", "fitbit", "galaxy watch"],
            "speaker": ["speaker", "soundlink", "emberton", "xboom"],
            "keyboard": ["keyboard", "hyperx", "xtrfy"],
            "mouse": ["mouse", "superstrike"],
            "laptop": ["laptop", "thinkpad", "t480"],
            "smart_tv": ["smart tv", "roku", "lg tv"],
            "smart_home": ["smart home", "homekit", "echo show", "adt", "home security", "camera"],
            "battery": ["battery", "电池", "续航", "usb-c", "usb c", "charging"],
            "toy": ["toy", "melissa & doug"],
        }
        for cat, words in categories.items():
            if any(w in lower for w in words):
                entities["product_category"].append(cat)
        for word in ("airpods", "sony", "jbl", "sennheiser", "garmin", "samsung"):
            if word in lower:
                entities["product_model"].append(word)
        for word in ("充电头", "充电线", "接口", "battery", "电池", "ear tip", "耳塞"):
            if word.lower() in lower:
                entities["component"].append(word)
        channels = ["reddit", "news", "review", "forum", "客服", "售后"]
        for ch in channels:
            if ch.lower() in lower:
                entities["channel"].append(ch)
        for level_words in RISK_WORDS.values():
            for w in level_words:
                if w in text:
                    entities["risk_words"].append(w)
        entities = {k: sorted(set(v)) for k, v in entities.items()}
        return {"entities": entities}
    return Tool(
        name="extract_entities",
        description="从文本中抽取产品品类、型号、部件、渠道和风险词实体。",
        args_schema={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        run=run,
        fallback=lambda: {"entities": {}},
    )


def make_analyze_sentiment() -> Tool:
    def run(args: dict[str, Any]) -> dict[str, Any]:
        text = str(args.get("text", ""))
        return detect_sentiment(text)
    return Tool(
        name="analyze_sentiment",
        description="判断文本情绪倾向：positive / neutral / negative，并给出信号词。中英文词表统一。",
        args_schema={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        run=run,
        fallback=lambda: {"sentiment": "neutral", "positive_signals": [], "negative_signals": []},
    )

def make_score_risk() -> Tool:
    def run(args: dict[str, Any]) -> dict[str, Any]:
        text = str(args.get("text", ""))
        result = detect_risk(text)
        return {
            "risk_level": result["risk_level"],
            "risk_score": result["risk_score"],
            "risk_types": result["risk_types"],
            "reasons": result["reasons"],
            "requires_human_review": result["requires_human_review"],
        }
    return Tool(
        name="score_risk",
        description="基于统一中英文风险词表评分。risk_level: none/low/medium/high。",
        args_schema={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        run=run,
        fallback=lambda: {"risk_level": "unknown", "risk_score": 0, "risk_types": [], "reasons": []},
    )

def make_reason_impact() -> Tool:
    def run(args: dict[str, Any]) -> dict[str, Any]:
        risk = str(args.get("risk_level", "low"))
        count = int(args.get("evidence_count", 0))
        spread = str(args.get("spread", "single"))
        mapping = {
            ("high", "broad"): "critical",
            ("high", "multiple"): "medium",
            ("high", "single"): "medium",
            ("medium", "broad"): "high",
            ("medium", "multiple"): "medium",
            ("medium", "single"): "medium",
            ("low", "broad"): "medium",
            ("low", "multiple"): "low",
            ("low", "single"): "low",
            ("none", "broad"): "low",
            ("none", "multiple"): "low",
            ("none", "single"): "none",
        }
        impact = mapping.get((risk, spread), "low" if risk == "low" else "none" if risk == "none" else "medium")
        if count == 0:
            impact = "none"
        return {
            "impact_level": impact,
            "risk_level": risk,
            "evidence_count": count,
            "spread": spread,
        }
    return Tool(
        name="reason_impact",
        description="基于风险等级、证据数量和传播范围推理影响级别。",
        args_schema={
            "type": "object",
            "properties": {
                "risk_level": {"type": "string", "enum": ["none", "low", "medium", "high"]},
                "evidence_count": {"type": "integer"},
                "spread": {"type": "string", "enum": ["single", "multiple", "broad"]},
            },
            "required": ["risk_level", "evidence_count", "spread"],
        },
        run=run,
        fallback=lambda: {"impact_level": "unknown"},
    )


def make_suggest_action() -> Tool:
    def run(args: dict[str, Any]) -> dict[str, Any]:
        impact = str(args.get("impact_level", "low"))
        actions = {
            "critical": {
                "customer_support": "立即升级至安全/质量负责人，逐条联系用户并暂停相关销售链接",
                "product": "启动质量排查与批次追踪，评估是否召回",
                "communication": "准备对外说明话术，避免扩大化传播",
            },
            "high": {
                "customer_support": "优先联系受影响用户，提供检测/退换方案",
                "product": "进入产品改进候选池并排查同批次",
                "communication": "统一客服口径，监测传播趋势",
            },
            "medium": {
                "customer_support": "记录售后跟进队列，48小时内回复",
                "product": "观察同类反馈是否聚集",
                "communication": "暂不对外声明，持续监测",
            },
            "low": {
                "customer_support": "常规跟进",
                "product": "无需立即动作",
                "communication": "常规监测",
            },
        }
        return {"impact_level": impact, "actions": actions.get(impact, actions["low"])}
    return Tool(
        name="suggest_action",
        description="根据影响级别生成客服、产品和传播侧行动建议。",
        args_schema={
            "type": "object",
            "properties": {"impact_level": {"type": "string", "enum": ["none", "low", "medium", "high", "critical"]}},
            "required": ["impact_level"],
        },
        run=run,
        fallback=lambda: {"impact_level": "unknown", "actions": {}},
    )


def register_analysis_tools(registry) -> None:
    """注册 Agent 可直接调用的分析入口。

    规则分析工具保留为内部函数，供 Offline Pipeline / Workflow rule backend 使用，
    不暴露给 LLM Agent，避免工具冗余和模型误调用。
    """
    registry.register(make_rewrite_query())

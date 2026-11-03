"""TechCompass — 评论分类/情绪/风险 评测运行器。

支持分类器：
- baseline_rule：确定性规则（离线、可复现）
- llm：真实 LLM（Tool Calling 强制结构化）

指标：
- category / sentiment / risk 三项准确率
- 漏报率（风险正样本被判成非风险）
- 误报率（非风险被判成风险）
- 总体准确率（三项都正确）
"""

from __future__ import annotations

import json
from typing import Any, Callable

from langchain_openai import ChatOpenAI

NEG_WORDS = ("发热", "烫", "安全", "隐患", "松动", "断充", "无法", "失望", "破损", "误导",
             "没效果", "太", "问题", "没人", "电流声", "晚", "接触不良", "充不进",
             "焦味", "不敢", "退货", "断开", "无声", "融化", "烫手", "下降")

RISK_WORDS = ("发热", "烫", "烫手", "安全", "隐患", "焦味", "融化", "烧", "爆炸",
              "漏电", "爆炸", "不敢", "无法开机", "断电", "损坏")


def baseline_rule(review: str) -> dict[str, str]:
    """确定性规则分类器。"""
    if any(w in review for w in ("快递", "发货", "物流", "催", "三天不更新")):
        category = "物流问题"
    elif any(w in review for w in ("宣传", "误导", "虚假", "介绍", "快充")):
        category = "宣传问题"
    elif any(w in review for w in ("说明书", "配对", "连接", "不会", "折腾", "蓝牙", "第一次")):
        category = "使用问题"
    elif any(w in review for w in NEG_WORDS):
        category = "产品问题"
    else:
        category = "其他"

    neg = any(w in review for w in NEG_WORDS)
    pos = any(w in review for w in ("满意", "清晰", "舒服", "喜欢", "不错", "顺利", "正常现象",
                                    "推荐", "点赞", "超出预期", "解决", "耐心", "便携", "精致"))
    if neg:
        sentiment = "negative"
    elif pos:
        sentiment = "positive"
    else:
        sentiment = "neutral"

    risk = "pos" if any(w in review for w in RISK_WORDS) else "neg"
    return {"category": category, "sentiment": sentiment, "risk": risk}


def classify_llm(llm: ChatOpenAI, review: str) -> dict[str, str]:
    """真实 LLM 分类，返回 category/sentiment/risk。"""
    tool_def = {
        "type": "function",
        "function": {
            "name": "classify_review",
            "description": "分类评论主题、情绪与风险",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "enum": ["产品问题", "物流问题", "宣传问题", "使用问题", "其他"]},
                    "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
                    "risk": {"type": "string", "enum": ["pos", "neg"], "description": "是否涉及安全/故障告警类风险"},
                },
                "required": ["category", "sentiment", "risk"],
            },
        },
    }
    from langchain_core.messages import SystemMessage, HumanMessage
    llm_tools = llm.bind_tools([tool_def])
    msg = llm_tools.invoke([
        SystemMessage(content="分类这条数码评论的主题、情绪与风险等级，调用 classify_review 工具。"),
        HumanMessage(content=review),
    ])
    tool_calls = getattr(msg, "tool_calls", None) or []
    if tool_calls and tool_calls[0].get("args"):
        a = tool_calls[0]["args"]
        return {
            "category": a.get("category", "其他"),
            "sentiment": a.get("sentiment", "neutral"),
            "risk": a.get("risk", "neg"),
        }
    return baseline_rule(review)


def load_eval_set(path: str) -> list[dict[str, Any]]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def run_classify_eval(
    path: str,
    classifier: Callable[[str], dict[str, str]],
    max_items: int | None = None,
) -> dict[str, Any]:
    items = load_eval_set(path)
    if max_items:
        items = items[:max_items]

    cat_correct = sent_correct = risk_correct = both_correct = 0
    # 漏报/误报计数
    fn_count = 0   # 漏报：真实 pos，预测 neg
    fp_count = 0   # 误报：真实 neg，预测 pos
    pos_total = 0
    neg_total = 0
    details = []

    for it in items:
        pred = classifier(it["review"])
        cat_ok = pred["category"] == it["label_category"]
        sent_ok = pred["sentiment"] == it["label_sentiment"]
        risk_ok = pred.get("risk", "neg") == it["label_risk"]
        both_ok = cat_ok and sent_ok and risk_ok

        cat_correct += cat_ok
        sent_correct += sent_ok
        risk_correct += risk_ok
        both_correct += both_ok

        true_risk = it["label_risk"]
        pred_risk = pred.get("risk", "neg")
        if true_risk == "pos":
            pos_total += 1
            if pred_risk == "neg":
                fn_count += 1
        else:
            neg_total += 1
            if pred_risk == "pos":
                fp_count += 1

        details.append({
            "id": it["id"], "review": it["review"],
            "true_category": it["label_category"], "pred_category": pred["category"],
            "true_sentiment": it["label_sentiment"], "pred_sentiment": pred["sentiment"],
            "true_risk": true_risk, "pred_risk": pred_risk,
            "category_ok": cat_ok, "sentiment_ok": sent_ok, "risk_ok": risk_ok,
        })

    n = len(items)
    return {
        "n": n,
        "category_accuracy": round(cat_correct / n, 4) if n else 0,
        "sentiment_accuracy": round(sent_correct / n, 4) if n else 0,
        "risk_accuracy": round(risk_correct / n, 4) if n else 0,
        "overall_accuracy": round(both_correct / n, 4) if n else 0,
        "recall": round((pos_total - fn_count) / pos_total, 4) if pos_total else None,
        "false_positive_rate": round(fp_count / neg_total, 4) if neg_total else None,
        "false_negative_count": fn_count,
        "false_positive_count": fp_count,
        "pos_total": pos_total,
        "neg_total": neg_total,
        "details": details,
    }


def report(summary: dict[str, Any]) -> str:
    lines = [
        f"样本数: {summary['n']}（风险正样本 {summary['pos_total']} / 负样本 {summary['neg_total']}）",
        f"主题分类准确率: {summary['category_accuracy']:.2%}",
        f"情绪判断准确率: {summary['sentiment_accuracy']:.2%}",
        f"风险判断准确率: {summary['risk_accuracy']:.2%}",
        f"总体(三项都对)准确率: {summary['overall_accuracy']:.2%}",
        f"风险召回率(recall): {summary['recall']:.2%}" if summary['recall'] is not None else "风险召回率: n/a",
        f"误报率(FPR): {summary['false_positive_rate']:.2%}" if summary['false_positive_rate'] is not None else "误报率: n/a",
        f"漏报数: {summary['false_negative_count']}，误报数: {summary['false_positive_count']}",
        "",
        "错误样例：",
    ]
    for d in summary["details"]:
        if not (d["category_ok"] and d["sentiment_ok"] and d["risk_ok"]):
            lines.append(
                f"  [{d['id']}] 主题({d['true_category']}/{d['pred_category']}) "
                f"情绪({d['true_sentiment']}/{d['pred_sentiment']}) "
                f"风险({d['true_risk']}/{d['pred_risk']}) | {d['review']}"
            )
    return "\n".join(lines)

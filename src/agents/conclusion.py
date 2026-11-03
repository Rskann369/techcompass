"""TechCompass — 结构化结果 → 运营可读自然语言结论层。

把一次 Agent 主链路输出的 JSON trace / 结构化字段，翻译成运营可直接读的自然语言：

- headline      ：一句话结论
- summary       ：结论正文（情绪 / 风险 / 根因 / 影响 / 建议）
- evidence_basis：结论依据，逐条给出标题 + 来源 + 具体链接 + 相关性
- confidence    ：置信度（等级 + 数值 + 解释）

设计原则：
- evidence_basis 与 confidence 始终由结构化数据确定性计算，保证「依据必带链接」。
- headline / summary 支持两种后端：
    rule —— 确定性模板（离线、快、稳定，默认）
    llm  —— 用 LLM 润色自然语言（可选，失败自动降级到 rule）
- 不新增第三方依赖，复用现有 langchain_openai + prompts。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from project_paths import DEFAULT_ENV_PATH

SENTIMENT_CN = {
    "positive": "积极",
    "neutral": "中性",
    "negative": "负面",
    "mixed": "混合（正负并存）",
}

LEVEL_CN = {
    "none": "无",
    "low": "低",
    "medium": "中",
    "high": "高",
    "critical": "严重",
}

ISSUE_TYPE_CN = {
    "product_quality": "产品质量",
    "safety": "安全",
    "privacy": "隐私",
    "security": "安全/漏洞",
    "logistics": "物流",
    "marketing_claim": "宣传/营销",
    "usability": "易用性",
    "after_sales": "售后",
    "firmware": "固件",
    "compatibility": "兼容性",
    "other": "其他",
    "unknown": "未识别",
}

SOURCE_CN = {
    "gnews": "GNews 新闻",
    "reddit_rss": "Reddit 讨论",
    "hackernews": "Hacker News 讨论",
    "newsapi": "NewsAPI 新闻",
    "dataset": "评论数据集",
}

CONFIDENCE_HINTS = {
    "high": "证据充分且相关性高，结论可采信",
    "medium": "证据基本可用，建议结合人工复核后决策",
    "low": "证据不足或相关性偏弱，仅作参考，不建议据此行动",
}


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _similarity_of(doc: dict[str, Any]) -> float:
    raw = doc.get("_semantic_similarity", doc.get("similarity", 0.0))
    try:
        return _clamp(float(raw))
    except (TypeError, ValueError):
        return 0.0


def _relevance_label(sim: float) -> str:
    if sim >= 0.70:
        return "高度相关"
    if sim >= 0.60:
        return "相关"
    if sim >= 0.55:
        return "弱相关"
    return "低相关"


def _actions_of(result: dict[str, Any]) -> dict[str, str]:
    raw = result.get("actions") or {}
    if isinstance(raw, dict) and isinstance(raw.get("actions"), dict):
        return raw["actions"]
    return raw if isinstance(raw, dict) else {}


def _sentiment_label(sentiment: Any) -> str:
    if isinstance(sentiment, dict):
        value = sentiment.get("sentiment", "neutral")
    else:
        value = sentiment or "neutral"
    return SENTIMENT_CN.get(value, str(value))


def _risk_label(risk: Any) -> str:
    if isinstance(risk, dict):
        value = risk.get("risk_level", "none")
    else:
        value = risk or "none"
    return LEVEL_CN.get(value, str(value))


def _impact_label(impact: Any) -> str:
    if isinstance(impact, dict):
        value = impact.get("impact_level", "none")
    else:
        value = impact or "none"
    return LEVEL_CN.get(value, str(value))


class ConclusionBuilder:
    """把主链路结果翻译成自然语言结论。

    use_llm=True 时用 LLM 润色 headline/summary；失败或关闭时回落到确定性模板。
    evidence_basis / confidence 始终确定性生成，保证链接不丢失。
    """

    def __init__(self, use_llm: bool = False, request_timeout: float = 60.0):
        self.use_llm = use_llm
        self.request_timeout = request_timeout

    # ------------------------------------------------------------------ #
    # 公共入口
    # ------------------------------------------------------------------ #
    def build(self, result: dict[str, Any]) -> dict[str, Any]:
        question = str(result.get("question", ""))
        evidence = self._evidence(result)
        refused = bool(result.get("refused", False))

        evidence_basis = self._build_evidence_basis(evidence)
        confidence = self._build_confidence(result, evidence, refused)

        prose = self._prose(result, evidence_basis, confidence, refused)
        return {
            "headline": prose["headline"],
            "summary": prose["summary"],
            "sentiment": _sentiment_label(result.get("sentiment")),
            "risk_level": _risk_label(result.get("risk")),
            "impact_level": _impact_label(result.get("impact")),
            "evidence_basis": evidence_basis,
            "confidence": confidence,
            "actions": list(self._action_lines(result).values()),
            "refused": refused,
        }

    # ------------------------------------------------------------------ #
    # 依据 & 置信度（确定性，保证带链接）
    # ------------------------------------------------------------------ #
    def _evidence(self, result: dict[str, Any]) -> list[dict[str, Any]]:
        return result.get("relevant_evidence") or result.get("evidence") or []

    def _build_evidence_basis(self, evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
        basis: list[dict[str, Any]] = []
        for i, doc in enumerate(evidence, start=1):
            sim = _similarity_of(doc)
            source = str(doc.get("source") or "")
            basis.append({
                "ref": i,
                "title": str(doc.get("title") or "").strip(),
                "url": str(doc.get("url") or "").strip(),
                "source": source,
                "source_label": SOURCE_CN.get(source, source or "未知来源"),
                "source_type": str(doc.get("source_type") or ""),
                "published_at": doc.get("published_at"),
                "similarity": round(sim, 4),
                "relevance": _relevance_label(sim),
                "excerpt": (str(doc.get("content") or doc.get("text") or "").strip())[:220],
            })
        return basis

    def _build_confidence(
        self,
        result: dict[str, Any],
        evidence: list[dict[str, Any]],
        refused: bool,
    ) -> dict[str, Any]:
        n = len(evidence)
        sims = [_similarity_of(d) for d in evidence]
        avg_sim = round(sum(sims) / len(sims), 4) if sims else 0.0

        analysis = result.get("analysis") or {}
        evidence_sufficient = bool(analysis.get("evidence_sufficient", not refused))
        llm_conf = analysis.get("confidence")

        if refused or not evidence_sufficient or n == 0:
            level = "low"
            score = round(avg_sim, 4)
        else:
            score = avg_sim
            if n >= 3 and avg_sim >= 0.60:
                score = max(score, 0.70)
            elif n >= 2 and avg_sim >= 0.55:
                score = max(score, 0.60)
            if llm_conf is not None:
                try:
                    score = (score + _clamp(float(llm_conf))) / 2
                except (TypeError, ValueError):
                    pass
            score = round(_clamp(score), 4)
            if score >= 0.70:
                level = "high"
            elif score >= 0.50:
                level = "medium"
            else:
                level = "low"

        parts = [f"证据 {n} 条", f"平均语义相似度 {avg_sim}"]
        if llm_conf is not None:
            parts.append(f"模型置信度 {llm_conf}")
        if refused or not evidence_sufficient:
            parts.append("证据充分性 gate 未通过")
        if result.get("risk", {}).get("hard_safety_signal"):
            parts.append("命中硬安全信号，需人工复核")
        explanation = "；".join(parts) + "。" + CONFIDENCE_HINTS[level]

        return {
            "level": level,
            "score": score,
            "evidence_count": n,
            "average_similarity": avg_sim,
            "explanation": explanation,
        }

    # ------------------------------------------------------------------ #
    # 自然语言正文（rule 模板 + 可选 LLM 润色）
    # ------------------------------------------------------------------ #
    def _action_lines(self, result: dict[str, Any]) -> dict[str, str]:
        actions = _actions_of(result)
        labels = {
            "customer_support": "客服",
            "product": "产品",
            "communication": "传播",
        }
        lines: dict[str, str] = {}
        for key, label in labels.items():
            value = str(actions.get(key, "")).strip()
            if value:
                lines[key] = f"{label}：{value}"
        return lines

    def _prose(
        self,
        result: dict[str, Any],
        evidence_basis: list[dict[str, Any]],
        confidence: dict[str, Any],
        refused: bool,
    ) -> dict[str, str]:
        question = str(result.get("question", ""))
        n = confidence["evidence_count"]
        sentiment = _sentiment_label(result.get("sentiment"))
        risk = _risk_label(result.get("risk"))
        impact = _impact_label(result.get("impact"))
        analysis = result.get("analysis") or {}
        actions = self._action_lines(result)

        if refused:
            headline = f"针对「{question}」，当前未检索到足够可靠的公开证据，无法给出舆情结论。"
            summary = (
                "系统执行了证据充分性校验（evidence gate）后判定证据不足，"
                "因此不生成风险结论、行动建议、周报或告警，避免在弱证据上强行输出模板化结论。"
                "建议：补充更精确的关键词、更换更聚焦的问题，或扩展数据源后重试。"
            )
            if self.use_llm:
                return self._llm_polish(question, result, evidence_basis, confidence, headline, summary)
            return {"headline": headline, "summary": summary}

        headline = (
            f"针对「{question}」，基于 {n} 条公开舆情证据，"
            f"整体情绪{'' if sentiment.endswith('）') else '为'}{sentiment}，"
            f"风险等级{risk}，影响等级{impact}。"
        )

        body: list[str] = []
        issue_type = analysis.get("issue_type")
        if issue_type and issue_type not in ("unknown",):
            body.append(f"问题类型：{ISSUE_TYPE_CN.get(issue_type, str(issue_type))}。")

        root_cause = str(analysis.get("root_cause") or "").strip()
        if root_cause:
            body.append(f"根因判断：{root_cause}。")

        affected_scope = str(analysis.get("affected_scope") or "").strip()
        if affected_scope:
            body.append(f"影响范围：{affected_scope}。")

        risk_types = result.get("risk", {}).get("risk_types") or []
        if risk_types:
            body.append(f"风险类型：{', '.join(str(t) for t in risk_types)}。")

        if result.get("risk", {}).get("requires_human_review"):
            body.append("存在需人工复核的安全信号，建议优先人工确认。")

        if not body:
            body.append("未识别到明确的根因或影响范围，建议结合下方证据进一步人工研判。")

        if not actions:
            body.append("暂未生成明确行动建议，建议持续监测。")

        summary = " ".join(body)

        if self.use_llm:
            return self._llm_polish(question, result, evidence_basis, confidence, headline, summary)
        return {"headline": headline, "summary": summary}

    # ------------------------------------------------------------------ #
    # 可选 LLM 润色（失败自动回退模板）
    # ------------------------------------------------------------------ #
    def _llm_polish(
        self,
        question: str,
        result: dict[str, Any],
        evidence_basis: list[dict[str, Any]],
        confidence: dict[str, Any],
        headline: str,
        summary: str,
    ) -> dict[str, str]:
        try:
            import os

            from dotenv import load_dotenv
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_openai import ChatOpenAI
            from pydantic import BaseModel, Field

            load_dotenv(DEFAULT_ENV_PATH)
            llm = ChatOpenAI(
                model=os.getenv("OPENAI_MODEL"),
                base_url=os.getenv("OPENAI_BASE_URL"),
                api_key=os.getenv("OPENAI_API_KEY"),
                temperature=0,
                request_timeout=self.request_timeout,
            )

            class Conclusion(BaseModel):
                headline: str = Field(description="一句话结论，面向运营，客观、克制")
                summary: str = Field(description="结论正文，2-4 句话，覆盖情绪/风险/根因/影响/建议")

            system = (
                "你是 TechCompass 的结论解读模块。请把结构化舆情分析翻译成运营可直接读的自然语言。"
                "必须客观、克制，不夸大风险，不臆测。保留原有事实与建议，不改动语义。"
            )
            user = (
                f"用户问题：{question}\n"
                f"模板一句话结论：{headline}\n"
                f"模板正文：{summary}\n"
                f"置信度：{confidence['level']}（{confidence['explanation']}）\n"
                "结论依据（含链接）：\n"
                + "\n".join(
                    f"[{e['ref']}] {e['title']} | {e['source_label']} | {e['url']}"
                    for e in evidence_basis
                )
            )
            prompt = ChatPromptTemplate.from_messages([("system", system), ("human", user)])
            structured = llm.with_structured_output(Conclusion)
            obj = structured.invoke(prompt.invoke({}))
            return {"headline": obj.headline, "summary": obj.summary}
        except Exception:
            # 任何 LLM 失败（网关慢/超时/结构化失败）都回退到确定性模板。
            return {"headline": headline, "summary": summary}



def build_conclusion_from_analysis(
    question: str,
    analysis: dict[str, Any],
    evidence_docs: list[dict[str, Any]],
    top_k: int = 5,
    threshold: float = 0.55,
) -> dict[str, Any]:
    """真实 LLM 主链路的统一结论入口。

    输入：真实 LLM 的 analyze_consumer_sentiment 结构化结果 + 检索证据。
    输出：{"conclusion": {...}, "structured": {...}}

    - 用 SemanticReranker 对证据做语义排序 + 阈值过滤（补上 _semantic_similarity）。
    - 用统一风险词表做硬安全信号兜底，避免模型低风险判断覆盖安全信号。
    - 最终交给 ConclusionBuilder 生成运营可读结论（含带链接依据 + 置信度）。
    """
    from rag.semantic_rerank import SemanticReranker
    from analysis.risk_lexicon import detect_risk

    reranker = SemanticReranker()
    relevant = reranker.rerank(question, list(evidence_docs), top_k=top_k, threshold=threshold)

    risk_level = analysis.get("risk_level", "none")
    if relevant:
        evidence_text = " ".join(
            f"{d.get('title', '')} {d.get('content') or d.get('text') or ''}" for d in relevant
        )
        ev_risk = detect_risk(evidence_text)
        if ev_risk.get("hard_safety_signal") and risk_level not in ("medium", "high", "critical"):
            risk_level = "medium"

    risk_type = analysis.get("risk_type", "")
    risk_types = [risk_type] if risk_type else []

    refused = not analysis.get("evidence_sufficient", True) or not relevant

    result = {
        "question": question,
        "refused": refused,
        "sentiment": {"sentiment": analysis.get("sentiment", "neutral")},
        "risk": {
            "risk_level": risk_level,
            "risk_types": risk_types,
            "requires_human_review": risk_level in ("medium", "high", "critical"),
        },
        "impact": {"impact_level": analysis.get("impact_level", "none")},
        "actions": {
            "actions": {
                "customer_support": analysis.get("customer_support_action", ""),
                "product": analysis.get("product_action", ""),
                "communication": analysis.get("communication_action", ""),
            }
        },
        "analysis": analysis,
        "relevant_evidence": relevant,
    }
    conclusion = ConclusionBuilder().build(result)
    return {"conclusion": conclusion, "structured": result}


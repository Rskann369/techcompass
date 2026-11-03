"""TechCompass — 模型驱动的开放域分析工具层。

设计原则：
- 代码不预设 topic / 品类 / 风险词。
- 通过 Tool Calling Schema 约束模型输出。
- 模型基于用户问题和证据自行判断、分类、抽取和推理。
- 工具只负责：输入组织、输出校验、失败降级。

核心工具：
- judge_evidence：判断证据是否相关、充分、可信。
- analyze_consumer_sentiment：开放域舆情结构化分析。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from project_paths import DEFAULT_ENV_PATH
from tools.registry import Tool
from prompts.classify_prompt import CLASSIFY_SYSTEM
from prompts.entity_prompt import ENTITY_SYSTEM
from prompts.reason_prompt import REASON_SYSTEM
from prompts.impact_prompt import IMPACT_SYSTEM
from prompts.action_prompt import ACTION_SYSTEM


JUDGE_SYSTEM_PROMPT = (
    "你是 TechCompass 的证据判断模块。"
    "你必须只基于给定证据判断，不使用外部知识。"
    "判断维度：相关性、充分性、可信度、覆盖面、证据缺口。"
)

ANALYSIS_SYSTEM_PROMPT = "\n\n".join([
    REASON_SYSTEM,
    ENTITY_SYSTEM,
    IMPACT_SYSTEM,
    ACTION_SYSTEM,
    CLASSIFY_SYSTEM,
    (
        "你是 TechCompass 的开放域消费品舆情分析模块。"
        "你不得预设主题、品类、品牌或风险类型。"
        "你必须只基于用户问题和编号证据进行客观分析。"
        "如果证据不足，必须输出 evidence_sufficient=false，不得编造事实。"
        "如果问题不属于消费品舆情，输出 issue_type=other 或 unknown。"
        "不要为了填满字段而猜测。"
        "你必须调用 emit_consumer_sentiment_analysis 工具返回结构化结果，禁止直接输出文本。"
        "product_categories 使用你从证据中实际识别到的小写下划线标签；"
        "如果证据不足以判断品类，输出空数组或 other。"
        "risk_level 和 impact_level 必须由证据支持，不得因关键词存在而自动升级。"
        "evidence_refs 必须引用输入证据编号，例如 [1]、[2]。"
    ),
])


class ConsumerSentimentAnalysis(BaseModel):
    """LangChain structured output schema for consumer sentiment analysis."""

    issue_type: Literal[
        "product_quality", "safety", "privacy", "security", "logistics",
        "marketing_claim", "usability", "after_sales", "firmware",
        "compatibility", "other", "unknown",
    ] = Field(description="问题类型")
    product_categories: list[str] = Field(default_factory=list)
    product_models: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    sentiment: Literal["positive", "neutral", "negative", "mixed"]
    sentiment_signals: list[str] = Field(default_factory=list)
    risk_type: str = ""
    risk_level: Literal["none", "low", "medium", "high", "critical"]
    root_cause: str = ""
    impact_level: Literal["none", "low", "medium", "high", "critical"]
    affected_scope: str = ""
    evidence_sufficient: bool
    evidence_refs: list[str] = Field(default_factory=list)
    customer_support_action: str = ""
    product_action: str = ""
    communication_action: str = ""
    confidence: float = Field(ge=0, le=1)


ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", ANALYSIS_SYSTEM_PROMPT),
    ("human", "用户问题：{question}\n\n证据列表：\n{evidence}\n\n请客观分析，并引用证据编号。"),
])


def _build_llm(request_timeout: float = 120, callbacks: list[Any] | None = None) -> ChatOpenAI:
    load_dotenv(DEFAULT_ENV_PATH)
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0,
        request_timeout=request_timeout,
        callbacks=callbacks,
    )


def _invoke_tool_llm(llm: ChatOpenAI, tool_def: dict[str, Any], system_prompt: str, user_prompt: str) -> dict[str, Any]:
    llm_with_tools = llm.bind_tools([tool_def])
    last_error: Exception | None = None
    for attempt in range(2):
        response = llm_with_tools.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        tool_calls = getattr(response, "tool_calls", None) or []
        if tool_calls and tool_calls[0].get("args"):
            return dict(tool_calls[0]["args"])
        last_error = ValueError("model did not return structured tool call")
        if attempt == 0:
            # 第二次尝试时进一步压缩输入，降低超时概率
            user_prompt = user_prompt[:4000]
    raise last_error or ValueError("model did not return structured tool call")


JUDGE_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "emit_evidence_judgement",
        "description": "输出对检索证据质量的判断",
        "parameters": {
            "type": "object",
            "properties": {
                "relevance": {"type": "string", "enum": ["high", "medium", "low", "none"]},
                "sufficiency": {"type": "string", "enum": ["sufficient", "partial", "insufficient", "none"]},
                "credibility": {"type": "string", "enum": ["high", "medium", "low", "unknown"]},
                "coverage": {"type": "string", "enum": ["broad", "moderate", "narrow", "none"]},
                "evidence_gaps": {"type": "array", "items": {"type": "string"}},
                "reason": {"type": "string"},
            },
            "required": ["relevance", "sufficiency", "credibility", "coverage", "reason"],
        },
    },
}

ANALYSIS_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "emit_consumer_sentiment_analysis",
        "description": "输出开放域消费品舆情结构化分析",
        "parameters": {
            "type": "object",
            "properties": {
                "issue_type": {
                    "type": "string",
                    "enum": [
                        "product_quality", "safety", "privacy", "security", "logistics",
                        "marketing_claim", "usability", "after_sales", "firmware",
                        "compatibility", "other", "unknown",
                    ],
                },
                "product_categories": {"type": "array", "items": {"type": "string"}},
                "product_models": {"type": "array", "items": {"type": "string"}},
                "entities": {"type": "array", "items": {"type": "string"}},
                "sentiment": {"type": "string", "enum": ["positive", "neutral", "negative", "mixed"]},
                "sentiment_signals": {"type": "array", "items": {"type": "string"}},
                "risk_type": {"type": "string"},
                "risk_level": {"type": "string", "enum": ["none", "low", "medium", "high", "critical"]},
                "root_cause": {"type": "string"},
                "impact_level": {"type": "string", "enum": ["none", "low", "medium", "high", "critical"]},
                "affected_scope": {"type": "string"},
                "evidence_sufficient": {"type": "boolean"},
                "evidence_refs": {"type": "array", "items": {"type": "string"}},
                "customer_support_action": {"type": "string"},
                "product_action": {"type": "string"},
                "communication_action": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": [
                "issue_type", "product_categories", "entities", "sentiment",
                "risk_level", "impact_level", "evidence_sufficient", "confidence",
            ],
        },
    },
}


def make_judge_evidence(request_timeout: float = 120) -> Tool:
    def run(args: dict[str, Any]) -> dict[str, Any]:
        question = str(args.get("question", ""))
        evidence = args.get("evidence", [])
        if not question:
            raise ValueError("question is required")

        evidence_text = "\n".join(
            f"[{i}] {str(item)[:2000]}" for i, item in enumerate(evidence, start=1)
        ) or "(no evidence)"
        user_prompt = (
            f"用户问题：{question}\n\n"
            f"证据列表：\n{evidence_text}\n\n"
            "请判断这些证据能否支持回答用户问题。"
        )
        llm = _build_llm(request_timeout, callbacks=callbacks)
        result = _invoke_tool_llm(llm, JUDGE_TOOL_DEF, JUDGE_SYSTEM_PROMPT, user_prompt)
        return result
    return Tool(
        name="judge_evidence",
        description="让模型判断检索证据的相关性、充分性、可信度和覆盖面。适合任意主题，不预设关键词。",
        args_schema={
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "evidence": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["question", "evidence"],
        },
        run=run,
        fallback=lambda: {
            "relevance": "unknown", "sufficiency": "none", "credibility": "unknown",
            "coverage": "none", "evidence_gaps": ["LLM evidence judgement unavailable"],
            "reason": "fallback",
        },
    )


CATEGORY_ALIASES = {
    "wireless earbuds": "headphones",
    "wireless_earbuds": "headphones",
    "earbuds": "headphones",
    "headphone": "headphones",
    "headphones": "headphones",
    "audio": "headphones",
    "smart tv": "smart_tv",
    "tv": "smart_tv",
    "gaming keyboard": "keyboard",
    "keyboard": "keyboard",
    "gaming mouse": "mouse",
    "mouse": "mouse",
    "smart home camera": "smart_home",
    "smart home": "smart_home",
    "camera": "smart_home",
    "speaker": "speaker",
    "bluetooth speaker": "speaker",
    "smartwatch": "smartwatch",
    "smart watch": "smartwatch",
    "watch": "smartwatch",
    "laptop": "laptop",
    "battery": "battery",
}


def _normalize_categories(categories: list[str]) -> list[str]:
    normalized = []
    for category in categories:
        key = str(category).strip().lower().replace(" ", "_")
        normalized.append(CATEGORY_ALIASES.get(key, key))
    return sorted(set(normalized))


def _format_evidence(evidence: list[Any]) -> tuple[str, list[str]]:
    """把字符串或结构化证据统一转成编号文本。"""
    lines: list[str] = []
    for i, item in enumerate(evidence, start=1):
        if isinstance(item, dict):
            title = str(item.get("title", "")).strip()
            source = str(item.get("source", "") or item.get("source_name", "") or "").strip()
            url = str(item.get("url", "") or "").strip()
            content = str(item.get("content", "") or item.get("text", "") or "").strip()
            parts = []
            if title:
                parts.append(f"title: {title}")
            if source:
                parts.append(f"source: {source}")
            if url:
                parts.append(f"url: {url}")
            if content:
                parts.append(f"content: {content[:1200]}")
            text = "; ".join(parts)
        else:
            text = str(item)[:1200]
        lines.append(f"[{i}] {text}")
    return "\n".join(lines), [f"[{i}]" for i in range(1, len(evidence) + 1)]


def make_analyze_consumer_sentiment(request_timeout: float = 120, callbacks: list[Any] | None = None) -> Tool:
    def run(args: dict[str, Any]) -> dict[str, Any]:
        question = str(args.get("question", ""))
        evidence = args.get("evidence", [])
        if not question:
            raise ValueError("question is required")

        evidence_text, evidence_refs = _format_evidence(evidence)
        if not evidence_text:
            evidence_text = "(no evidence)"

        prompt_value = ANALYSIS_PROMPT.invoke({
            "question": question,
            "evidence": evidence_text,
        })
        llm = _build_llm(request_timeout, callbacks=callbacks)

        # Preferred path: LangChain structured output with Pydantic schema.
        try:
            structured_llm = llm.with_structured_output(ConsumerSentimentAnalysis)
            analysis_obj = structured_llm.invoke(prompt_value)
            result = analysis_obj.model_dump()
        except Exception:
            # Fallback: OpenAI-compatible tool calling, required for gateways
            # that do not reliably support structured output.
            user_prompt = prompt_value.to_string()
            result = _invoke_tool_llm(
                llm, ANALYSIS_TOOL_DEF, ANALYSIS_SYSTEM_PROMPT, user_prompt
            )

        result["product_categories"] = _normalize_categories(result.get("product_categories", []))
        if not result.get("evidence_refs"):
            result["evidence_refs"] = evidence_refs
        return result
    return Tool(
        name="analyze_consumer_sentiment",
        description="开放域模型分析：不预设主题，由模型基于编号证据客观判断品类、实体、情绪、风险、影响和建议。",
        args_schema={
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "evidence": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "证据列表，支持字符串或包含 title/source/url/content 的结构化对象。",
                },
            },
            "required": ["question", "evidence"],
        },
        run=run,
        fallback=lambda: {
            "issue_type": "unknown", "product_categories": [], "entities": [],
            "sentiment": "neutral", "risk_level": "none", "impact_level": "none",
            "evidence_sufficient": False, "confidence": 0,
            "customer_support_action": "证据不足，需人工复核",
            "product_action": "暂无动作", "communication_action": "暂无动作",
        },
    )

def register_llm_analysis_tools(registry) -> None:
    """只注册最终开放域分析工具；证据充分性由该工具输出。"""
    registry.register(make_analyze_consumer_sentiment())

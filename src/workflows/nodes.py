"""TechCompass — Workflow 节点定义。

把 Agent 链路拆成职责单一的节点，每个节点都是「输入 → 处理 → 输出」，
由 runner 按顺序调度。Week 2 先用纯 Python 函数实现，Week 3 再引入
工具调用与更多 LLM 推理。

节点链路：
  ingest -> normalize -> index_retrieve -> analyze -> reason -> output -> archive
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from adapters.schema import Item


# --------------------------------------------------------------------------- #
# 共享 State：节点之间传递数据的容器
# --------------------------------------------------------------------------- #
@dataclass
class WorkflowState:
    raw_items: list[Item] = field(default_factory=list)
    normalized: list[Item] = field(default_factory=list)
    retrieved: list[dict[str, Any]] = field(default_factory=list)
    analysis: dict[str, Any] = field(default_factory=dict)
    recommendation: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)
    log: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def record(self, msg: str) -> None:
        self.log.append(msg)


# --------------------------------------------------------------------------- #
# 节点实现
# --------------------------------------------------------------------------- #


    def to_checkpoint(self) -> dict[str, Any]:
        """序列化可恢复状态。Item 转为 dict。"""
        def serialize_items(items):
            return [it.to_dict() if isinstance(it, Item) else it for it in items]
        return {
            "raw_items": serialize_items(self.raw_items),
            "normalized": serialize_items(self.normalized),
            "retrieved": self.retrieved,
            "analysis": self.analysis,
            "recommendation": self.recommendation,
            "output": self.output,
            "log": self.log,
            "errors": self.errors,
        }

    @classmethod
    def from_checkpoint(cls, data: dict[str, Any]) -> "WorkflowState":
        """从 checkpoint 恢复状态。当前恢复为 dict 容器，分析节点兼容 dict。"""
        state = cls()
        state.raw_items = data.get("raw_items", [])
        state.normalized = data.get("normalized", [])
        state.retrieved = data.get("retrieved", [])
        state.analysis = data.get("analysis", {})
        state.recommendation = data.get("recommendation", {})
        state.output = data.get("output", {})
        state.log = data.get("log", [])
        state.errors = data.get("errors", [])
        return state

def node_ingest(state: WorkflowState, items: list[Item]) -> WorkflowState:
    """IngestSource：接收原始 Item（由适配器产出）。"""
    state.raw_items = items
    state.record(f"ingest: {len(items)} items received")
    return state


def node_normalize(state: WorkflowState) -> WorkflowState:
    """Normalize：去重 + 字段兜底 + 标注。"""
    seen: set[str] = set()
    out: list[Item] = []
    for it in state.raw_items:
        key = (it.source_type, it.id or (it.title + it.content))
        if key in seen:
            state.record(f"normalize: deduplicated {it.id}")
            continue
        seen.add(key)
        # 兜底空字段
        if not it.product_category:
            it.product_category = "unknown"
        if not it.title:
            it.title = it.content[:20]
        out.append(it)
    state.normalized = out
    state.record(f"normalize: {len(out)} items after dedup")
    return state


def node_index_retrieve(state: WorkflowState, query: str, top_k: int = 5) -> WorkflowState:
    """Index & Retrieve：与 Agent 主链路统一使用 Hybrid RAG。

    - 多查询：query 原句 + 简单关键词切分
    - 引擎：LlamaIndex retrieval
    - 排序：Reciprocal Rank Fusion
    - 去重：document-level
    """
    from rag.hybrid_retrieval import hybrid_search

    base_query = query.strip() or "consumer electronics"
    queries = [base_query]
    # 简单扩展：按空格拆分后的关键词组合
    tokens = [x for x in base_query.replace("，", " ").replace(",", " ").split() if x]
    if tokens:
        queries.extend(tokens[:3])

    result = hybrid_search(
        query=base_query,
        queries=queries,
        top_k=top_k,
        db_path="data/rag/techcompass.db",
    )
    state.retrieved.extend(result.get("results", []))
    state.record(
        f"index_retrieve: hybrid RAG returned {result.get('count', 0)} documents "
        f"for {result.get('queries', [])}"
    )

    # 兼容本地 Item 输入
    for it in state.normalized:
        state.retrieved.append(it.to_dict())
    return state

def node_analyze(state: WorkflowState, analysis_backend: str = "rule", query: str = "") -> WorkflowState:
    """Analyze：支持 rule / llm 两种分析后端。

    - rule：统一中英文风险/情绪词表，离线可复现
    - llm：调用 analyze_consumer_sentiment，开放域模型判断
    """
    from analysis.risk_lexicon import detect_risk, detect_sentiment

    topics: dict[str, int] = {}
    negative_items: list[dict[str, Any]] = []
    risk_items: list[dict[str, Any]] = []

    for it in state.retrieved:
        content = str(it.get("content", "")) or str(it.get("text", ""))
        sentiment_result = detect_sentiment(content)
        risk_result = detect_risk(content)
        if sentiment_result["sentiment"] == "negative":
            negative_items.append(it)
        if risk_result["risk_level"] in ("medium", "high"):
            risk_items.append(it)
        topic = str(it.get("title", "other"))
        topics[topic] = topics.get(topic, 0) + 1

    if analysis_backend == "llm":
        from tools.llm_analysis_tools import make_analyze_consumer_sentiment

        evidence = [
            f"{it.get('title', '')}. {it.get('content', '') or it.get('text', '')}"
            for it in state.retrieved
        ]
        tool = make_analyze_consumer_sentiment()
        model_analysis = tool.run({"question": query or "consumer electronics", "evidence": evidence})
        state.analysis = {
            "mode": "llm",
            "model_analysis": model_analysis,
            "topics": topics,
            "negative_count": 1 if model_analysis.get("sentiment") == "negative" else 0,
            "negative_items": negative_items,
            "risk_count": 1 if model_analysis.get("risk_level") in ("medium", "high", "critical") else 0,
            "risk_items": risk_items,
        }
        state.record(
            f"analyze: llm mode, sentiment={model_analysis.get('sentiment')}, "
            f"risk={model_analysis.get('risk_level')}, impact={model_analysis.get('impact_level')}"
        )
        return state

    state.analysis = {
        "mode": "unified-rule-based",
        "topics": topics,
        "negative_count": len(negative_items),
        "negative_items": negative_items,
        "risk_count": len(risk_items),
        "risk_items": risk_items,
    }
    state.record(
        f"analyze: {len(negative_items)} negative, {len(risk_items)} risk items, topics={topics}"
    )
    return state

def node_reason(state: WorkflowState) -> WorkflowState:
    """Reason：优先使用模型分析；规则结果仅作为 fallback。"""
    model = state.analysis.get("model_analysis", {})
    if model:
        priority = model.get("impact_level", "low")
        state.recommendation = {
            "priority": priority,
            "customer_support": model.get("customer_support_action", ""),
            "product": model.get("product_action", ""),
            "communication": model.get("communication_action", ""),
            "root_cause": model.get("root_cause", ""),
            "confidence": model.get("confidence", 0),
        }
        state.record(f"reason: model priority={priority}")
        return state

    neg_count = state.analysis.get("negative_count", 0)
    risk_count = state.analysis.get("risk_count", 0)
    priority = "high" if risk_count > 0 or neg_count >= 2 else "medium" if neg_count else "low"
    state.recommendation = {
        "priority": priority,
        "customer_support": "优先联系负面反馈用户，记录到售后跟进队列" if neg_count else "暂无明显风险，保持常规跟进",
        "product": "进入产品改进候选池" if risk_count else "无需产品动作",
    }
    state.record(f"reason: rule priority={priority}")
    return state

def node_output(state: WorkflowState) -> WorkflowState:
    """Output：生成结构化摘要结果。"""
    state.output = {
        "summary": {
            "total_reviewed": len(state.retrieved),
            "negative_count": state.analysis.get("negative_count", 0),
            "topics": state.analysis.get("topics", {}),
        },
        "recommendation": state.recommendation,
        "priority": state.recommendation.get("priority"),
    }
    state.record("output: digest produced")
    return state


def node_archive(state: WorkflowState, path: str) -> WorkflowState:
    """Archive：把结果写回历史库（Week 2 用 JSON 落盘）。"""
    data = {
        "log": state.log,
        "errors": state.errors,
        "output": state.output,
    }
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    state.record(f"archive: saved to {path}")
    return state


# 舆情风险词表：Week 2 用规则命中负面信号，Week 3 换 LLM 主题/情绪分类
RISK_WORDS = ("发热", "烫", "安全", "隐患", "过热", "烧", "松动", "断充",
              "换货", "退货", "爆炸", "漏电", "过敏", "故障", "问题", "担心")


def keyword_filter(items: list[Item], query: str) -> list[Item]:
    """确定性风险词命中：只要条目文本含任一风险词就命中。

    不依赖 embedding 与分词，Week 2 用于保证负面舆情不被随机检索漏掉。
    返回命中列表；若 query 里显式提到某个词，优先用该词之外再叠加风险词表。
    """
    words = set(RISK_WORDS)
    # 中文 query 没有空格，直接把 query 里命中的风险词也加入（覆盖更准）
    for w in RISK_WORDS:
        if w in query:
            words.add(w)

    hits: list[Item] = []
    seen_ids: set[str] = set()
    for it in items:
        text = it.title + it.content
        if any(w in text for w in words):
            key = it.id or (it.title + it.content)
            if key not in seen_ids:
                seen_ids.add(key)
                hits.append(it)
    return hits

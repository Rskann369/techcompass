"""TechCompass — 自动生成可复现评测集。

从 SQLite 真实 RAG documents 生成：
1. RAG 检索评测集
2. Agent Pipeline 评测集

规则：
- 按 title+content 去重，避免同一新闻多条 copy 污染指标。
- 每个 query 的相关文档集合来自明确主题词。
- Pipeline expected 标签来自确定性实体/品类规则，可追溯。
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from rag.store import RagStore

RAG_OUTPUT = Path("data/evalsets/generated_rag_queries.jsonl")
PIPELINE_OUTPUT = Path("data/evalsets/generated_pipeline_cases.jsonl")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def extract_keywords(title: str) -> list[str]:
    """从标题提取简洁关键词，过滤停用词和过长词。"""
    stop = {
        "the", "a", "an", "and", "or", "for", "with", "of", "to", "in", "on", "at",
        "is", "are", "was", "were", "new", "best", "how", "why", "what", "this",
        "that", "these", "those", "you", "your", "it", "its", "from", "by", "as",
    }
    words = re.findall(r"[A-Za-z]{3,}|\d+", title.lower())
    return [w for w in words if w not in stop and w not in {"2026", "2025"}]


def main() -> None:
    store = RagStore()
    rows = store.conn.execute(
        "SELECT id, source, source_type, title, content FROM documents ORDER BY id"
    ).fetchall()

    # 去重文档：同一 title+content 视为同一文档
    unique_docs: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = normalize_text(row["title"] + row["content"])
        if key not in unique_docs:
            unique_docs[key] = dict(row)

    docs = list(unique_docs.values())

    # 主题桶：基于标题关键词聚类，简单但可解释
    topic_patterns = {
        "wireless earbuds": re.compile(r"earbud|headphone|airpods|buds", re.I),
        "smartwatch battery": re.compile(r"smartwatch|watch|battery|garmin|fitbit", re.I),
        "consumer electronics safety": re.compile(r"counterfeit|smuggling|safety|hazard", re.I),
        "gaming mouse": re.compile(r"mouse|gaming", re.I),
        "noise cancellation": re.compile(r"noise|anc|cancellation", re.I),
        "solar charging": re.compile(r"solar|charging", re.I),
        "product review": re.compile(r"review|favourite|recommend", re.I),
    }

    rag_cases: list[dict[str, Any]] = []
    pipeline_cases: list[dict[str, Any]] = []

    for topic, pattern in topic_patterns.items():
        relevant = [d for d in docs if pattern.search(d["title"] + " " + d["content"])]
        if not relevant:
            continue

        titles = [d["title"] for d in relevant]
        rag_cases.append({
            "query": topic,
            "relevant_titles": titles,
        })

        # Pipeline case：期望值由主题规则推导
        category = "unknown"
        entities = []
        if re.search(r"earbud|headphone|airpods|buds", topic, re.I):
            category = "headphones"
            entities = ["earbuds"]
        elif re.search(r"smartwatch|watch|fitbit|garmin", topic, re.I):
            category = "smartwatch"
            entities = ["smartwatch"]
        elif re.search(r"mouse|gaming", topic, re.I):
            category = "accessory"
            entities = ["mouse"]
        elif re.search(r"battery|charging|solar", topic, re.I):
            category = "battery"
            entities = ["battery"]

        expected_risk = "pos" if re.search(r"safety|hazard|counterfeit|smuggling", topic, re.I) else "neg"
        expected_impact = "medium" if expected_risk == "pos" else "low"

        pipeline_cases.append({
            "question": topic,
            "expected": {
                "category": category,
                "sentiment": "neutral",
                "impact_level": expected_impact,
                "entities": entities,
                "risk": expected_risk,
            },
        })

    # 扩展到更多 query：每个文档标题生成一个 query，用于增强覆盖率
    for doc in docs:
        keywords = extract_keywords(doc["title"])
        if not keywords:
            continue
        query = " ".join(keywords[:4])
        rag_cases.append({
            "query": query,
            "relevant_titles": [doc["title"]],
        })
        pipeline_cases.append({
            "question": query,
            "expected": {
                "category": "unknown",
                "sentiment": "neutral",
                "impact_level": "low",
                "entities": [],
                "risk": "neg",
            },
        })

    # 写文件
    RAG_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with RAG_OUTPUT.open("w", encoding="utf-8") as f:
        for case in rag_cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    with PIPELINE_OUTPUT.open("w", encoding="utf-8") as f:
        for case in pipeline_cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    print(f"unique documents: {len(docs)}")
    print(f"generated rag cases: {len(rag_cases)} -> {RAG_OUTPUT}")
    print(f"generated pipeline cases: {len(pipeline_cases)} -> {PIPELINE_OUTPUT}")
    store.close()


if __name__ == "__main__":
    main()

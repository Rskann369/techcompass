"""TechCompass — RAG 检索质量评分器。

适用对象：实时公开数据源（GNews / Reddit RSS）的召回结果。

评分维度：
- coverage        ：查询关键词在召回文本中的覆盖比例
- title_match     ：标题是否包含关键词
- content_match   ：正文是否包含关键词
- freshness       ：发布时间新鲜度
- completeness    : title/content/url/published_at 是否齐全
- source_diversity：是否覆盖多个来源/域名
- result_count    ：是否返回足够候选
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from adapters.realtime_base import RealtimeItem


def _normalize_text(text: str) -> str:
    return (text or "").lower()


def _query_terms(query: str) -> list[str]:
    terms = [t for t in query.replace("，", " ").replace(",", " ").split() if t]
    # 针对英文复合词做简单 bigram，避免 "wireless earbuds" 只能整串命中
    if len(terms) >= 2:
        terms.append(" ".join(terms[:2]))
    return terms


def evaluate_item(query: str, item: RealtimeItem, now: datetime | None = None) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    terms = _query_terms(query)
    title = _normalize_text(item.title)
    content = _normalize_text(item.content)

    matched = [t for t in terms if t.lower() in title or t.lower() in content]
    coverage = len(matched) / len(terms) if terms else 0.0
    title_match = any(t.lower() in title for t in terms)
    content_match = any(t.lower() in content for t in terms)

    freshness = 0.0
    if item.published_at:
        try:
            age_days = max((now - item.published_at).days, 0)
            if age_days <= 7:
                freshness = 1.0
            elif age_days <= 30:
                freshness = 0.5
            else:
                freshness = 0.0
        except Exception:
            freshness = 0.0

    completeness_fields = {
        "title": bool(item.title),
        "content": bool(item.content),
        "url": bool(item.url),
        "published_at": bool(item.published_at),
    }
    completeness = sum(completeness_fields.values()) / 4

    overall = round(
        coverage * 0.35
        + (1.0 if title_match else 0.0) * 0.10
        + (1.0 if content_match else 0.0) * 0.05
        + freshness * 0.20
        + completeness * 0.20,
        4,
    )

    return {
        "id": item.id,
        "source": item.source,
        "title": item.title,
        "published_at": item.published_at.isoformat() if item.published_at else None,
        "matched_terms": matched,
        "coverage": round(coverage, 4),
        "title_match": title_match,
        "content_match": content_match,
        "freshness": freshness,
        "completeness": round(completeness, 4),
        "overall_score": overall,
    }


def evaluate_rag_results(
    query: str,
    items: list[RealtimeItem],
    now: datetime | None = None,
) -> dict[str, Any]:
    if not items:
        return {
            "query": query,
            "count": 0,
            "avg_coverage": 0.0,
            "avg_freshness": 0.0,
            "avg_completeness": 0.0,
            "source_diversity": 0.0,
            "result_count_score": 0.0,
            "overall_score": 0.0,
            "details": [],
        }

    details = [evaluate_item(query, it, now=now) for it in items]
    n = len(details)
    avg_coverage = sum(d["coverage"] for d in details) / n
    avg_freshness = sum(d["freshness"] for d in details) / n
    avg_completeness = sum(d["completeness"] for d in details) / n

    # 来源多样性：不同 source_name / subreddit / domain
    source_names = set()
    for it in items:
        meta = it.metadata or {}
        source_names.add(meta.get("source_name") or meta.get("subreddit") or it.source)
    diversity = len(source_names) / max(n, 1)
    diversity = min(diversity, 1.0)

    result_count_score = 1.0 if n >= 5 else n / 5

    overall = round(
        avg_coverage * 0.35
        + avg_freshness * 0.20
        + avg_completeness * 0.20
        + diversity * 0.10
        + result_count_score * 0.15,
        4,
    )

    return {
        "query": query,
        "count": n,
        "avg_coverage": round(avg_coverage, 4),
        "avg_freshness": round(avg_freshness, 4),
        "avg_completeness": round(avg_completeness, 4),
        "source_diversity": round(diversity, 4),
        "result_count_score": round(result_count_score, 4),
        "overall_score": overall,
        "details": details,
    }

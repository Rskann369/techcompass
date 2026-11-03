"""TechCompass — 语义重排序。

对多源聚合后的候选样本做「query 相关语义排序」：
- 用本地 BGE 模型对 query 与每个样本（title+content）做 embedding
- 余弦相似度打分（向量已归一化，dot == cosine）
- 按相似度降序、过滤低于阈值的样本、取 top_k
- 结果带上 _semantic_similarity，供后续分析与结论引用

这是优化点二里「样本过多时先用 agent 做语义分析和 query 相关排序，
再对高排名样本做分析」的确定性实现；LLM 判断可作为后续增强。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .semantic_embedding import SemanticEmbedder


def _doc_text(doc: dict[str, Any]) -> str:
    title = str(doc.get("title") or "").strip()
    content = str(doc.get("content") or doc.get("text") or "").strip()
    return f"{title}. {content}" if title and content else (title or content)


class SemanticReranker:
    def __init__(self, embedder: SemanticEmbedder | None = None):
        self.embedder = embedder or SemanticEmbedder()

    def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> list[dict[str, Any]]:
        if not documents:
            return []
        texts = [_doc_text(d) for d in documents]
        query_vec = self.embedder.embed(query)
        doc_vecs = self.embedder.embed_batch(texts)
        ranked: list[dict[str, Any]] = []
        for doc, vec in zip(documents, doc_vecs):
            sim = float(np.dot(query_vec, vec))
            out = dict(doc)
            out["_semantic_similarity"] = round(sim, 6)
            ranked.append(out)
        ranked.sort(key=lambda d: d["_semantic_similarity"], reverse=True)
        filtered = [d for d in ranked if d["_semantic_similarity"] >= threshold]
        return filtered[:top_k]

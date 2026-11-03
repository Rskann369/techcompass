"""TechCompass — LlamaIndex 语义检索层。

由于当前内网网关不支持 /v1/embeddings，本模块提供：
- TechCompassEmbedding：继承 LlamaIndex BaseEmbedding，当前使用确定性 Hash Embedding。
- build_llamaindex_index：从 SQLite documents/chunks 构建 VectorStoreIndex。
- llamaindex_search：通过 LlamaIndex Retriever 检索。
- EmbeddingBackend 抽象：未来可无缝替换为 OpenAI/本地语义模型。

意义：让 LlamaIndex 真正参与主 RAG 链路，而不只是历史 Demo。
"""

from __future__ import annotations

from typing import Any, Sequence

from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.retrievers import VectorIndexRetriever

from rag.store import HashEmbedder, RagStore


class TechCompassEmbedding(BaseEmbedding):
    """LlamaIndex 兼容 embedding。

    backend:
    - semantic：BAAI/bge-small-en-v1.5，本地真实语义模型
    - hash：HashEmbedder，离线确定性基线

    默认使用 semantic，便于 RAG 主链路获得真实语义检索能力。
    """

    model_name: str = "BAAI/bge-small-en-v1.5"
    dimensions: int = 384
    backend: str = "semantic"

    @classmethod
    def class_name(cls) -> str:
        return "TechCompassEmbedding"

    def __init__(self, backend: str = "semantic", **kwargs):
        kwargs["backend"] = backend
        super().__init__(**kwargs)
        if backend == "semantic":
            from rag.semantic_embedding import SemanticEmbedder
            self._embedder = SemanticEmbedder()
        elif backend == "hash":
            self._embedder = HashEmbedder()
            self.model_name = HashEmbedder.model_name
            self.dimensions = HashEmbedder.dim
        else:
            raise ValueError(f"unknown embedding backend: {backend}")

    def _embed(self, text: str) -> list[float]:
        vector = self._embedder.embed(text)
        return vector.tolist()

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._embed(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._embed(text)

    def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)

    def _aget_text_embedding(self, text: str) -> list[float]:
        return self._get_text_embedding(text)


def build_llamaindex_index(db_path: str = "data/rag/techcompass.db") -> VectorStoreIndex:
    """从 SQLite RAG 构建 LlamaIndex VectorStoreIndex。"""
    store = RagStore(db_path)
    try:
        rows = store.conn.execute(
            """
            SELECT c.id AS chunk_id, c.text, d.source, d.source_type, d.title, d.url,
                   d.published_at, d.language
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            """
        ).fetchall()
    finally:
        store.close()

    documents = []
    for row in rows:
        documents.append(Document(
            text=row["text"],
            metadata={
                "chunk_id": row["chunk_id"],
                "source": row["source"],
                "source_type": row["source_type"],
                "title": row["title"],
                "url": row["url"],
                "published_at": row["published_at"],
                "language": row["language"],
            },
        ))

    embed_model = TechCompassEmbedding(backend="semantic")
    Settings.embed_model = embed_model
    Settings.llm = None
    return VectorStoreIndex.from_documents(documents, embed_model=embed_model)


def llamaindex_search(index: VectorStoreIndex, query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """使用 LlamaIndex retriever 检索。"""
    retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k)
    nodes = retriever.retrieve(query)
    results = []
    for scored_node in nodes:
        node = scored_node.node
        metadata = node.metadata or {}
        results.append({
            "chunk_id": metadata.get("chunk_id"),
            "document_id": None,
            "text": node.get_content(),
            "score": scored_node.score or 0.0,
            "title": metadata.get("title"),
            "source": metadata.get("source"),
            "source_type": metadata.get("source_type"),
            "url": metadata.get("url"),
            "published_at": metadata.get("published_at"),
        })
    return results

"""TechCompass — SQLite RAG 数据层。

目标：建立真实可用的本地 RAG 语料库，而不是 MockEmbedding 演示。

设计：
- documents 保存原始实时数据（GNews/Reddit）
- chunks 保存切分文本
- embeddings 保存向量（当前使用确定性 Hash Embedding，可替换）
- retrievals 保存每次检索命中记录
- runs 保存每次采集/构建/检索运行状态
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_DB = "data/rag/techcompass.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    kind TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 1,
    config TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    external_id TEXT NOT NULL,
    source_type TEXT NOT NULL DEFAULT 'unknown',
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    url TEXT,
    published_at TEXT,
    language TEXT,
    metadata TEXT NOT NULL DEFAULT '{}',
    content_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    ingested_at TEXT,
    UNIQUE(source, external_id)
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    token_estimate INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(document_id) REFERENCES documents(id),
    UNIQUE(document_id, chunk_index)
);
CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id INTEGER PRIMARY KEY,
    dim INTEGER NOT NULL,
    vector BLOB NOT NULL,
    model TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(chunk_id) REFERENCES chunks(id)
);
CREATE TABLE IF NOT EXISTS retrievals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT NOT NULL,
    top_k INTEGER NOT NULL,
    chunk_ids TEXT NOT NULL,
    scores TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_type TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    metrics TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS agent_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pipeline TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT,
    status TEXT NOT NULL,
    report_path TEXT,
    alert_path TEXT,
    archive_path TEXT,
    metrics TEXT NOT NULL DEFAULT '{}',
    started_at TEXT NOT NULL,
    finished_at TEXT
);

CREATE TABLE IF NOT EXISTS agent_tool_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    seq INTEGER NOT NULL,
    call_type TEXT NOT NULL,
    tool_name TEXT,
    kind TEXT,
    status TEXT NOT NULL,
    args TEXT,
    result_summary TEXT,
    elapsed_ms REAL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(run_id) REFERENCES agent_runs(id)
);

CREATE INDEX IF NOT EXISTS idx_agent_tool_calls_run ON agent_tool_calls(run_id);
CREATE TABLE IF NOT EXISTS workflow_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workflow_name TEXT NOT NULL,
    status TEXT NOT NULL,
    query TEXT,
    checkpoint TEXT NOT NULL DEFAULT '{}',
    metrics TEXT NOT NULL DEFAULT '{}',
    started_at TEXT NOT NULL,
    finished_at TEXT
);
CREATE TABLE IF NOT EXISTS workflow_node_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workflow_run_id INTEGER NOT NULL,
    node_name TEXT NOT NULL,
    status TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    elapsed_ms REAL,
    error TEXT,
    checkpoint TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(workflow_run_id) REFERENCES workflow_runs(id)
);
CREATE INDEX IF NOT EXISTS idx_workflow_node_runs_run ON workflow_node_runs(workflow_run_id);
CREATE INDEX IF NOT EXISTS idx_agent_tool_calls_run ON agent_tool_calls(run_id);

CREATE TABLE IF NOT EXISTS llm_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_run_id INTEGER,
    event_type TEXT NOT NULL,
    run_uuid TEXT,
    tool_name TEXT,
    prompt TEXT,
    output TEXT,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    elapsed_ms REAL,
    error TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(agent_run_id) REFERENCES agent_runs(id)
);
CREATE INDEX IF NOT EXISTS idx_llm_events_agent_run ON llm_events(agent_run_id);

CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class HashEmbedder:
    model_name = "hash-256-v1"
    dim = 256

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        normalized = " ".join((text or "").lower().split())
        if not normalized:
            return vec
        for n in (1, 2, 3):
            for i in range(0, len(normalized) - n + 1):
                gram = normalized[i:i+n]
                idx = int(hashlib.md5(gram.encode("utf-8")).hexdigest(), 16) % self.dim
                sign = 1 if int(hashlib.md5((gram + "s").encode()).hexdigest(), 16) % 2 == 0 else -1
                vec[idx] += sign
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec


@dataclass
class RetrievedChunk:
    chunk_id: int
    document_id: int
    text: str
    score: float
    title: str
    source: str
    url: str | None
    published_at: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "text": self.text,
            "score": self.score,
            "title": self.title,
            "source": self.source,
            "url": self.url,
            "published_at": self.published_at,
        }


def chunk_text(text: str, chunk_size: int = 600, overlap: int = 80) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    step = max(chunk_size - overlap, 1)
    for i in range(0, len(text), step):
        part = text[i:i+chunk_size].strip()
        if part:
            chunks.append(part)
        if i + chunk_size >= len(text):
            break
    return chunks


class RagStore:
    def __init__(self, db_path: str = DEFAULT_DB):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self._migrate()
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def _migrate(self) -> None:
        """为旧库补充新增列，保持向后兼容。"""
        columns = {row["name"] for row in self.conn.execute("PRAGMA table_info(documents)")}
        additions = {
            "source_type": "TEXT NOT NULL DEFAULT 'unknown'",
            "language": "TEXT",
            "ingested_at": "TEXT",
        }
        for column, definition in additions.items():
            if column not in columns:
                self.conn.execute(f"ALTER TABLE documents ADD COLUMN {column} {definition}")

    def upsert_document(self, item: dict[str, Any]) -> int | None:
        """幂等写入统一 TechCompassDocument。

        兼容旧 RealtimeItem dict，但会先归一化为统一 schema。
        """
        from adapters.unified_schema import document_from_dict

        doc = document_from_dict(item)
        external_id = doc.raw.get("id") or item.get("id") or item.get("url") or item.get("title")
        if not external_id:
            return None

        cur = self.conn.execute(
            """
            INSERT INTO documents(source, external_id, source_type, title, content, url,
                                  published_at, language, metadata, content_hash,
                                  created_at, ingested_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source, external_id) DO UPDATE SET
                source_type=excluded.source_type,
                title=excluded.title,
                content=excluded.content,
                url=excluded.url,
                published_at=excluded.published_at,
                language=excluded.language,
                metadata=excluded.metadata,
                content_hash=excluded.content_hash,
                ingested_at=excluded.ingested_at
            """,
            (doc.source, str(external_id), doc.source_type, doc.title, doc.content,
             doc.url or "", doc.published_at.isoformat() if doc.published_at else None,
             doc.language, json.dumps(doc.metadata, ensure_ascii=False), doc.content_hash,
             utcnow(), doc.ingested_at.isoformat()),
        )
        self.conn.commit()
        return cur.lastrowid

    def replace_chunks(self, document_id: int, chunks: list[str]) -> None:
        # 先删除旧 embeddings，避免 chunks 被替换后留下孤儿向量
        self.conn.execute(
            "DELETE FROM embeddings WHERE chunk_id IN (SELECT id FROM chunks WHERE document_id=?)",
            (document_id,),
        )
        self.conn.execute("DELETE FROM chunks WHERE document_id=?", (document_id,))
        self.conn.commit()
        embedder = HashEmbedder()
        for i, text in enumerate(chunks):
            cur = self.conn.execute(
                "INSERT INTO chunks(document_id, chunk_index, text, token_estimate, created_at) VALUES (?, ?, ?, ?, ?)",
                (document_id, i, text, max(1, len(text) // 2), utcnow()),
            )
            chunk_id = cur.lastrowid
            vec = embedder.embed(text)
            self.conn.execute(
                "INSERT INTO embeddings(chunk_id, dim, vector, model, created_at) VALUES (?, ?, ?, ?, ?)",
                (chunk_id, len(vec), vec.tobytes(), HashEmbedder.model_name, utcnow()),
            )
        self.conn.commit()

    def stats(self) -> dict[str, int]:
        def count(sql: str) -> int:
            return self.conn.execute(sql).fetchone()[0]
        return {
            "documents": count("SELECT COUNT(*) FROM documents"),
            "chunks": count("SELECT COUNT(*) FROM chunks"),
            "embeddings": count("SELECT COUNT(*) FROM embeddings"),
            "retrievals": count("SELECT COUNT(*) FROM retrievals"),
            "runs": count("SELECT COUNT(*) FROM runs"),
            "agent_runs": count("SELECT COUNT(*) FROM agent_runs"),
            "agent_tool_calls": count("SELECT COUNT(*) FROM agent_tool_calls"),
            "workflow_runs": count("SELECT COUNT(*) FROM workflow_runs"),
            "workflow_node_runs": count("SELECT COUNT(*) FROM workflow_node_runs"),
            "llm_events": count("SELECT COUNT(*) FROM llm_events"),
        }

    def start_agent_run(self, pipeline: str, question: str) -> int:
        cur = self.conn.execute(
            "INSERT INTO agent_runs(pipeline, question, status, started_at) VALUES (?, ?, 'running', ?)",
            (pipeline, question, utcnow()),
        )
        self.conn.commit()
        return cur.lastrowid

    def finish_agent_run(
        self,
        run_id: int,
        status: str,
        answer: str,
        metrics: dict[str, Any],
        report_path: str | None = None,
        alert_path: str | None = None,
        archive_path: str | None = None,
    ) -> None:
        self.conn.execute(
            """
            UPDATE agent_runs
            SET status=?, answer=?, report_path=?, alert_path=?, archive_path=?,
                metrics=?, finished_at=?
            WHERE id=?
            """,
            (status, answer, report_path, alert_path, archive_path,
             json.dumps(metrics, ensure_ascii=False), utcnow(), run_id),
        )
        self.conn.commit()

    def insert_agent_trace(self, run_id: int, trace: list[dict[str, Any]]) -> None:
        for seq, item in enumerate(trace, start=1):
            call_type = str(item.get("type", "unknown"))
            tool_name = item.get("tool")
            kind = item.get("kind")
            status = str(item.get("status", "unknown"))
            args = item.get("args")
            result_summary = item.get("result") or item.get("output") or item.get("answer")
            self.conn.execute(
                """
                INSERT INTO agent_tool_calls
                (run_id, seq, call_type, tool_name, kind, status, args, result_summary,
                 elapsed_ms, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id, seq, call_type,
                    tool_name, kind, status,
                    json.dumps(args, ensure_ascii=False) if args is not None else None,
                    str(result_summary)[:2000] if result_summary is not None else None,
                    item.get("elapsed_ms"),
                    utcnow(),
                ),
            )
        self.conn.commit()

    def insert_llm_events(self, agent_run_id: int, events: list[dict[str, Any]]) -> None:
        """Persist LangChain callback events for an agent run."""
        for event in events:
            self.conn.execute(
                """
                INSERT INTO llm_events(
                    agent_run_id, event_type, run_uuid, tool_name,
                    prompt, output, prompt_tokens, completion_tokens, total_tokens,
                    elapsed_ms, error, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    agent_run_id,
                    str(event.get("type", "unknown")),
                    str(event.get("run_id", "")) if event.get("run_id") is not None else None,
                    event.get("tool"),
                    json.dumps(event.get("prompts"), ensure_ascii=False) if event.get("prompts") is not None else None,
                    json.dumps(event.get("output"), ensure_ascii=False) if event.get("output") is not None else None,
                    event.get("prompt_tokens"),
                    event.get("completion_tokens"),
                    event.get("total_tokens"),
                    event.get("elapsed_ms"),
                    event.get("error"),
                    utcnow(),
                ),
            )
        self.conn.commit()

    def list_llm_events(self, agent_run_id: int) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM llm_events WHERE agent_run_id=? ORDER BY id", (agent_run_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def list_agent_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM agent_runs ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_agent_run_trace(self, run_id: int) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM agent_tool_calls WHERE run_id=? ORDER BY seq", (run_id,)
        ).fetchall()
        return [dict(r) for r in rows]


    def start_workflow_run(self, workflow_name: str, query: str | None = None) -> int:
        cur = self.conn.execute(
            "INSERT INTO workflow_runs(workflow_name, status, query, started_at) VALUES (?, 'running', ?, ?)",
            (workflow_name, query, utcnow()),
        )
        self.conn.commit()
        return cur.lastrowid

    def checkpoint_workflow_run(self, run_id: int, checkpoint: dict[str, Any], metrics: dict[str, Any] | None = None) -> None:
        self.conn.execute(
            "UPDATE workflow_runs SET checkpoint=?, metrics=COALESCE(?, metrics) WHERE id=?",
            (json.dumps(checkpoint, ensure_ascii=False), json.dumps(metrics, ensure_ascii=False) if metrics else None, run_id),
        )
        self.conn.commit()

    def finish_workflow_run(self, run_id: int, status: str, checkpoint: dict[str, Any], metrics: dict[str, Any]) -> None:
        self.conn.execute(
            "UPDATE workflow_runs SET status=?, checkpoint=?, metrics=?, finished_at=? WHERE id=?",
            (status, json.dumps(checkpoint, ensure_ascii=False), json.dumps(metrics, ensure_ascii=False), utcnow(), run_id),
        )
        self.conn.commit()

    def record_workflow_node(self, run_id: int, node_name: str, status: str, attempts: int, elapsed_ms: float | None, error: str | None, checkpoint: dict[str, Any]) -> None:
        self.conn.execute(
            "INSERT INTO workflow_node_runs(workflow_run_id,node_name,status,attempts,elapsed_ms,error,checkpoint,created_at) VALUES (?,?,?,?,?,?,?,?)",
            (run_id, node_name, status, attempts, elapsed_ms, error, json.dumps(checkpoint, ensure_ascii=False), utcnow()),
        )
        self.conn.commit()

    def get_workflow_run(self, run_id: int) -> dict[str, Any]:
        row = self.conn.execute("SELECT * FROM workflow_runs WHERE id=?", (run_id,)).fetchone()
        return dict(row) if row else {}

    def list_workflow_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM workflow_runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def list_workflow_nodes(self, run_id: int) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM workflow_node_runs WHERE workflow_run_id=? ORDER BY id", (run_id,)).fetchall()
        return [dict(r) for r in rows]

    def start_run(self, run_type: str) -> int:
        cur = self.conn.execute(
            "INSERT INTO runs(run_type, status, started_at) VALUES (?, 'running', ?)",
            (run_type, utcnow()),
        )
        self.conn.commit()
        return cur.lastrowid

    def finish_run(self, run_id: int, status: str, metrics: dict[str, Any]) -> None:
        self.conn.execute(
            "UPDATE runs SET status=?, finished_at=?, metrics=? WHERE id=?",
            (status, utcnow(), json.dumps(metrics, ensure_ascii=False), run_id),
        )
        self.conn.commit()


def search_store(store: RagStore, query: str, top_k: int = 5) -> list[RetrievedChunk]:
    if top_k <= 0:
        return []
    qvec = HashEmbedder().embed(query)
    rows = store.conn.execute(
        """
        SELECT e.chunk_id, e.vector, c.text, c.document_id,
               d.title, d.source, d.url, d.published_at
        FROM embeddings e
        JOIN chunks c ON c.id=e.chunk_id
        JOIN documents d ON d.id=c.document_id
        """
    ).fetchall()

    scored: list[RetrievedChunk] = []
    for row in rows:
        vec = np.frombuffer(row["vector"], dtype=np.float32)
        denom = np.linalg.norm(qvec) * np.linalg.norm(vec)
        score = float(np.dot(qvec, vec) / denom) if denom else 0.0
        scored.append(RetrievedChunk(
            chunk_id=row["chunk_id"], document_id=row["document_id"],
            text=row["text"], score=score, title=row["title"],
            source=row["source"], url=row["url"], published_at=row["published_at"],
        ))

    scored.sort(key=lambda x: x.score, reverse=True)
    results = scored[:top_k]
    if results:
        store.conn.execute(
            "INSERT INTO retrievals(query, top_k, chunk_ids, scores, created_at) VALUES (?, ?, ?, ?, ?)",
            (query, top_k,
             json.dumps([r.chunk_id for r in results]),
             json.dumps([r.score for r in results]), utcnow()),
        )
        store.conn.commit()
    return results

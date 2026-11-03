# TechCompass Architecture

TechCompass is a query-driven consumer-electronics sentiment agent. A user submits a natural-language question, and the system returns an evidence-grounded LLM analysis, risk and impact assessment, action recommendations, confidence, and traceable source links.

## High-level flow

```text
User query
   │
   ▼
LangGraph ReAct Agent
   │
   ├── hybrid_rag_search
   │       ├── local SQLite + LlamaIndex retrieval
   │       ├── multi-query RRF fusion
   │       └── local BGE semantic embeddings
   │
   ├── search_live_sources
   │       ├── Hacker News
   │       ├── GNews
   │       └── Reddit RSS
   │
   ├── deduplicate_docs
   ├── semantic rerank + relevance threshold
   ├── evidence gate
   │
   ├── analyze_consumer_sentiment
   │       ├── issue type
   │       ├── entities
   │       ├── sentiment
   │       ├── risk and impact
   │       ├── evidence sufficiency
   │       └── confidence
   │
   ├── hard-safety signal protection
   ├── action recommendations
   │
   ├── generate_weekly_report
   ├── create_alert
   └── archive_run
```

## Layers

### 1. Entry layer

- `scripts/run_main_pipeline.py`: main user entry.
- `scripts/build_rag.py`: builds the local RAG database.
- `scripts/fetch_multi_source.py`: demonstrates live multi-source retrieval and semantic reranking.
- `scripts/eval_node_and_e2e.py`: node-level and end-to-end evaluation.

### 2. Agent layer

- `src/agents/langgraph_react_agent.py`: LangGraph `create_react_agent` wrapper.
- `src/agents/conclusion.py`: turns structured analysis into an operator-readable conclusion.
- `src/agents/langchain_callbacks.py`: captures LLM token, latency, and event metrics.

### 3. Adapter layer

- `src/adapters/multi_source.py`: aggregates multiple realtime sources.
- `src/adapters/realtime_hackernews.py`
- `src/adapters/realtime_gnews.py`
- `src/adapters/realtime_reddit.py`
- `src/adapters/unified_schema.py`: defines `TechCompassDocument`.

### 4. Retrieval layer

- `src/rag/store.py`: SQLite documents, chunks, embeddings, runs, traces, and workflow state.
- `src/rag/hybrid_retrieval.py`: multi-query retrieval with reciprocal rank fusion.
- `src/rag/semantic_embedding.py`: local BGE embedding.
- `src/rag/semantic_rerank.py`: reranks index and live-search evidence.

### 5. Tool layer

Agent-facing tools:

```text
hybrid_rag_search
search_live_sources
rewrite_query
analyze_consumer_sentiment
deduplicate_docs
generate_weekly_report
create_alert
archive_run
```

Internal analysis tools are intentionally not exposed to the LLM:

```text
extract_entities
analyze_sentiment
score_risk
reason_impact
suggest_action
```

Management tools are also not exposed to the LLM:

```text
export_csv
query_history
agent_history
agent_trace
```

### 6. Workflow layer

`src/workflows/` provides a deterministic workflow runner for regression and reliability testing:

```text
ingest → normalize → index_retrieve → analyze → reason → output → archive
```

It supports YAML node configuration, retries, backoff, timeout, strict/continue failure policies, checkpointing, and SQLite persistence.

### 7. Evaluation layer

`src/eval/` covers:

- retrieval hit rate, recall, precision, and MRR;
- classification, sentiment, and reasoning accuracy;
- entity precision and recall;
- false positives and false negatives;
- task completion and parse success;
- tool success rate, latency, token usage, and cost.

## Persistence model

The local SQLite database stores:

- documents;
- chunks;
- embeddings;
- retrieval history;
- build runs;
- agent runs;
- tool traces;
- LangChain LLM events;
- workflow runs and node checkpoints.

The database is a runtime artifact and is not committed to Git.

## Safety and quality gates

1. **Evidence gate**: unrelated or insufficient evidence triggers refusal rather than a forced answer.
2. **Hard-safety signal gate**: safety-related terms such as fire, explosion, recall, and privacy leak cannot be silently downgraded by the LLM.
3. **Confidence**: combines evidence count, semantic similarity, and model confidence.
4. **Traceable conclusions**: every evidence item includes title, source, URL, relevance, and excerpt.
5. **Human review**: high-risk or uncertain results are marked for human review.

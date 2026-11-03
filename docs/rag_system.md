# TechCompass — RAG 系统设计与评估

## 1. 当前 RAG 架构

```
GNews / Reddit RSS
        ↓
RealtimeSource Adapter
        ↓
SQLite documents
        ↓
chunk_text
        ↓
Hash Embedding
        ↓
SQLite embeddings
        ↓
cosine similarity search
        ↓
RAG evidence
        ↓
LangChain Agent Tool Calling
        ↓
行动建议 / 周报 / 告警
```

## 2. SQLite 表

| 表 | 作用 |
| --- | --- |
| documents | 原始实时文档 |
| chunks | 文本切片 |
| embeddings | 向量 |
| retrievals | 检索记录 |
| runs | 采集/构建运行记录 |
| sources | 数据源配置（预留） |

## 3. 已接入工具

| 工具 | 作用 |
| --- | --- |
| rag_search | 本地 SQLite RAG 检索 |
| rag_stats | 查看语料库规模 |
| search_news | GNews 实时新闻 |
| search_discussions | Reddit RSS 真实讨论 |
| search_reviews | 本地评论检索（当前库已清空） |
| classify_review | 分类工具 |

## 4. 真实语料状态

首次构建结果：

- documents: 17
- chunks: 20
- embeddings: 20
- 来源：GNews 12 条，Reddit RSS 5 条

注意：当前语料规模仍偏小，用于验证 RAG 闭环；后续应扩充 query 与采集频率。

## 5. 检索评估结果

评估器：`src/eval/sqlite_rag_eval.py`
评估集：`data/evalsets/sqlite_rag_queries.jsonl`
运行：`PYTHONPATH=src ./.venv311/bin/python scripts/eval_node_and_e2e.py`

| 指标 | 结果 |
| --- | ---: |
| Hit rate | 100% |
| Recall@5 | 76% |
| Precision@5 | 64% |
| MRR@5 | 90% |
| NDCG@5 | 85.01% |

## 6. 当前限制

1. Hash Embedding 是确定性字面匹配，不具备深层语义能力。
2. 语料只有 17 条，评测集只有 5 个 query。
3. 已接入本地 BGE 语义 embedding（BAAI/bge-small-en-v1.5）。
4. 已实现 SemanticReranker 语义重排。
5. 查询重写已实现（rewrite_query）。

## 7. LlamaIndex 主链路接入（v1.1）

当前内网网关不支持 `/v1/embeddings`，因此没有使用 OpenAI embedding。

已实现：

- `TechCompassEmbedding`：继承 LlamaIndex `BaseEmbedding`
- `build_llamaindex_index`：从 SQLite documents/chunks 构建 `VectorStoreIndex`
- `llamaindex_search`：使用 `VectorIndexRetriever` 检索
- `llamaindex_search` 工具已注册进 Agent ToolRegistry
- 离线 LLM Pipeline 已切换为 LlamaIndex 检索

运行：

```bash
PYTHONPATH=src ./.venv311/bin/python scripts/run_main_pipeline.py "wireless earbuds safety"
```

评估：

```bash
PYTHONPATH=src ./.venv311/bin/python scripts/eval_llamaindex_rag.py
```

当前 LlamaIndex RAG 指标：

| 指标 | 结果 |
| --- | ---: |
| Hit rate | 100% |
| Recall@5（去重后文档级） | 69.5% |
| Precision@5 | 86.67% |
| MRR@5 | 100% |

说明：
- 当前 embedding 后端仍是 Hash Embedding，不是语义模型。
- LlamaIndex 已经真实参与主链路，满足技术栈目标中的 LlamaIndex 使用。
- 未来可替换 `TechCompassEmbedding` 内部实现为真实语义模型。

## 8. HTML 清洗与多查询融合检索（v1.1）

### HTML 清洗

新增：

```text
src/adapters/text_cleaner.py
```

能力：

- 移除 Reddit RSS 中的 HTML 标签
- HTML entity 反转义
- 空白规范化
- 检测文本是否 HTML-like

旧数据迁移：

```bash
PYTHONPATH=src ./.venv311/bin/python scripts/clean_rag_html.py
```

清洗结果：

- scanned: 35
- cleaned: 10
- remaining html-like: 0

### 多查询融合检索

新增：

```text
src/rag/hybrid_retrieval.py
```

流程：

1. 多个 rewrite queries
2. 每个查询分别走 LlamaIndex retrieval
3. document-level 去重
4. RRF 融合排序
5. 返回 top_k

公式：

```text
score(d) = Σ 1 / (60 + rank_i(d))
```

新增工具：

```text
hybrid_rag_search
```

已接入离线主链路，替代单查询 `llamaindex_search`。

### 数据一致性修复

修复 `replace_chunks` 遗留 orphan embeddings 的问题：

- 替换 chunks 前先删除旧 embeddings
- 清理脚本：`scripts/cleanup_orphan_embeddings.py`

当前一致性：

```text
chunks: 41
embeddings: 41
```

## 9. 本地语义 Embedding（v1.2）

模型：

```text
BAAI/bge-small-en-v1.5
```

本地路径：

```text
models/bge-small-en-v1.5
```

特点：

- 本地推理，不依赖外部 API
- 输出 384 维向量
- L2 normalize
- 首次下载后可完全离线运行
- SQLite embeddings 已全部重建为 semantic 向量

### 实现文件

```text
src/rag/semantic_embedding.py
src/rag/llamaindex_store.py
scripts/rebuild_semantic_embeddings.py
```

### 重建命令

```bash
PYTHONPATH=src ./.venv311/bin/python scripts/rebuild_semantic_embeddings.py
```

### 重建结果

```text
chunks: 215
model: BAAI/bge-small-en-v1.5
dim: 384
```

### 语义验证示例

```text
sim(battery swelling and fire risk, battery expansion and safety hazard) = 0.8606
sim(battery swelling and fire risk, shipping delay and packaging damage) = 0.5988
```

说明语义相近文本相似度明显高于无关文本。

### RAG 评估

使用语义 embedding 后：

| 指标 | Hash Embedding | Semantic Embedding |
| --- | ---: | ---: |
| Case 数 | 63 | 62 |
| Hit rate | 96.83% | 96.77% |
| Recall@5 | 89.79% | 90.73% |
| Precision@5 | 31.88% | 34.68% |
| MRR@5 | 81.61% | 87.50% |

说明：

- Recall@5 提升
- Precision@5 提升
- MRR@5 明显提升
- Hit rate 基本持平

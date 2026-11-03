# TechCompass — 消费品舆情推理与行动 Agent

TechCompass 面向数码消费品品牌团队，解决舆情来源分散、人工判断慢、能力边界不清、缺乏量化评估的问题。用户输入一个 query，即可得到 **LLM 推理结论、风险/影响判断、行动建议和可追溯证据依据**。

完整链路：

```text
多源采集 → 清洗/去重 → RAG 检索 → 实时补搜 → 语义重排
        → Evidence Gate → LLM 结构化分析 → 风险/影响推理
        → 行动建议 → 运营可读结论 → 周报/告警/归档
```

**Tech stack:** Python, LangChain, LangGraph `create_react_agent`, LlamaIndex, SQLite, local BGE embeddings, Hacker News / GNews / Reddit public sources.

---

## 0. Quick setup

```bash
git clone https://github.com/<your-username>/techcompass.git
cd techcompass

python3 -m venv .venv311
source ./.venv311/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

cp .env.example .env
# Edit .env and add your LLM and optional GNews credentials.
```

The repository does not include the local embedding model, RAG database, or runtime outputs. They are generated locally:

```bash
# Build the local SQLite RAG database
PYTHONPATH=src python scripts/build_rag.py

# Rebuild semantic embeddings after ingesting data
PYTHONPATH=src python scripts/rebuild_semantic_embeddings.py
```

The first semantic-embedding run downloads `BAAI/bge-small-en-v1.5` from Hugging Face unless you place the model under `models/bge-small-en-v1.5/`.

---

## 1. 使用方式：输入 query，输出结论和依据

### 推荐主入口

```bash
cd techcompass
source ./.venv311/bin/activate

PYTHONPATH=src python scripts/run_main_pipeline.py \
  "smart home camera privacy"
```

保存完整 JSON：

```bash
PYTHONPATH=src python scripts/run_main_pipeline.py \
  "wireless earbuds battery overheating" \
  --output outputs/runs/result.json
```

调整 Agent 最大步数（默认 8）：

```bash
PYTHONPATH=src python scripts/run_main_pipeline.py \
  "smart home camera privacy" \
  --max-steps 10
```

### 输入说明

输入是一个自然语言 query，建议包含三部分：

```text
产品/品类 + 问题类型 + 风险或业务问题
```

示例：

```text
smart home camera privacy
wireless earbuds battery overheating
laptop battery recall risk
游戏鼠标固件升级后失灵
蓝牙音箱续航投诉是否集中
```

可选参数：

| 参数 | 说明 |
|---|---|
| `--max-steps N` | Agent 最大推理步数，默认 8 |
| `--output PATH` | 保存完整 JSON（默认打印到 stdout） |

---

## 2. 输出结果类型

### 2.1 控制台运营可读结论

终端直接输出：

```text
舆情结论（运营可读）
结论：针对「smart home camera privacy」，基于 5 条公开舆情证据，整体情绪为负面，风险等级高，影响等级中。
正文：风险类型：privacy。存在需人工复核的安全信号...
情绪：负面 | 风险：高 | 影响：中
置信度：high（0.7）
依据（含链接）：
  [1] 证据标题
      GNews 新闻 · 相关性 高度相关（0.813）
      https://example.com/...
行动建议：
  - 客服：...
  - 产品：...
```

### 2.2 结构化 JSON

核心字段：

| 字段 | 说明 |
|---|---|
| `question` | 用户 query |
| `answer` | Agent 最终自然语言回答 |
| `conclusion` | headline、summary、证据依据（含链接）、置信度、行动建议 |
| `evidence` | 证据列表：标题、来源、URL、相似度、摘要 |
| `analysis` | LLM 结构化分析：issue_type、entities、sentiment、risk、impact、confidence、evidence_sufficient |
| `metrics` | LLM 调用、token、工具调用、延迟 |
| `run_id` | 本次运行 ID（对应 SQLite trace） |

`conclusion` 示例：

```json
{
  "headline": "针对「smart home camera privacy」，基于 5 条公开舆情证据，整体情绪为负面，风险等级高，影响等级中。",
  "summary": "风险类型：privacy。存在需人工复核的安全信号，建议优先人工确认。",
  "sentiment": "负面",
  "risk_level": "高",
  "impact_level": "中",
  "evidence_basis": [
    {
      "ref": 1,
      "title": "证据标题",
      "url": "https://example.com",
      "source_label": "GNews 新闻",
      "similarity": 0.813,
      "relevance": "高度相关",
      "excerpt": "证据摘要"
    }
  ],
  "confidence": {
    "level": "high",
    "score": 0.7,
    "evidence_count": 5,
    "average_similarity": 0.6989,
    "explanation": "证据 5 条；平均语义相似度 0.6989"
  },
  "actions": ["客服：...", "产品：..."],
  "refused": false
}
```

### 2.3 拒答输出

当证据不足或相关性不达标时，系统不强行生成结论：

```text
针对「write a poem about autumn」，当前未检索到足够可靠的公开证据，
无法给出舆情结论。
```

拒答时：

- 不生成风险结论；
- 不生成行动建议；
- 不生成周报和告警；
- 只保留 partial 归档；
- `refused = true`，置信度为 low。

### 2.4 运行产物

| 产物 | 路径 | 说明 |
|---|---|---|
| 周报 | `outputs/weekly_reports/*.md` | 运营可读报告 |
| 告警 | `outputs/alerts/*.json` | 高风险或需人工复核事件 |
| 归档 | `outputs/runs/*.json` | 单次运行结果 |
| Trace | `data/rag/techcompass.db` | run、tool call、LLM event、workflow 状态 |

---

## 3. 项目架构

```text
用户 query
   │
   ▼
Query Rewrite
   │
   ▼
Hybrid RAG Search
   │
   ├── 索引证据不足 / 相关性弱
   │        │
   │        ▼
   │   Live Multi-source Search
   │   HackerNews / GNews / Reddit
   │
   ▼
Dedup + Semantic Rerank + Evidence Gate
   │
   ├── 证据不足 → 拒答 / partial archive
   ▼
LLM Structured Analysis
(issue_type / entities / sentiment / risk / impact / confidence)
   │
   ├── 硬安全信号 → 人工复核，不允许模型降级
   ▼
Reason + Action Suggestion
   │
   ▼
ConclusionBuilder
(headline / summary / evidence_basis / confidence / actions)
   │
   ├── Markdown 周报
   ├── 高风险告警
   ├── JSON 归档
   └── SQLite trace / metrics
```

### 3.1 数据与适配层

- `src/adapters/`
- HackerNews、GNews、Reddit RSS、本地数据集；
- `TechCompassDocument` 统一 source、source_type、title、content、url、published_at、metadata、content_hash；
- `MultiSourceFetcher` 做多源聚合与 URL / 内容去重。

### 3.2 检索与排序层

- `src/rag/`
- SQLite 保存 documents、chunks、embeddings、retrievals、runs；
- LlamaIndex 构建本地索引；
- 多查询 RRF 融合；
- BGE 本地语义 embedding；
- `SemanticReranker` 对索引证据和实时证据统一排序；
- 相似度阈值默认 `0.55`，无相关证据时不再取弱相关 top-N 兜底。

### 3.3 Agent 与推理层

- `src/agents/`
- `LangGraphReactAgent`：唯一主链路 Agent，基于 LangGraph `create_react_agent`，真实 LLM 编排；
- `analyze_consumer_sentiment`：真实 LLM 开放域结构化分析（issue_type / entities / sentiment / risk / impact / confidence / evidence_sufficient）；
- `ConclusionBuilder` + `build_conclusion_from_analysis`：结构化结果 → 运营可读结论（headline / summary / 带链接依据 / 置信度）；
- `TechCompassCallbackHandler`：记录 LLM token、延迟和事件；
- Prompt 覆盖检索意图、分类、实体、原因推理、影响评估、行动建议和摘要。

### 3.4 工具层

Agent-facing tools，共 8 个：

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

Internal analysis tools，不暴露给 LLM：

```text
extract_entities
analyze_sentiment
score_risk
reason_impact
suggest_action
```

Management tools，不暴露给 LLM：

```text
export_csv
query_history
agent_history
agent_trace
```

`src/tools/registry.py` 统一支持 OpenAI function-calling schema、LangChain `StructuredTool`、参数校验和 fallback。

### 3.5 Workflow 层

- `src/workflows/`
- YAML 配置节点顺序、失败策略、重试、退避和超时；
- 节点链路：`ingest → normalize → index_retrieve → analyze → reason → output → archive`；
- 节点级 checkpoint 和 SQLite 持久化。

### 3.6 评估与观测层

- `src/eval/`
- 指标覆盖任务完成率、分类准确率、情绪准确率、推理准确率、实体 precision / recall、漏报率、误报率、解析成功率、延迟、token 成本、工具成功率；
- Agent run、tool trace、LLM event、workflow run 均落 SQLite。

---

## 4. 项目特点

1. **结论必带依据**：`evidence_basis` 包含标题、来源、链接、相似度和摘要。
2. **证据不足会拒答**：不用弱相关证据强行生成模板化结论。
3. **索引 + 实时混合检索**：本地 RAG 不足时现场补搜 HackerNews / GNews / Reddit。
4. **LLM 结构化分析**：输出 issue_type、entities、sentiment、risk、impact、confidence、evidence_sufficient。
5. **硬安全信号保护**：fire、explosion、recall、privacy leak 等信号触发人工复核，不允许模型随意降级。
6. **运营可读结论**：业务人员可直接读 headline、summary、置信度、证据和行动建议。
7. **过程可观测**：tool call、LLM event、token、延迟、run status 全部持久化。
8. **工程化基础完整**：YAML 调度、重试、超时、checkpoint、参数校验、路径白名单、fallback。

---

## 5. 其他入口

### Workflow

```bash
PYTHONPATH=src python -m workflows.runner
```

用于验证节点重试、超时、失败策略和 checkpoint。

---

## 6. 数据与 RAG 构建

```bash
# 采集真实公开数据
PYTHONPATH=src python scripts/build_rag.py

# 多源采集
PYTHONPATH=src python scripts/fetch_multi_source.py "smart home camera privacy"

# 清洗 HTML
PYTHONPATH=src python scripts/clean_rag_html.py

# 重建语义向量
PYTHONPATH=src python scripts/rebuild_semantic_embeddings.py
```

---

## 7. 测试与评估

Core tests run without private credentials, the local embedding model, or a prebuilt RAG database:

```bash
make test
```

Equivalent command:

```bash
PYTHONPATH=src ./.venv311/bin/python -m pytest tests -q -m "not local"
```

Full local integration tests require an LLM credential, the downloaded BGE model, and a built RAG database:

```bash
make test-local
```

Node-level and end-to-end evaluation:

```bash
PYTHONPATH=src python scripts/eval_node_and_e2e.py
```

GitHub Actions runs the core test suite on every push and pull request.

---

## 8. 目录结构

```text
src/
├── adapters/       多源采集、统一 Schema、清洗去重
├── agents/         主链路、LangGraph Agent、结论生成、callback
├── analysis/       规则风险与情绪基线
├── eval/           节点级 / 端到端 / RAG 评估
├── prompts/        模块化 Prompt
├── rag/            SQLite、LlamaIndex、Hybrid RAG、语义重排
├── tools/          Agent-facing / internal / management tools
└── workflows/      YAML 调度、重试、超时、checkpoint

scripts/            构建、运行、评估、多源采集入口
tests/              单元测试、集成测试、回归测试
data/               RAG 数据库与评测集
outputs/            周报、告警、运行归档
config/             Workflow YAML
docs/               架构与评估文档
```

---

## 9. 当前边界

- 项目仍是 Agent MVP，不是生产系统；
- 真实 LLM 网关的延迟和稳定性需要持续评估；
- 中文 query 与英文语料的跨语言检索质量仍有提升空间；
- 生产化还缺 API/UI、鉴权、租户隔离、部署、监控和告警通知渠道；
- The embedding model, RAG database, and runtime outputs are intentionally not committed.

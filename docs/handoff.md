# TechCompass — 交接说明

## 当前能力

- 实时采集：GNews + Reddit RSS
- SQLite RAG：documents/chunks/embeddings/retrievals/runs
- Agent 主链路：查询重写、RAG 检索、去重、实体、情绪、风险、影响、建议、周报、告警、归档
- 8 个 Agent-facing 工具（另有 internal / management 工具）
- 8 类模块化 Prompt
- Workflow 节点级重试、指数退避、失败策略
- 全链路评估与 RAG 评估

## 一键运行

```bash
source ./.venv311/bin/activate

PYTHONPATH=src python scripts/build_rag.py
PYTHONPATH=src python scripts/run_main_pipeline.py "wireless earbuds safety"
PYTHONPATH=src python scripts/eval_node_and_e2e.py
```

## 输出物

- 周报：`outputs/weekly_reports/*.md`
- 告警：`outputs/alerts/*.json`
- 运行归档：`outputs/runs/*.json`
- RAG 数据库：`data/rag/techcompass.db`

## 后续建议

1. 替换 Hash Embedding 为语义 embedding。
2. 多查询融合检索。
3. 增加真实 LLM 版最终摘要。
4. 扩充 RAG 语料和评测集。
5. 引入 token/cost 统计。

## v1.1 增量能力

- 统一跨源 Schema：TechCompassDocument
- 旧 RAG 数据迁移脚本
- 真实 LLM Tool Calling Pipeline
- 离线 LLM Pipeline（快速可复现）
- token / 成本 / 工具调用成功率指标
- 输出文件名唯一化，避免同秒覆盖

## Agent Trace 持久化

- 每次 LangGraphReactAgent 运行都会写入 SQLite。
- 可通过 `agent_history` 查询最近运行。
- 可通过 `agent_trace` 查询某次 run 的完整工具与 LLM 调用链。
- SQLite 表：`agent_runs`、`agent_tool_calls`。

## HTML 清洗与 Hybrid RAG

- Reddit RSS 输入会自动清理 HTML。
- 旧数据可用 `scripts/clean_rag_html.py` 迁移。
- 主链路使用 `hybrid_rag_search` 做多查询融合。
- RRF 按 document 去重，避免同文档多 chunk 重复。
- Agent-facing 工具：8 个。

## 扩充评测集

- 自动生成脚本：`scripts/generate_evalset.py`
- RAG 评估：`scripts/eval_generated_rag.py`
- Pipeline 评估：`scripts/eval_generated_pipeline.py`
- 当前规模：63 case
- 数据基础：70 篇真实公开数据，56 篇去重文档

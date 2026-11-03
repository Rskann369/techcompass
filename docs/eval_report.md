# TechCompass — 性能评估报告

> 注：本文为历史评估快照。当前评估入口为 `scripts/eval_node_and_e2e.py`（节点级 + 端到端，走真实 LLM）。

## 1. RAG 检索评估

命令：

```bash
PYTHONPATH=src ./.venv311/bin/python scripts/eval_node_and_e2e.py
```

结果：

| 指标 | 结果 |
| --- | ---: |
| Hit rate | 100% |
| Recall@5 | 76% |
| Precision@5 | 64% |
| MRR@5 | 90% |
| NDCG@5 | 85.01% |

## 2. Agent 主链路评估

命令：

```bash
PYTHONPATH=src ./.venv311/bin/python scripts/eval_node_and_e2e.py
```

当前评估集：5 个真实查询。

| 指标 | 结果 |
| --- | ---: |
| 任务完成率 | 100% |
| 工具调用成功率 | 100% |
| 解析成功率 | 100% |
| 平均端到端延迟 | ~1.35ms |
| 平均工具延迟 | ~0.13ms |
| 主题分类准确率 | 80% |
| 情绪准确率 | 100% |
| 推理准确率 | 80% |
| 实体 Precision | 100% |
| 实体 Recall | 20% |
| 误报率 | 20% |
| 漏报率 | 暂无正样本，无法计算 |

## 3. 调度器测试

命令：

```bash
PYTHONPATH=src ./.venv311/bin/python tests/test_scheduler.py
```

覆盖：

- 正常执行
- 节点重试后成功
- 严格失败后跳过后续节点
- continue 失败后继续执行
- allow_fail 兼容
- 指数退避时序

结果：6/6 通过。

## 4. 已知问题

1. 实体召回率低：`solar` 等实体未被当前规则抽取。
2. 风险评估存在误报：`counterfeit goods` 被判为 high，但当前 case 标注为非风险。
3. Hash Embedding 偏字面匹配，语义能力有限。
4. RAG 语料仅 17 个文档，规模偏小。
5. 主链路已切换为 LangGraph `create_react_agent` 真实 LLM 编排与结构化分析。

## 5. 离线 LLM Tool Calling Pipeline 评估

命令：

```bash
PYTHONPATH=src ./.venv311/bin/python scripts/eval_node_and_e2e.py
```

| 指标 | 结果 |
| --- | ---: |
| 任务完成率 | 100% |
| 工具调用成功率 | 100% |
| 周报生成率 | 100% |
| 平均延迟 | 1.697ms |
| LLM 调用次数 | 15 |
| Prompt tokens | 2442 |
| Completion tokens | 114 |
| Total tokens | 2556 |
| 平均 tokens/case | 511.2 |
| 告警生成数 | 1 |

## 6. LlamaIndex RAG 检索评估

命令：

```bash
PYTHONPATH=src ./.venv311/bin/python scripts/eval_llamaindex_rag.py
```

| 指标 | 结果 |
| --- | ---: |
| Hit rate | 100% |
| Recall@5 | 69.5% |
| Precision@5 | 86.67% |
| MRR@5 | 100% |

说明：评估已按 document_id 去重，避免同一文档多 chunk 被重复计数。

## 7. Hybrid RAG 证据质量

- 多查询融合：`hybrid_rag_search`
- 引擎：LlamaIndex + RRF
- 去重粒度：document-level
- 输入清洗：Reddit RSS HTML 已清理

实测问题 `wireless earbuds safety`：

- evidence_count: 5
- tool_call_success_rate: 100%
- 平均延迟约 336ms（包含构建 LlamaIndex 索引）

测试：`tests/test_hybrid_retrieval.py`，结果 6/6 passed。

## 8. 扩充评测集（63 case）

新增自动生成脚本：

```bash
PYTHONPATH=src ./.venv311/bin/python scripts/generate_evalset.py
```

生成结果：

- RAG cases: 63
- Pipeline cases: 63
- 基于当前 SQLite 中 70 篇真实公开数据、56 篇去重文档

### RAG 检索评估

命令：

```bash
PYTHONPATH=src ./.venv311/bin/python scripts/eval_generated_rag.py
```

| 指标 | 结果 |
| --- | ---: |
| Case 数 | 63 |
| Hit rate | 96.83% |
| Recall@5 | 89.79% |
| Precision@5 | 31.88% |
| MRR@5 | 81.61% |

### Pipeline 评估

命令：

```bash
PYTHONPATH=src ./.venv311/bin/python scripts/eval_generated_pipeline.py
```

| 指标 | 结果 |
| --- | ---: |
| Case 数 | 63 |
| 任务完成率 | 100% |
| 工具调用成功率 | 100% |
| 解析成功率 | 100% |
| 平均延迟 | 1.48ms |
| 分类准确率（有效标注 4 条） | 75% |
| 情绪准确率 | 100% |
| 推理准确率 | 87.3% |
| 实体 Precision | 100% |
| 实体 Recall | 50% |
| 漏报率 | 100% |
| 误报率 | 11.29% |

说明：

- 分类准确率只统计非 `unknown` 的有效标注，共 4 条，样本仍偏少。
- 漏报率基于当前仅 1 条风险正样本，统计意义有限。
- Precision@5 较低主要因为部分主题相关文档集合较大，top_k=5 无法覆盖全部。

## 9. 模型驱动开放域分析工具验证

测试主题：`smart home camera privacy leak`（非充电/发热主题）

### judge_evidence 实测结果

```json
{
  "relevance": "medium",
  "sufficiency": "insufficient",
  "credibility": "low",
  "coverage": "narrow",
  "evidence_gaps": [
    "缺乏厂商或官方安全公告确认泄露事件",
    "缺少泄露的技术细节",
    "缺少受影响用户数量与影响范围数据"
  ]
}
```

### analyze_consumer_sentiment 实测结果

```json
{
  "issue_type": "privacy",
  "product_categories": ["smart home camera"],
  "sentiment": "negative",
  "risk_level": "high",
  "impact_level": "high",
  "evidence_sufficient": false,
  "confidence": 0.5
}
```

结论：模型驱动工具可以对非预设主题做开放域分析，并明确输出证据不足边界。

## 10. 高质量标注 + 成本指标评估（61 case）

评估命令：

```bash
PYTHONPATH=src ./.venv311/bin/python scripts/eval_node_and_e2e.py
```

已改为真实 LLM Agent（LangGraphReactAgent），成本结构记录真实 token usage。

| 指标 | 结果 |
| --- | ---: |
| Case 数 | 61 |
| 任务完成率 | 100% |
| 工具调用成功率 | 100% |
| 解析成功率 | 100% |
| 平均延迟 | 164.98ms |
| 平均工具延迟 | 12.33ms |
| 分类准确率 | 53.57% |
| 情绪准确率 | 49.18% |
| 推理准确率 | 16.39% |
| 实体 Precision | 100% |
| 实体 Recall | 27.12% |
| 漏报率 | 40% |
| 误报率 | 32.61% |
| Prompt tokens | 43,611 |
| Completion tokens | 1,566 |
| Total tokens | 45,177 |
| LLM calls | 183 |
| 平均 tokens/case | 740.61 |
| 平均 tool calls/case | 10.34 |

## 11. 节点级与端到端评测（v1.2）

新增：

```text
data/evalsets/node_eval_cases.jsonl
src/eval/node_eval.py
src/eval/end_to_end_eval.py
scripts/eval_node_and_e2e.py
tests/test_node_and_e2e_eval.py
```

运行：

```bash
PYTHONPATH=src ./.venv311/bin/python scripts/eval_node_and_e2e.py
```

### Retrieval Node

| 指标 | 结果 |
| --- | ---: |
| Hit rate | 100% |
| Recall@5 | 72.00% |
| Precision@5 | 71.67% |
| MRR@5 | 86.67% |
| Evidence relevance rate | 100% |

### Analysis Node（rule baseline）

| 指标 | 结果 |
| --- | ---: |
| Category accuracy | 20.00% |
| Sentiment accuracy | 40.00% |
| Risk accuracy | 100.00% |
| Impact accuracy | 0.00% |
| Entity precision | 26.92% |
| Entity recall | 53.85% |

### Output Node

| 指标 | 结果 |
| --- | ---: |
| Report generation rate | 100% |
| Report structure rate | 100% |
| Archive generation rate | 100% |
| Evidence reference rate | 100% |
| Alert trigger accuracy | 100% |

### End-to-End

| 指标 | 结果 |
| --- | ---: |
| Task completion rate | 100% |
| Tool call success rate | 100% |
| Retrieval hit rate | 100% |
| Analysis exact match rate | 0% |
| Report generation rate | 100% |
| Archive generation rate | 100% |
| Alert trigger accuracy | 100% |
| Evidence traceable rate | 100% |
| Avg latency | 906.83ms |
| Prompt tokens | 3,379 |
| Completion tokens | 130 |
| Total tokens | 3,509 |
| Avg tokens/case | 701.80 |
| Avg tool calls/case | 13.80 |

说明：

- 检索节点和输出节点表现稳定。
- 规则分析层 exact match 为 0，说明影响等级判断与标注存在明显差距。
- 这进一步验证应使用 `analyze_consumer_sentiment` 的 LLM backend 做最终分析评估。

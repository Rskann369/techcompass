# TechCompass — 检索召回评估报告（v1.1）

## 背景

v1.0 之前只评估了「评论分类/情绪/风险」，缺少对「检索召回」的评估。
同时，给 LLM 生成结果用的召回库只有 4 条手写样例，证据严重不足。

v1.1 补齐两件事：
1. 扩充本地召回库（用于跑通召回评估框架）。
2. 新增检索召回评估器，量化 hit_rate / recall / precision。

## 召回库

- 文件：`data/samples/sample_reviews_large.jsonl`（156 条）
- 说明：**仍是项目构造的中文评论样本**，不是真实公开数据。用于让召回评估在
  较大规模下有意义。真实公开数据集（HuggingFace Amazon）接入受阻于缓存权限，
  需后续处理（见 handoff）。

## 检索逻辑升级

- 升级前：整段 query 作为子串去匹配 title+content，导致「发烫」查不到「充电发热」等。
- 升级后：分片关键词命中 + 同义词扩展 + 计分排序（见 `src/tools/domain_tools.py`）。

## 评估结果

评估集：`data/evalsets/retrieval_queries.jsonl`（10 个 query，每个带 relevant_ids 标注）
运行：`PYTHONPATH=src ./.venv311/bin/python scripts/wk4_retrieval_eval.py`

### 升级前后对比

| 指标 | 升级前 | 升级后 |
| --- | --- | --- |
| 命中率 hit_rate | 60.00% | **100.00%** |
| 平均召回率 avg_recall | 33.33% | **58.33%** |
| 平均精确率 avg_precision | 60.00% | **100.00%** |

### 说明

- avg_recall 58.33% 的瓶颈是 top_k=5 截断：每个 query 应命中 8 条，最多返回 5 条，
  召回上限即 62.5%。放大 top_k 可进一步提高 recall。
- precision=100% 表示「返回的基本都相关」，质量可靠。

## 下一步

- [ ] 用真实公开数据集替换构造样本。
- [ ] 接入真实 embedding 做语义检索（当前仍为关键词检索）。
- [ ] 扩大 query 标注集规模，做显著性统计。

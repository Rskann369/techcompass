# TechCompass — RAG 逻辑与评分报告

## 1. 当前 RAG 逻辑（如实描述）

当前系统有两种“检索”链路，不要混淆。

### A. Agent / 工具调用链路（当前主要可用链路）

```
用户问题
  → LLM（bind_tools）
  → 调用 search_news / search_discussions / search_reviews
  → 外部公开 API / 本地检索返回结果
  → 工具结果 JSON 回填给 LLM
  → LLM 基于工具结果生成结论
```

这不是传统“先建向量库再检索”的 RAG，而是 **Tool-based Retrieval + LLM 合成**。

当前实时数据源：

| 源 | 作用 | 特点 |
| --- | --- | --- |
| GNews | 新闻/媒体评测 | 真实数据，免费版 12h 延迟，偶发 0 结果 |
| Reddit RSS | 用户讨论 | 真实数据，无需凭据，可能 429 限流，结果新旧混杂 |
| NewsAPI | 候选 | 当前网络不可达，未使用 |

### B. Workflow 内的 LlamaIndex 链路（当前仍是占位）

`src/workflows/nodes.py` 里 `node_index_retrieve` 仍使用：

- `MockEmbedding`
- `MockLLM`
- 只用于验证流程，不提供真实语义检索能力

也就是说：**Workflow 里的向量 RAG 尚未真正落地，真正可用的是实时 API 检索。**

---

## 2. RAG 评分设计

新增评分器：`src/eval/rag_eval.py`

评分维度：

| 维度 | 权重 | 含义 |
| --- | --- | --- |
| coverage | 35% | 查询关键词在标题/正文中的覆盖比例 |
| freshness | 20% | 7 天内 1 分，30 天内 0.5 分 |
| completeness | 20% | title/content/url/published_at 是否齐全 |
| source_diversity | 10% | 是否来自多个不同站点/社区 |
| result_count | 15% | 是否返回足够候选（≥5 条满分） |

单条结果另有 `title_match` / `content_match` / `matched_terms` 明细。

---

## 3. 实测结果

运行命令：

```bash
PYTHONPATH=src ./.venv311/bin/python scripts/rag_eval.py
```

测试 query：

- `wireless earbuds`
- `smartwatch battery`

### 汇总

| Source | Query | Count | Overall | Coverage | Freshness | Diversity |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| gnews | wireless earbuds | 5 | **95.67%** | 93.33% | 100% | 80% |
| gnews | smartwatch battery | 0 | 0% | 0% | 0% | 0% |
| reddit_rss | wireless earbuds | 5 | **78.00%** | 100% | 30% | 20% |
| reddit_rss | smartwatch battery | 0 | 0% | 0% | 0% | 0% |

### 结论

1. **GNews 对主关键词表现很好**：coverage、freshness、completeness 都高。
2. **Reddit 对主关键词覆盖好，但新旧混杂**：freshness 只有 30%，因为返回了 2018 年的帖子。
3. **两个源对 `smartwatch battery` 都返回 0**：说明多关键词组合查询能力弱，必须做 query rewrite / 拆词 / 同义词扩展。
4. **NewsAPI 当前不可达**，不参与评分。

---

## 4. 暴露的问题与下一步

| 问题 | 影响 | 优化方向 |
| --- | --- | --- |
| 多词查询无结果 | `smartwatch battery` 返回 0 | 拆词 / GNews OR 查询 / 查询重写 |
| Reddit 返回旧帖子 | freshness 低 | 按时间过滤或排序 |
| Reddit 来源单一 | diversity 低 | 限定多个 subreddit / 合并新闻源 |
| 无缓存 | 重复请求易限流 | 本地 TTL cache |
| Workflow 仍是 MockEmbedding | 批处理语义检索不可用 | 接入真实 embedding 或改用 API 检索 |

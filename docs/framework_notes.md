# LangChain / LlamaIndex 对比笔记 — Week 1

> 记录阶段：Sprint 1 / Week 1。目标是“能理解各自定位、能跑通离线最小 demo”，不是背 API。

---

## 1. LangChain 当前定位

你当前环境实际安装：`langchain==1.4.4`（Python 3.13 venv）。

LangChain 更适合用于：

- Agent 的**控制流**：Prompt、Chain、Runnable、Function Calling、多轮调日志。
- 组合模型调用和普通代码。
- 连接不同的模型 / memory / 工具 / structured output。

本项目里的主要用途：

- 构建 Agent 执行链。
- 管理 Prompt Template。
- 做 Tool Calling。
- 做结构化输出（JSON Schema）。
- 控制工具失败 / 重试 / 降级。

---

## 2. LlamaIndex 当前定位

你当前环境实际安装：`llama-index==0.14.25`（Python 3.13 venv）。

LlamaIndex 更适合用于：

- 文档 / 评价数据的**切分、Embedding、索引、检索**。
- 对外部知识库进行 RAG 检索。
- 处理本地文件、知识库、多个数据源的检索链路。

本项目里的主要用途：

- 读入公开评论 / 新闻样本。
- 建知识索引。
- 做相关评论 / 新闻检索。
- 给 Agent 提供“基于已知证据回答”的数据基础。

---

## 3. 本周实际结论（重要）

### 结论 A：一定要用 Python 3.13，不要用 3.9

原因：

- `banks==2.5.1` 中有 `Path | None` 这种 Python 3.10+ 写法。
- Python 3.9 import LlamaIndex 直接报：
  `TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'`
- LangChain 1.x / LlamaIndex 0.14 在当前版本组合下已经不适合 3.9。

所以本项目统一方案：

- 使用机器上已有的 `/opt/miniconda3/bin/python3.13`。
- 新建环境：`./.venv311`。
- 后续命令统一使用 `./.venv311/bin/python`、`./.venv311/bin/pip`。

### 结论 B：离线 demo 一定要配置 MockEmbedding + MockLLM

如果只配 `MockEmbedding`，LlamaIndex 查询时还会尝试加载 OpenAI LLM，会报缺 `OPENAI_API_KEY`。

正确写法：

```python
from llama_index.core import Settings
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.llms import MockLLM

Settings.embed_model = MockEmbedding(embed_dim=8)
Settings.llm = MockLLM()
```

说明：

- MockEmbedding 只验证“建索引/检索链路”能跑，不负责语义质量。
- 第二周必须换成真实 Embedding（或本地 embedding）。

---

## 4. 各框架 Quickstart 关键词

### LangChain 重点记住这些概念

| 概念 | 用途 |
| --- | --- |
| ChatPromptTemplate | 构造多角色 Prompt |
| LLM | 模型调用 |
| Chain | 把多个组件串起来 |
| Tool | 给 Agent 调用的确定性能力 |
| Structured Output | 按 JSON Schema 输出结果 |
| Runnable | LangChain 通用链路接口 |

### LlamaIndex 重点记住这些概念

| 概念 | 用途 |
| --- | --- |
| Document | 原始文档对象 |
| VectorStoreIndex | 向量索引 |
| Settings | 统一配置 LLM / embedding |
| query_engine | 基于索引执行查询 |
| chunks | 切分后的文本片段 |

---

## 5. 下一步怎么衔接 Week 2

- Week 2 会把现在这个最小 Demo 拆成 Workflow。
- 检索阶段的 `MockEmbedding` 会替换成真实 embedding 或本地 embedding。
- 情感分析的离线规则会替换成 Prompt + LLM + 结构化输出。
- 数据读取会变成 `src/adapters` 里的数据集适配器。

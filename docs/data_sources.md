# TechCompuss — 数据源与 API 候选清单 — Week 1

> 说明：这份清单用于固定 Week 1 已经识别出的资源，后续 Week 2 会从这里选择最稳定的几个接入。

---

## 1. 公开数据集候选

### 首选：Amazon 评论数据（数码品类）

- 来源：HuggingFace `McAuley-Lab/Amazon-Reviews-2023`
- 配置：`raw_meta_Electronics`（数码电子品类）
- 使用方式：`datasets` 流式读取，抽样后保存为 `data/raw/*.jsonl`
- 适合任务：
  - Week 1：确认能加载数据。
  - Week 2：评价统一 Schema 适配器。
  - Week 3：主题/情绪分类 Prompt 调试。
  - Week 4：评测集构建与性能评估。

### 其他补充候选

| 数据集 | 来源 | 适用场景 |
| --- | --- | --- |
| Amazon Fine Food Reviews | Kaggle | 评价情感分类、客服建议 |
| Amazon Customer Reviews | Kaggle | 评论星级 + 文本分析 |
| News Headlines 数据集 | HuggingFace | 新闻标题检索、风险主题识别 |
| Support ticket / consumer complaints | Kaggle / HuggingFace | 客服工单分类、行动建议 |
| 中文电商评论（可选） | HuggingFace 搜索 `chinese sentiment` | 中文场景验证 |

---

## 2. 公开 API 候选

### 当前浏览器已打开

- 你当前在浏览器里看的是 `https://newsapi.org/`
- 项目里可以把它作为新闻检索 API 候选之一。
- 前提：需要注册并拿到 `NEWS_API_KEY`。
- Week 1 只需要“能调用并拿到一条结果”即可，不强制接入。

### 首选：News API

- 官网：https://newsapi.org/
- 用途：关键词/日期检索新闻。
- 计费：免费额度有限，注意限制和地区政策。

### 可选新闻 API

| API | 地址 | 备注 |
| --- | --- | --- |
| GNews | https://gnews.io | 新闻检索，免费额度 |
| The Guardian API | https://open-platform.theguardian.com | 新闻检索，需注册 |

### 本地 Mock API（第一优先级）

在 Week 1 固定做法：先写 `mock_api.py`，不依赖外部网络。

它提供：

- `POST /search_news`：返回固定新闻样例。
- `POST /analyze_reviews`：返回固定分析结果。
- `GET /health`：健康检查。

目标：保证即使没有真实 key，Step 5 链路也可以跑通。

---

## 3. 环境变量清单

对应 `.env.example`：

```text
# LLM / OpenAI-compatible API
OPENAI_API_KEY=
OPENAI_BASE_URL=
OPENAI_MODEL=gpt-4o-mini

# Public API
NEWS_API_KEY=
GNEWS_API_KEY=

# Hugging Face token（private/gated data）
HUGGING_FACE_HUB_TOKEN=
```

重要规则：

- 真实值只放 `.env`。
- `.env` 已加入 `.gitignore`。
- `.env.example` 只放名字，不放真实值。

---

## 4. 周执行结论

- 已确定主框架：LangChain + LlamaIndex。
- 已确定数据类：数码产品评论 + 新闻标题。
- 已确定 API 方向：News API（候选）+ 本地 Mock API（第一优先级）。
- 已确定 Python 环境：`./.venv311`。

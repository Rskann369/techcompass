# TechCompass — Risik清单 / 技术遗留项 — Week 1

> 这里记录的是“当前已经识别但暂时没完全解决、后续需要持续处理”的问题。

---

## 风险清单

| # | 风险 | 影响 | 缓解措施 |
| --- | --- | --- | --- |
| 1 | Python 3.9 与 LangChain/LlamaIndex 依赖不兼容 | 原 .venv 无法跑 LlamaIndex | 统一改用 `./.venv311`（Python 3.13） |
| 2 | 没有 OPENAI_API_KEY 时无法跑真实 LLM | 最小 Demo 只能离线规则；不验证真实模型能力 | 后续获得 key 后切换到 ChatOpenAI |
| 3 | News API 是免费额度有限 | 多次测试可能因次数超额失败 | Week 2 必须做本地缓存 + 结果落盘 + 降级 |
| 4 | 公开数据集下载可能很大 | 容易把本地环境搞慢 | 使用 `streaming=True` 抽样，禁止全量下载 |
| 5 | MockEmbedding 无语义 | 现在检索引不到正确语义结果 | Week 2 替换为真实 embedding 或本地 embedding |
| 6 | 最小 Demo 尚未工程化 | 不适合生产，只用于 Week 1 验证 | Week 2 拆 Workflow + Adapter + Runner |
| 7 | API key 泄露风险 | 真实密钥可能被提交 | `.env` 已 ignore，后续接入必须继续遵守 |

---

## 技术遗留项

1. Python 环境迁移：
   - 已完成：`.venv311` 已创建，核心依赖已装。
   - 待完成：把后续所有命令 / 文档统一为 `.venv311`，不要继续用 `.venv`（3.9）。

2. 真实 LLM 接入：
   - 待完成：拿到 `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL` 后做一次真实调研。

3. 真实新闻 API 接入：
   - 待完成：拿到 `NEWS_API_KEY` 后验证 NewsAPI。

4. 本地 Mock API 设计：
   - 待完成：Week 2 正式写 `mock_api.py` 或改造成工具服务。

5. 数据统一 Schema：
   - 待完成：Week 2 定义 `src/adapters` 的统一 `Item` Schema。

---

## 当前可以确认的结论

- LangChain 离线链路：✅ 已跑通。
- LlamaIndex 离线索引/检索链路：✅ 已在 Python 3.13 环境跑通。
- 最小 Demo 离线兜底链路：✅ 已跑通。
- 真实 LLM / 真实 API：⏳ 等待用户提供 key 后接入并验证。

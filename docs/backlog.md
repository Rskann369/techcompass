# TechCompass — Sprint Backlog

> 轻量敏捷：4 个 Sprint 对应项目四周。每个 Sprint 有一条「完成定义（DoD）」。

## Sprint 1（Week 1）：环境搭建 + 框架熟悉 + 最小 Demo

**目标**：具备可运行基础 Demo，圈定数据源与 API。

- [x] 初始化项目目录结构
- [x] 搭建 Python 虚拟环境（.venv）
- [x] 安装 LangChain / LlamaIndex / datasets / FastAPI 等依赖
- [ ] 摘录 LangChain Quickstart（调 LLM + Function Calling）
- [ ] 摘录 LlamaIndex Quickstart（建索引 + 检索）
- [ ] 确定数据集候选清单（docs/data_sources.md）
- [ ] 确定公开 API 候选清单（docs/data_sources.md）
- [ ] 跑通最小 Demo（scripts/mini_demo.py）
- [ ] 记录技术遗留项（docs/risks.md）

**DoD**：
1. 干净环境可按 README 复现安装。
2. LangChain 示例可成功调 LLM 并输出。
3. LlamaIndex 示例可成功建索引并检索。
4. 至少一个 API（mock 或公开）返回 ok=True 数据。
5. 目录、README、tests、scripts、.env.example、.gitignore 已建立。

## Sprint 2（Week 2）：Agent 核心 Workflow 与调度逻辑

**目标**：把零散能力沉淀为可调度、可监控、可扩展的 Workflow。

- [x] 设计主流程图（Ingest → Normalize → Index/Retrieve → Analyze → Reason → Output → Archive）
- [x] 定义统一数据 Schema
- [x] 开发数据集适配器 + 公开 API 适配器（公开 API 适配器 Week3 深化）
- [x] 开发调度器与错误重试/降级逻辑
- [x] 开发 Pipeline Runner
- [x] 端到端舆情摘要首次跑通

**DoD**：固定样例可自动生成舆情摘要；API 失败可降级本地样本；重复执行不产生重复输出。

## Sprint 3（Week 3）：Prompt 调优 + Tool Calling + 控制流闭环

**目标**：让 Agent 从「能跑通」到「稳定可控」、工具调用可靠。

- [x] 按节点模块化设计 Prompt
- [x] 定义结构化输出 Schema（Tool Calling 强制结构）
- [x] 开发工具链（search_reviews / search_news / classify_review）
- [x] 定义统一工具注册规范（ToolRegistry）
- [x] 完成 6 类控制流闭环测试（成功/工具失败/重复调用/信息不足/越界/最大轮数）

**DoD**：目标用例输出稳定为合法 JSON；核心工具成功率达标；无死循环；有可复现调优记录。

## Sprint 4（Week 4）：性能评估 + 技术架构文档交付

**目标**：形成可评审、可汇报、可交接的 v1.0 交付包。

- [x] 构建评测集（分类/情绪，10 条标注 ground truth）
- [x] 评估指标（category/sentiment/overall accuracy，baseline 可跑）
- [x] 修复 search_reviews 证据污染；LLM 超时+重试
- [x] 输出完整技术架构文档（docs/architecture.md）
- [x] 输出 README / Demo / 评估报告 / 交接说明

**DoD**：clean 环境可复现全流程；结果可追溯；核心链路达标；高风险路径有人工确认保护。

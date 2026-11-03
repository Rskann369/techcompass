# TechCompass — Agent 核心 Workflow 说明（Week 2）

> 本文档是 Week 2 的技术交付之一，描述 Workflow 的架构、节点职责、调度与错误处理。

---

## 1. 总体架构

```
                    ┌─────────────────────┐
   local jsonl/csv  │   adapters          │
   ───────────────▶ │  (dataset / api)    │──────▶ Item 列表（统一 Schema）
                    └─────────────────────┘
                             │
                             ▼
        ┌──────────────── Workflow Runner ────────────────┐
        │                                                 │
        │  ingest ─▶ normalize ─▶ index_retrieve          │
        │                 │               │              │
        │                 ▼               ▼              │
        │             analyze ───────▶ reason             │
        │                 │               │              │
        │                 └──────▶ output ◀───────┘      │
        │                              │                 │
        │                              ▼                 │
        │                          archive                │
        └─────────────────────────────────────────────────┘
```

## 2. 节点职责

| 节点 | 文件 | 职责 |
| --- | --- | --- |
| ingest | nodes.py | 接收适配器产出的 Item |
| normalize | nodes.py | 去重、字段兜底 |
| index_retrieve | nodes.py | 建索引 + 检索（Week2 用 MockEmbedding/MockLLM） |
| analyze | nodes.py | 主题/情绪/风险规则分析（Week3 换 LLM） |
| reason | nodes.py | 影响判断 → 行动建议 |
| output | nodes.py | 生成结构化摘要 |
| archive | nodes.py | 结果落盘（JSON） |

## 3. 数据与 State

- 统一数据模型：`src/adapters/schema.py` 的 `Item`。
- 节点间共享 `WorkflowState`（`src/workflows/nodes.py`）。
- 每个节点输入输出都是 state，方便测试与扩展。

## 4. 调度与错误处理

- `src/workflows/scheduler.py`：配置化节点顺序（`WorkflowScheduler`）。
- 每个节点可声明 `allow_fail=True`，失败时记录并继续。
- 默认严格模式：未声明允许失败的节点出错会停止并记录 error。
- `src/workflows/runner.py`：数据源失败时降级到本地样本，避免崩溃。

## 5. 已验证能力

- [x] 端到端跑通（7 节点全 ok）
- [x] 数据源文件缺失 → 降级本地样本，ok=True 不崩溃
- [x] 同批数据重复执行 → 幂等，结果不翻倍

## 6. 当前的技术债（Week 3 待解决）

1. `index_retrieve` 仍用 MockEmbedding，无语义区分度 → Week 3 换真实 embedding。
2. `analyze` 仍用规则 → Week 3 换成 Prompt + LLM 结构化输出。
3. 尚无真正的外部 News API 适配器 → Week 3 与 Tool Calling 一起完成。
4. 尚无「意图 → 多个工具 → 汇总」的多步骤 Agent 控制循环 → Week 3 核心。

## 7. 复现方式

```bash
cd techcompass
source ./.venv311/bin/activate
PYTHONPATH=src python -m workflows.runner
```

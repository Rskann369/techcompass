# TechCompass — Week 3：Prompt + Tool Calling + 控制流闭环

> 本文档记录 Week 3 的实现与验证。

---

## 1. 本周完成的事

把 Week 2 的「规则 + 假 embedding」升级为「真实 LLM Agent + Tool Calling」。

主要新增：

| 模块 | 文件 | 作用 |
| --- | --- | --- |
| Prompt 模板 | `src/prompts/*.py` | 角色/任务/结构/禁止事项统一约定 |
| 输出 Schema | `src/agents/schemas.py` | Pydantic + 工具参数 Schema |
| 工具注册 | `src/tools/registry.py` | 统一 Tool + Registry |
| 领域工具 | `src/tools/domain_tools.py` | search_reviews / search_news / classify_review |
| Agent 控制循环 | `src/agents/agent.py` | 多轮 bind_tools 循环 + 控制流保护 |
| Demo 入口 | `scripts/wk3_agent_demo.py` | 一次真实端到端运行 |

---

## 2. 关键技术决策

### 决策 1：不用 with_structured_output 直接赌 JSON

实测发现，当前模型（DeepSeek-V4-Flash）在 `with_structured_output` 下返回的是纯文本
「分类：充电/电池/...」，不是 JSON，导致 Pydantic 校验失败。

结论：**用 Tool Calling（bind_tools）强制结构化**，通过 `tool_calls` 拿参数，比赌 LLM 输出 JSON 更稳。

### 决策 2：重复调用保护要精确

第一版规则是「同一工具调 N 次就停」，结果误杀了正常探索——Agent 用不同关键词查了
3 次 `search_reviews`，这是合理行为。

修正为：只拦截「完全相同（同工具名 + 同参数）的调用反复重放」。

---

## 3. 控制流闭环测试（6/6 通过）

| 场景 | 状态 |
| --- | --- |
| 成功闭环 | PASS |
| 工具失败降级 | PASS |
| 相同调用重复（防死循环） | PASS |
| 信息不足 | PASS |
| 越界（超出范围） | PASS |
| 最大轮数截断 | PASS |

测试文件：`tests/test_wk3_control_flow.py`

---

## 4. 真实运行示例

demo 实际输出（节选）：

- 工具调用：search_reviews(充电发热) → search_reviews(安全隐患 过热) → search_reviews(充电 烫 危险) → search_news → classify_review(产品问题, negative)
- 最终结论：仅 1 条直接相关负面反馈，明确标注「样本太小不足以判定普遍缺陷」，并给出客服建议。

关键质量点：Agent 会**明说证据不足、不夸大**，这是有效的舆情边界行为。

---

## 5. 遗留技术债（Week 4 待处理）

1. `search_reviews` 在关键词未命中时会把无关评论兜底塞进结果（`items[:max]`）→ 需改成语义检索或严格命中。
2. 检索仍是规则命中，未用真实 embedding 做语义检索。
3. 尚未做批量评测集与指标量化。
4. 尚未做成本/延迟/工具成功率统计。

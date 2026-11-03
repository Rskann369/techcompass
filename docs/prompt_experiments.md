# TechCompass — Prompt 调优实验记录

## 目标

建立 Prompt 版本化、可复现实验和失败样例分析机制，避免“只改 Prompt 不留痕”。

## 版本约定

- `v0.x`：规则基线，不使用 LLM
- `v1.x`：LLM Tool Calling 初版
- `v2.x`：开放域模型驱动分析

## Prompt 清单

| 模块 | 文件 | 当前版本 | 说明 |
| --- | --- | --- | --- |
| 检索意图 | `src/prompts/retrieval_prompt.py` | v2 | 多查询改写，禁止编造品牌 |
| 主题/情绪分类 | `src/prompts/classify_prompt.py` | v2 | 支持开放域，不限定产品品类 |
| 实体抽取 | `src/prompts/entity_prompt.py` | v2 | 输出品类、型号、部件、渠道、风险词 |
| 原因推理 | `src/prompts/reason_prompt.py` | v2 | 强制基于证据，证据不足则说明 |
| 影响评估 | `src/prompts/impact_prompt.py` | v2 | 输出 low/medium/high/critical |
| 行动建议 | `src/prompts/action_prompt.py` | v2 | 客服、产品、传播三维建议 |
| 周报生成 | `src/prompts/report_prompt.py` | v2 | Markdown 结构化周报 |
| 证据判断 | `src/tools/llm_analysis_tools.py` | v2 | 相关性、充分性、可信度、覆盖面 |
| 开放域分析 | `src/tools/llm_analysis_tools.py` | v2 | 不预设 topic，由模型自行判断 |

## 实验 1：规则基线 v0

### 设置

- 评估集：`data/evalsets/enriched_pipeline_cases.jsonl`
- Case 数：61
- 引擎：确定性规则工具
- 不调用真实 LLM

### 结果

| 指标 | 结果 |
| --- | ---: |
| 任务完成率 | 100% |
| 工具调用成功率 | 100% |
| 分类准确率 | 46.43% |
| 情绪准确率 | 54.10% |
| 推理准确率 | 13.11% |
| 实体 Precision | 100% |
| 实体 Recall | 21.19% |
| 漏报率 | 40.00% |
| 误报率 | 45.65% |

### 结论

规则层稳定性好，但开放域判断能力明显不足：
- 分类和实体召回偏低
- 风险误报高
- 推理准确率低

这验证了“硬编码 topic 不可泛化”的问题。

---

## 实验 2：开放域模型驱动分析 v2 抽样

### 设置

- 评估集：`data/evalsets/enriched_pipeline_cases.jsonl`
- 抽样：5 case（固定间隔抽样）
- 工具：`analyze_consumer_sentiment`
- 证据：Hybrid RAG top 5

### 结果

| 指标 | 结果 |
| --- | ---: |
| 完成率 | 60% |
| 分类准确率 | 20% |
| 情绪准确率 | 20% |
| 风险判断准确率 | 60% |
| 影响准确率 | 40% |

### 失败样例

1. **Request timed out**
   - 问题：`5 wireless earbuds for audiophiles who`
   - 原因：内网网关超时
   - 改进：增加超时重试；降低证据长度

2. **model did not return structured tool call**
   - 问题：`HyperX Alloy Rise Gaming Keyboard Review`
   - 原因：模型未按 Tool Calling 返回结构化参数
   - 改进：Prompt 增加强制 Tool Calling 指令；失败后自动重试一次

3. **分类口径不一致**
   - 例子：模型输出 `wireless_earbuds`，标注为 `headphones`
   - 改进：建立 category normalization 映射，例如：
     - `wireless_earbuds` → `headphones`
     - `audio` → `headphones`
     - `wireless speaker` → `speaker`

4. **情绪口径不一致**
   - 例子：模型对正向评测输出 `positive`，标注为 `neutral`
   - 原因：标注规则将非风险默认为 neutral，而模型会识别正向评价
   - 改进：评测标注应允许 `positive` / `neutral` / `negative`，不能一刀切 neutral

## 下一步 Prompt v2.1

1. 增加强制 Tool Calling 指令。
2. 输出 category 时使用标准化标签。
3. 降低 evidence 输入长度，避免网关超时。
4. 增加失败重试一次。
5. 修正评测标注的情绪规则。

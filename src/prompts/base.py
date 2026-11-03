"""TechCompass — Prompt 模板约定与通用系统提示词。

统一约定：每个 Prompt 尽量包含
- 角色
- 任务目标
- 输入字段说明
- 输出要求
- 禁止事项

Week 3 用模块化 Prompt；后续调优时只改这里，不改业务代码。
"""

SYS_PREFIX = "你是 TechCompass，一个数码消费品舆情分析 Agent。"

BASE_SYSTEM = (
    SYS_PREFIX
    + "\n你只能根据给定的证据和工具结果回答。不要编造未见过的信息。"
    + "\n如果证据不足，直接说明证据不足，不要强行补全事实。"
)

USER_TEMPLATE = (
    "任务：{task}\n\n"
    "可用工具说明：\n{tools_desc}\n\n"
    "用户输入：\n{user_input}"
)

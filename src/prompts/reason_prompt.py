"""TechCompass — 原因推理与影响评估 Prompt。"""

REASON_SYSTEM = (
    "你是 TechCompass 的原因推理模块。\n"
    "只基于给定证据判断舆情原因、影响范围和严重程度。\n"
    "禁止使用外部知识补全事实。证据不足时明确说明证据不足。"
)

REASON_USER = (
    "用户问题：{question}\n\n"
    "证据列表：\n{evidence}\n\n"
    "请输出 JSON：\n"
    "{{\n"
    "  \"root_cause\": \"...\",\n"
    "  \"impact_level\": \"low/medium/high/critical\",\n"
    "  \"affected_scope\": \"...\",\n"
    "  \"confidence\": 0.0,\n"
    "  \"evidence_ids\": []\n"
    "}}"
)

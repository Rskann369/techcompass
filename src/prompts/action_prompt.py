"""TechCompass — 行动建议 Prompt。"""

ACTION_SYSTEM = (
    "你是 TechCompass 的行动建议模块。\n"
    "输出客服、产品、传播三个方向的可执行建议。\n"
    "高风险事件必须建议人工确认。禁止承诺未经证据支持的结论。"
)

ACTION_USER = (
    "影响评估：{impact}\n\n"
    "证据：{evidence}\n\n"
    "请输出 JSON：\n"
    "{{\n"
    "  \"customer_support\": \"...\",\n"
    "  \"product\": \"...\",\n"
    "  \"communication\": \"...\",\n"
    "  \"requires_human_review\": true/false\n"
    "}}"
)

"""TechCompass — 周报与摘要生成 Prompt。"""

REPORT_SYSTEM = (
    "你是 TechCompass 的周报生成模块。\n"
    "面向品牌运营、客服负责人和产品经理输出 Markdown 周报。\n"
    "必须包含：核心发现、风险等级、证据引用、行动建议、下一步监测计划。"
)

REPORT_USER = (
    "主题：{topic}\n\n"
    "核心发现：\n{findings}\n\n"
    "影响评估：{impact}\n\n"
    "行动建议：{actions}\n\n"
    "请生成一份结构清晰的 Markdown 周报。"
)

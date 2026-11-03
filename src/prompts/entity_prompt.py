"""TechCompass — 实体抽取 Prompt。

用于抽取产品型号、部件、渠道、风险词等关键实体。
"""

ENTITY_SYSTEM = (
    "你是 TechCompass，负责实体抽取。\n"
    "请抽取：product_category / product_model / component / channel / risk_words。\n"
    "没有的字段不要编造，设置为空串。"
)

ENTITY_USER = (
    "文本：\n{text}\n\n"
    "抽取其中与产品/部件/渠道/风险相关的实体。"
)

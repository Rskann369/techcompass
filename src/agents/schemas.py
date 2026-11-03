"""TechCompass — 结构化输出 Schema（Week 3）。

核心策略：
- 不依赖 LLM 直接输出合法 JSON。
- 用 `bind_tools` 把输出结构作为 Tool Calling 参数 Schema，
  让 LLM 调用工具时给出的 JSON 参数更稳定，再用 Pydantic 校验。
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ClassifySchema(BaseModel):
    category: str = Field(description="产品问题 / 物流问题 / 宣传问题 / 使用问题 / 其他")
    sentiment: str = Field(description="positive / neutral / negative / risk")


class EntitySchema(BaseModel):
    product_category: Optional[str] = None
    product_model: Optional[str] = None
    component: Optional[str] = None
    channel: Optional[str] = None
    risk_words: list[str] = Field(default_factory=list)


class ImpactSchema(BaseModel):
    priority: str = Field(description="low / medium / high")
    evidence: list[str] = Field(default_factory=list)
    affected_scope: str = Field(default="")
    customer_support: str = Field(default="")
    product: str = Field(default="")


SCHEMA_TO_TOOL_DEF = {
    "classify": {
        "name": "classify_review",
        "description": "对一条数码消费品评论做主题分类和情绪判断",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {"type": "string"},
                "sentiment": {"type": "string"},
            },
            "required": ["category", "sentiment"],
        },
    },
    "impact": {
        "name": "emit_impact",
        "description": "根据证据输出风险等级和行动建议",
        "parameters": {
            "type": "object",
            "properties": {
                "priority": {"type": "string", "description": "low/medium/high"},
                "evidence": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "affected_scope": {"type": "string"},
                "customer_support": {"type": "string"},
                "product": {"type": "string"},
            },
            "required": ["priority"],
        },
    },
}

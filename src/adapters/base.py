"""TechCompass — 适配器公共基类与统一 API 结果模型。

所有数据源适配器（dataset / api）都实现统一接口，方便 Workflow 无差别调用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .schema import Item


@dataclass
class APIResult:
    """统一的 API/源返回结构，方便错误分类与降级。"""

    ok: bool
    items: list[Item] = field(default_factory=list)
    raw_data: Any = None
    error_code: Optional[str] = None
    retryable: bool = False
    fallback_action: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "count": len(self.items),
            "error_code": self.error_code,
            "retryable": self.retryable,
            "fallback_action": self.fallback_action,
        }


class BaseAdapter:
    """所有适配器的统一基类。"""

    name: str = "base"

    def fetch(self) -> APIResult:
        raise NotImplementedError

    def validate_config(self) -> bool:
        return True

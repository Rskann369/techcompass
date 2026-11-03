"""TechCompass — 统一工具注册规范与注册表。

约定每个工具至少提供：
- name / description / args_schema / run / fallback

支持两种输出：
- OpenAI function-calling schema
- LangChain StructuredTool
"""

from __future__ import annotations

from typing import Any, Callable, Optional


class Tool:
    """统一工具对象。"""

    def __init__(
        self,
        name: str,
        description: str,
        args_schema: dict[str, Any],
        run: Callable[[dict[str, Any]], dict[str, Any]],
        fallback: Optional[Callable[[], dict[str, Any]]] = None,
    ):
        self.name = name
        self.description = description
        self.args_schema = args_schema
        self.run = run
        self.fallback = fallback

    def to_openai(self) -> dict[str, Any]:
        """转成 OpenAI function-calling 格式。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_schema,
            },
        }

    def validate_args(self, args: dict[str, Any]) -> None:
        """轻量 JSON Schema 校验：required + type + enum。"""
        schema = self.args_schema or {}
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        for key in required:
            if key not in args or args[key] in (None, ""):
                raise ValueError(f"missing required argument: {key}")

        for key, value in args.items():
            if key not in properties:
                # 允许扩展字段，但工具内部自行处理
                continue
            expected = properties[key].get("type")
            if expected == "string" and not isinstance(value, str):
                raise TypeError(f"{key} must be string")
            if expected == "integer" and not isinstance(value, int):
                raise TypeError(f"{key} must be integer")
            if expected == "number" and not isinstance(value, (int, float)):
                raise TypeError(f"{key} must be number")
            if expected == "boolean" and not isinstance(value, bool):
                raise TypeError(f"{key} must be boolean")
            if expected == "array" and not isinstance(value, list):
                raise TypeError(f"{key} must be array")
            if expected == "object" and not isinstance(value, dict):
                raise TypeError(f"{key} must be object")
            enum_values = properties[key].get("enum")
            if enum_values and value not in enum_values:
                raise ValueError(f"{key} must be one of {enum_values}")

    def execute(self, args: dict[str, Any]) -> dict[str, Any]:
        self.validate_args(args)
        try:
            return self.run(args)
        except Exception as e:  # noqa: BLE001
            if self.fallback:
                return {"_error": str(e), "_fallback": True, **self.fallback()}
            return {"_error": str(e)}


def to_langchain_tool(tool: Tool):
    """Convert a TechCompass Tool to a LangChain StructuredTool.

    StructuredTool 会把 args_schema 里的字段以关键字参数形式传给 func，
    而 TechCompass Tool.run 只接受一个 dict，所以这里包一层。
    """
    from langchain_core.tools import StructuredTool

    def _run(**kwargs):
        return tool.execute(dict(kwargs))

    return StructuredTool.from_function(
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
        func=_run,
    )


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def all_defs(self) -> list[dict[str, Any]]:
        return [t.to_openai() for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def to_langchain_tools(self) -> list:
        """Return all tools as LangChain StructuredTools."""
        return [to_langchain_tool(tool) for tool in self._tools.values()]

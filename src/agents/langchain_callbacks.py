"""TechCompass — LangChain callback handler.

用 LangChain Callback 体系记录：
- LLM start/end/error
- Tool start/end/error
- token usage
- latency

事件先写入内存 buffer，由 Pipeline 持久化到 SQLite 或输出 trace。
"""

from __future__ import annotations

import time
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler


class TechCompassCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.events: list[dict[str, Any]] = []
        self._llm_started_at: dict[int, float] = {}
        self._tool_started_at: dict[int, float] = {}

    def _serialize(self, value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, (list, tuple)):
            return [self._serialize(v) for v in value]
        if isinstance(value, dict):
            return {str(k): self._serialize(v) for k, v in value.items()}
        return str(value)

    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs):
        self._llm_started_at[run_id] = time.perf_counter()
        self.events.append({
            "type": "llm_start",
            "run_id": str(run_id),
            "prompts": self._serialize(prompts),
        })

    def on_llm_end(self, response, *, run_id, **kwargs):
        elapsed = time.perf_counter() - self._llm_started_at.pop(run_id, time.perf_counter())
        usage = getattr(response, "usage_metadata", None) or {}
        output = getattr(response, "generations", None)
        self.events.append({
            "type": "llm_end",
            "run_id": str(run_id),
            "elapsed_ms": round(elapsed * 1000, 3),
            "prompt_tokens": usage.get("input_tokens"),
            "completion_tokens": usage.get("output_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "output": self._serialize(output),
        })

    def on_llm_error(self, error, *, run_id, **kwargs):
        elapsed = time.perf_counter() - self._llm_started_at.pop(run_id, time.perf_counter())
        self.events.append({
            "type": "llm_error",
            "run_id": str(run_id),
            "error": str(error),
            "elapsed_ms": round(elapsed * 1000, 3),
        })

    def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):
        self._tool_started_at[run_id] = time.perf_counter()
        self.events.append({
            "type": "tool_start",
            "run_id": str(run_id),
            "tool": serialized.get("name") if isinstance(serialized, dict) else None,
            "input": self._serialize(input_str),
        })

    def on_tool_end(self, output, *, run_id, **kwargs):
        elapsed = time.perf_counter() - self._tool_started_at.pop(run_id, time.perf_counter())
        self.events.append({
            "type": "tool_end",
            "run_id": str(run_id),
            "output": self._serialize(output),
            "elapsed_ms": round(elapsed * 1000, 3),
        })

    def on_tool_error(self, error, *, run_id, **kwargs):
        elapsed = time.perf_counter() - self._tool_started_at.pop(run_id, time.perf_counter())
        self.events.append({
            "type": "tool_error",
            "run_id": str(run_id),
            "error": str(error),
            "elapsed_ms": round(elapsed * 1000, 3),
        })

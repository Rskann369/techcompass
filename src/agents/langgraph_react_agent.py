"""TechCompass — LangGraph create_react_agent 主链路 Agent。

目的：
- 使用 LangChain/LangGraph 生态的原生 React Agent 做真实 LLM 编排。
- 保持 TechCompass 的 trace、callback、SQLite 持久化。
- 输出运营可读结论（headline / summary / 带链接依据 / 置信度）。

实现：
    from langgraph.prebuilt import create_react_agent
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

from project_paths import DEFAULT_ENV_PATH

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from agents.langchain_callbacks import TechCompassCallbackHandler
from rag.store import RagStore
from tools.domain_tools import build_registry


@dataclass
class ReactAgentMetrics:
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    tool_calls: int = 0
    tool_success: int = 0
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "llm_calls": self.llm_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "tool_calls": self.tool_calls,
            "tool_success": self.tool_success,
            "tool_call_success_rate": round(self.tool_success / self.tool_calls, 4) if self.tool_calls else None,
            "elapsed_ms": self.elapsed_ms,
        }


class LangGraphReactAgent:
    """LangGraph create_react_agent wrapper for TechCompass."""

    def __init__(
        self,
        env_path: str | None = None,
        request_timeout: float = 120,
        max_steps: int = 8,
        verbose: bool = True,
    ):
        load_dotenv(env_path or DEFAULT_ENV_PATH)
        self.verbose = verbose
        self.max_steps = max_steps

        self.raw_llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0,
            request_timeout=request_timeout,
            max_retries=2,
        )
        # create_react_agent requires a LangChain Runnable LLM.
        # Use ChatOpenAI's native max_retries instead of custom wrapper.
        self.llm = self.raw_llm

        self.callback_handler = TechCompassCallbackHandler()
        self.registry = build_registry(
            dataset_path="data/samples/nonexistent.jsonl",
            env_path=env_path,
            callbacks=[self.callback_handler],
        )

        self.system_prompt = (
            "你是 TechCompass，消费品舆情侦察与行动 Agent。"
            "你必须基于工具返回的证据回答，不能编造。"
            "请依次使用 hybrid_rag_search 检索证据；"
            "若索引证据不足或与问题无关，可用 search_live_sources 现场拉取最新数据补充。"
            "再使用 analyze_consumer_sentiment 做开放域结构化分析。"
            "如果证据不足，不要生成建议。"
            "最后使用 generate_weekly_report 生成周报；"
            "高风险时使用 create_alert，然后使用 archive_run 归档。"
        )

        self.agent = create_react_agent(
            model=self.llm,
            tools=self.registry.to_langchain_tools(),
            prompt=self.system_prompt,
        )

        self.metrics = ReactAgentMetrics()
        self.trace: list[dict[str, Any]] = []
        self.store = RagStore()
        self.run_id: int | None = None

    def _extract_answer(self, state: dict[str, Any]) -> str:
        messages = state.get("messages", [])
        for message in reversed(messages):
            content = getattr(message, "content", None)
            if content and not getattr(message, "tool_calls", None):
                return str(content)
        return ""

    def _extract_tool_trace(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        trace: list[dict[str, Any]] = []
        for message in state.get("messages", []):
            tool_calls = getattr(message, "tool_calls", None) or []
            if tool_calls:
                for call in tool_calls:
                    self.metrics.tool_calls += 1
                    # LangGraph tools raise on failure; successful calls reach this branch.
                    self.metrics.tool_success += 1
                    trace.append({
                        "type": "tool",
                        "tool": call.get("name"),
                        "args": call.get("args"),
                        "status": "ok",
                    })
        return trace

    def _collect_usage(self, state: dict[str, Any]) -> None:
        for message in state.get("messages", []):
            usage = getattr(message, "usage_metadata", None) or {}
            if not usage:
                continue
            self.metrics.llm_calls += 1
            self.metrics.prompt_tokens += int(usage.get("input_tokens", 0) or 0)
            self.metrics.completion_tokens += int(usage.get("output_tokens", 0) or 0)
            self.metrics.total_tokens += int(usage.get("total_tokens", 0) or 0)

    def _extract_tool_results(self, state: dict[str, Any]) -> dict[str, list[Any]]:
        """从 LangGraph state 里提取每个工具的真实返回（ToolMessage content 解析为 dict）。"""
        from langchain_core.messages import ToolMessage

        results: dict[str, list[Any]] = {}
        for msg in state.get("messages", []):
            if not isinstance(msg, ToolMessage):
                continue
            name = getattr(msg, "name", "") or ""
            content = getattr(msg, "content", "")
            if isinstance(content, str):
                try:
                    data = json.loads(content)
                except (json.JSONDecodeError, ValueError):
                    data = content
            else:
                data = content
            results.setdefault(name, []).append(data)
        return results

    def _build_conclusion(self, question: str, state: dict[str, Any]) -> dict[str, Any] | None:
        """把真实 LLM 主链路结果翻译成结论 + 带链接依据 + 置信度。"""
        from agents.conclusion import build_conclusion_from_analysis

        tool_results = self._extract_tool_results(state)
        analysis: dict[str, Any] = {}
        for res in tool_results.get("analyze_consumer_sentiment", []):
            if isinstance(res, dict):
                analysis = res
        if not analysis:
            return None

        evidence_docs: list[dict[str, Any]] = []
        for name in ("hybrid_rag_search", "search_live_sources"):
            for res in tool_results.get(name, []):
                if isinstance(res, dict):
                    evidence_docs.extend(res.get("results", []))
        return build_conclusion_from_analysis(question, analysis, evidence_docs)

    def run(self, question: str) -> dict[str, Any]:
        started = time.perf_counter()
        self.run_id = self.store.start_agent_run(
            pipeline="langgraph_react_agent", question=question
        )

        config = {
            "recursion_limit": self.max_steps,
            "callbacks": [self.callback_handler],
        }
        state = self.agent.invoke(
            {"messages": [HumanMessage(content=question)]},
            config=config,
        )

        answer = self._extract_answer(state)
        tool_trace = self._extract_tool_trace(state)
        self._collect_usage(state)
        self.metrics.elapsed_ms = round((time.perf_counter() - started) * 1000, 3)

        conclusion = self._build_conclusion(question, state)

        self.trace.extend(tool_trace)
        self.store.insert_agent_trace(self.run_id, self.trace)
        self.store.insert_llm_events(self.run_id, self.callback_handler.events)
        self.store.finish_agent_run(
            run_id=self.run_id,
            status="success" if answer else "failed",
            answer=answer,
            metrics=self.metrics.to_dict(),
        )

        return {
            "ok": bool(answer),
            "agent_type": "langgraph_react_agent",
            "run_id": self.run_id,
            "question": question,
            "answer": answer,
            "tool_calls": tool_trace,
            "metrics": self.metrics.to_dict(),
            "trace": self.trace,
            "langchain_events": self.callback_handler.events,
            "conclusion": conclusion["conclusion"] if conclusion else None,
            "relevant_evidence": conclusion["structured"]["relevant_evidence"] if conclusion else [],
            "analysis": conclusion["structured"]["analysis"] if conclusion else {},
            "messages": state.get("messages", []),
        }

"""TechCompass — Workflow 调度器。

能力：
- 配置化节点顺序
- 节点级重试
- 指数退避
- 失败策略：strict / continue
- 执行记录：attempt、耗时、错误、最终状态

设计原则：
- 不把流程写死，节点由 NodeSpec 配置。
- strict 节点失败会终止整条 Workflow。
- continue 节点失败会记录错误但继续执行后续节点。
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from typing import Any, Callable, Literal


NodeStatus = Literal["ok", "failed", "skipped"]
FailurePolicy = Literal["strict", "continue"]


@dataclass
class NodeSpec:
    """Workflow 节点配置。"""

    name: str
    fn: Callable[..., Any]
    kwargs: dict[str, Any] = field(default_factory=dict)
    allow_fail: bool = False
    failure_policy: FailurePolicy = "strict"
    max_retries: int = 0          # 0 表示失败后不重试
    backoff_seconds: float = 0.1  # 指数退避基数
    timeout_seconds: float | None = None  # 预留，当前版本不做强杀线程

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if self.backoff_seconds < 0:
            raise ValueError("backoff_seconds must be >= 0")
        if self.failure_policy not in ("strict", "continue"):
            raise ValueError("failure_policy must be 'strict' or 'continue'")
        # 兼容旧字段：allow_fail=True 等价于 continue
        if self.allow_fail:
            self.failure_policy = "continue"


class WorkflowScheduler:
    """按顺序执行节点，并返回详细执行 summary。"""

    def __init__(
        self,
        specs: list[NodeSpec],
        verbose: bool = False,
        on_node_finished: Callable[..., None] | None = None,
    ):
        self.specs = specs
        self.verbose = verbose
        self.on_node_finished = on_node_finished

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message, flush=True)

    def _append_error(self, state: Any, message: str) -> None:
        errors = getattr(state, "errors", None)
        if isinstance(errors, list):
            errors.append(message)

    def run(self, state: Any) -> tuple[Any, list[dict[str, Any]]]:
        summary: list[dict[str, Any]] = []
        terminal_failure = False

        for spec in self.specs:
            if terminal_failure:
                record = {
                    "node": spec.name,
                    "status": "skipped",
                    "reason": "previous_strict_failure",
                }
                summary.append(record)
                if self.on_node_finished:
                    self.on_node_finished(record)
                continue

            attempts = 0
            node_ok = False
            last_error: Exception | None = None
            started_at = time.perf_counter()

            max_attempts = 1 + max(spec.max_retries, 0)
            for attempt in range(1, max_attempts + 1):
                attempts = attempt
                try:
                    if spec.timeout_seconds is None:
                        result = spec.fn(state, **spec.kwargs)
                    else:
                        with ThreadPoolExecutor(max_workers=1) as executor:
                            future = executor.submit(spec.fn, state, **spec.kwargs)
                            try:
                                result = future.result(timeout=spec.timeout_seconds)
                            except FutureTimeoutError as exc:
                                raise TimeoutError(
                                    f"node {spec.name} timed out after {spec.timeout_seconds}s"
                                ) from exc
                    if result is not None:
                        state = result
                    node_ok = True
                    last_error = None
                    break
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    self._log(f"[scheduler] node={spec.name} attempt={attempt} failed: {exc}")
                    if attempt < max_attempts:
                        sleep_for = spec.backoff_seconds * (2 ** (attempt - 1))
                        if sleep_for > 0:
                            time.sleep(sleep_for)

            elapsed_ms = round((time.perf_counter() - started_at) * 1000, 3)

            if node_ok:
                record = {
                    "node": spec.name,
                    "status": "ok",
                    "attempts": attempts,
                    "elapsed_ms": elapsed_ms,
                }
                summary.append(record)
                if self.on_node_finished:
                    self.on_node_finished(record)
                continue

            error_text = str(last_error) if last_error else "unknown error"
            record = {
                "node": spec.name,
                "status": "failed",
                "attempts": attempts,
                "elapsed_ms": elapsed_ms,
                "error": error_text,
                "failure_policy": spec.failure_policy,
            }
            summary.append(record)
            if self.on_node_finished:
                self.on_node_finished(record)

            if spec.failure_policy == "continue":
                self._append_error(state, f"{spec.name}: {error_text} (continued)")
            else:
                self._append_error(state, f"{spec.name}: {error_text}")
                terminal_failure = True

        return state, summary

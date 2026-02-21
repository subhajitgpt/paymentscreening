"""A tiny agentic-flow framework for deterministic pipelines.

This repo's v4 UI describes an "agentic-style" flow (sample → algorithms → match → scoring → decision
→ explain). This module packages that concept into reusable primitives:

- Flow + Step: linear, observable execution with retries/timeouts
- Context: shared state passing between steps
- ToolRegistry: optional tool-calling abstraction (deterministic, no LLM)
- Audit events: append-only JSONL audit trail compatible with audit_trail.jsonl

This is intentionally dependency-free (stdlib only) so it can be reused from the Flask UI,
REST API, batch job runner, or CLI scripts.

Example
-------

    from agentic_framework import (
        Agent, Context, Flow, Step, JsonlAuditSink, StepResult, ToolRegistry
    )

    def step_sample(ctx: Context) -> StepResult:
        payload = ctx.input
        ctx.audit("sample_data", "Received input payload", {"keys": sorted(payload.keys())})
        return StepResult.ok()

    flow = Flow(
        name="payment_screening_flow",
        steps=[Step(name="sample_data", fn=step_sample)],
    )

    agent = Agent(flow=flow, audit_sink=JsonlAuditSink("audit_trail.jsonl"))
    result = agent.run({"payer_name": "Alice", "benef_name": "Bob"})

"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Protocol, Tuple


def utc_now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


# -----------------------------
# Audit trail
# -----------------------------


@dataclass(frozen=True)
class AuditEvent:
    ts: str
    job_id: str
    item_index: int
    step: str
    message: str
    data: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {
                "ts": self.ts,
                "job_id": self.job_id,
                "item_index": self.item_index,
                "step": self.step,
                "message": self.message,
                "data": self.data,
            },
            ensure_ascii=False,
        )


class AuditSink(Protocol):
    def emit(self, event: AuditEvent) -> None:  # pragma: no cover
        ...


class InMemoryAuditSink:
    """Thread-safe in-memory audit sink."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: List[AuditEvent] = []

    def emit(self, event: AuditEvent) -> None:
        with self._lock:
            self._events.append(event)

    def events(self) -> List[AuditEvent]:
        with self._lock:
            return list(self._events)


class JsonlAuditSink:
    """Append-only JSONL audit sink (best-effort)."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._lock = threading.Lock()

    def emit(self, event: AuditEvent) -> None:
        line = event.to_json() + "\n"
        directory = os.path.dirname(os.path.abspath(self.path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with self._lock:
            try:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line)
            except Exception:
                # Best-effort audit: never crash business logic due to logging.
                pass


class CompositeAuditSink:
    def __init__(self, sinks: Iterable[AuditSink]):
        self._sinks = list(sinks)

    def emit(self, event: AuditEvent) -> None:
        for sink in self._sinks:
            sink.emit(event)


# -----------------------------
# Flow execution model
# -----------------------------


@dataclass
class StepResult:
    """Outcome of a step.

    - ok=True continues execution.
    - ok=False stops the flow (unless caller overrides).
    """

    ok: bool
    output: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @staticmethod
    def ok_result(output: Optional[Mapping[str, Any]] = None) -> "StepResult":
        return StepResult(ok=True, output=dict(output or {}), error=None)

    @staticmethod
    def fail(error: str, output: Optional[Mapping[str, Any]] = None) -> "StepResult":
        return StepResult(ok=False, output=dict(output or {}), error=error)

    # Backwards-friendly aliases (nice ergonomics)
    @staticmethod
    def ok(output: Optional[Mapping[str, Any]] = None) -> "StepResult":
        return StepResult.ok_result(output)


@dataclass
class Context:
    """Shared state passed through a flow."""

    job_id: str
    item_index: int
    input: Dict[str, Any]
    state: Dict[str, Any] = field(default_factory=dict)
    audit_sink: Optional[AuditSink] = None

    def audit(self, step: str, message: str, data: Optional[Mapping[str, Any]] = None) -> None:
        if not self.audit_sink:
            return
        event = AuditEvent(
            ts=utc_now_iso(),
            job_id=self.job_id,
            item_index=self.item_index,
            step=step,
            message=message,
            data=dict(data or {}),
        )
        self.audit_sink.emit(event)


StepFn = Callable[[Context], StepResult]


@dataclass
class Step:
    name: str
    fn: StepFn
    retries: int = 0
    retry_delay_s: float = 0.0
    timeout_s: Optional[float] = None


class FlowError(RuntimeError):
    pass


@dataclass
class FlowRunResult:
    ok: bool
    job_id: str
    item_index: int
    step_results: Dict[str, StepResult] = field(default_factory=dict)
    final_state: Dict[str, Any] = field(default_factory=dict)
    error_step: Optional[str] = None
    error: Optional[str] = None


class Flow:
    def __init__(self, name: str, steps: List[Step]) -> None:
        if not name:
            raise ValueError("Flow name is required")
        if not steps:
            raise ValueError("Flow must contain at least one step")
        self.name = name
        self.steps = steps

    def run(self, ctx: Context) -> FlowRunResult:
        ctx.audit("flow", f"Starting flow: {self.name}", {"steps": [s.name for s in self.steps]})

        step_results: Dict[str, StepResult] = {}
        for step in self.steps:
            res = self._run_step(ctx, step)
            step_results[step.name] = res

            # Merge step outputs into context state for convenient chaining.
            if res.output:
                ctx.state.update(res.output)

            if not res.ok:
                ctx.audit(step.name, "Step failed; stopping flow", {"error": res.error})
                return FlowRunResult(
                    ok=False,
                    job_id=ctx.job_id,
                    item_index=ctx.item_index,
                    step_results=step_results,
                    final_state=dict(ctx.state),
                    error_step=step.name,
                    error=res.error,
                )

        ctx.audit("flow", f"Completed flow: {self.name}", {"ok": True})
        return FlowRunResult(
            ok=True,
            job_id=ctx.job_id,
            item_index=ctx.item_index,
            step_results=step_results,
            final_state=dict(ctx.state),
        )

    def _run_step(self, ctx: Context, step: Step) -> StepResult:
        attempts = 0
        max_attempts = 1 + max(0, step.retries)

        while attempts < max_attempts:
            attempts += 1
            ctx.audit(step.name, "Starting step", {"attempt": attempts, "max_attempts": max_attempts})

            start = time.time()
            try:
                if step.timeout_s is None:
                    result = step.fn(ctx)
                else:
                    result = _run_with_timeout(step.fn, ctx, timeout_s=step.timeout_s)

                duration_ms = int((time.time() - start) * 1000)
                ctx.audit(step.name, "Finished step", {"ok": result.ok, "duration_ms": duration_ms})
                return result

            except TimeoutError as e:
                duration_ms = int((time.time() - start) * 1000)
                ctx.audit(step.name, "Step timed out", {"duration_ms": duration_ms, "timeout_s": step.timeout_s})
                err = f"timeout: {e}"

            except Exception as e:
                duration_ms = int((time.time() - start) * 1000)
                ctx.audit(step.name, "Step exception", {"duration_ms": duration_ms, "type": type(e).__name__})
                err = f"exception: {type(e).__name__}: {e}"

            if attempts < max_attempts:
                if step.retry_delay_s > 0:
                    time.sleep(step.retry_delay_s)
                ctx.audit(step.name, "Retrying step", {"attempt": attempts + 1})
                continue

            return StepResult.fail(err)


def _run_with_timeout(fn: StepFn, ctx: Context, timeout_s: float) -> StepResult:
    """Runs a step with a wall-clock timeout.

    Uses a thread to avoid external dependencies. Note: the underlying function is not
    forcefully killed; we just stop waiting.
    """

    result_box: Dict[str, Any] = {}
    error_box: Dict[str, Any] = {}

    def runner() -> None:
        try:
            result_box["result"] = fn(ctx)
        except Exception as e:
            error_box["error"] = e

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    t.join(timeout=timeout_s)

    if t.is_alive():
        raise TimeoutError(f"step exceeded {timeout_s}s")

    if "error" in error_box:
        raise error_box["error"]

    return result_box["result"]


# -----------------------------
# Tools + Agent wrapper
# -----------------------------


ToolFn = Callable[[Context, Dict[str, Any]], Dict[str, Any]]


@dataclass(frozen=True)
class Tool:
    name: str
    fn: ToolFn
    description: str = ""


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name]

    def call(self, ctx: Context, name: str, args: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        tool = self.get(name)
        ctx.audit("tool", f"Calling tool: {name}", {"args": dict(args or {})})
        out = tool.fn(ctx, dict(args or {}))
        ctx.audit("tool", f"Finished tool: {name}", {"keys": sorted(out.keys())})
        return out

    def list(self) -> List[Dict[str, str]]:
        return [
            {"name": t.name, "description": t.description}
            for t in sorted(self._tools.values(), key=lambda x: x.name)
        ]


class Agent:
    """Thin wrapper around a Flow.

    This keeps the concept of "agent run" (job_id + input + audit) separate from the flow definition.
    """

    def __init__(
        self,
        flow: Flow,
        audit_sink: Optional[AuditSink] = None,
        tools: Optional[ToolRegistry] = None,
        job_id_prefix: str = "sync",
    ) -> None:
        self.flow = flow
        self.audit_sink = audit_sink
        self.tools = tools or ToolRegistry()
        self.job_id_prefix = job_id_prefix

    def new_job_id(self) -> str:
        return f"{self.job_id_prefix}-{uuid.uuid4().hex}"

    def run(
        self,
        payload: Mapping[str, Any],
        *,
        item_index: int = 0,
        job_id: Optional[str] = None,
        initial_state: Optional[Mapping[str, Any]] = None,
    ) -> FlowRunResult:
        ctx = Context(
            job_id=job_id or self.new_job_id(),
            item_index=item_index,
            input=dict(payload),
            state=dict(initial_state or {}),
            audit_sink=self.audit_sink,
        )
        ctx.state.setdefault("tools", self.tools)
        return self.flow.run(ctx)


# -----------------------------
# Minimal demo flow (optional)
# -----------------------------


def demo_flow() -> Flow:
    def sample_data(ctx: Context) -> StepResult:
        ctx.audit("sample_data", "Received input payload", {"keys": sorted(ctx.input.keys())})
        return StepResult.ok()

    def algorithms(ctx: Context) -> StepResult:
        ctx.audit(
            "algorithms",
            "Prepared normalization + similarity algorithms",
            {"algorithms": ["normalize_text", "Jaro-Winkler", "Jaccard"]},
        )
        return StepResult.ok()

    def decision(ctx: Context) -> StepResult:
        # Dummy decision for demo purposes.
        amount = float(ctx.input.get("amount") or 0)
        decision = "ESCALATE" if amount >= 10_000 else "RELEASE"
        reason = "High Amount" if decision == "ESCALATE" else "Below Threshold"
        ctx.audit("decision", "Applied automated decision policy", {"decision": decision, "reason": reason})
        return StepResult.ok({"decision": decision, "reason": reason})

    return Flow(
        name="demo_agentic_flow",
        steps=[
            Step(name="sample_data", fn=sample_data),
            Step(name="algorithms", fn=algorithms),
            Step(name="decision", fn=decision),
        ],
    )


if __name__ == "__main__":
    # Run a small demo and write audit events.
    audit = JsonlAuditSink("audit_trail.jsonl")
    agent = Agent(flow=demo_flow(), audit_sink=audit)

    payload = {
        "amount": 12500,
        "currency": "USD",
        "payer_name": "Global Trade LLC",
        "benef_name": "Olena Petrenko",
        "reference": "INV-1001",
    }

    r = agent.run(payload)
    print(json.dumps({"ok": r.ok, "job_id": r.job_id, "final_state": r.final_state}, indent=2))

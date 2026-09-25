"""The agentic loop.

    UNDERSTAND -> PLAN -> ACT -> OBSERVE -> EVALUATE -> ADAPT -> ... -> COMPLETE

The loop is deliberately written as data (a ``RunResult`` transcript) rather
than as a chat: every step is recorded with its tool calls, results, timings
and errors, so the run can be inspected, persisted, and resumed.

What makes this a real agent loop rather than "one model call per message":

* multiple tool calls per turn, executed concurrently when they are independent
* every tool result is fed back before the next decision
* failures become structured observations the model can reason about
* repeated *identical* failures trigger an explicit strategy change, not a retry
* the scratchpad (plan/facts/blockers) survives context trimming
* hard step and wall-clock budgets bound the whole run
* the loop ends only on a final answer, a budget, or a genuine block
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .errors import (
    AgentError,
    ApprovalRequired,
    BudgetExceeded,
    ModelUnavailableError,
    PermissionDenied,
    RateLimited,
)
from .events import EventLog, log as default_log
from .llm import ModelRouter
from .memory import (
    Conversation,
    LongTermMemory,
    MemoryBundle,
    TaskRecord,
    TaskStore,
    WorkingState,
    new_id,
)
from .prompt import (
    build_budget_nudge,
    build_failure_nudge,
    build_loop_break_nudge,
    build_objective_message,
    build_step_context,
    build_system_prompt,
)
from .redaction import redactor
from .registry import ToolContext, ToolRegistry, build_default_registry
from .security import ApprovalGate, Authorizer, PathGuard, Principal, RateLimiter
from .skills import SkillLoader

log = logging.getLogger(__name__)

MAX_IDENTICAL_FAILURES = 3


@dataclass
class StepRecord:
    index: int
    started_at: float
    assistant_text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    duration_ms: int = 0
    model: str = ""
    usage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "assistant_text": (self.assistant_text or "")[:600],
            "tool_calls": self.tool_calls,
            "duration_ms": self.duration_ms,
            "model": self.model,
            "usage": self.usage,
        }


@dataclass
class RunResult:
    """Everything that happened during one objective."""

    task_id: str
    objective: str
    status: str = "running"  # completed | failed | blocked | budget_exceeded
    answer: str = ""
    steps: list[StepRecord] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    error: str | None = None
    duration_ms: int = 0
    usage: dict[str, int] = field(default_factory=dict)
    stop_reason: str = ""
    needs_approval: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "objective": self.objective,
            "status": self.status,
            "answer": self.answer,
            "stop_reason": self.stop_reason,
            "error": self.error,
            "needs_approval": self.needs_approval,
            "duration_ms": self.duration_ms,
            "usage": self.usage,
            "artifacts": self.artifacts,
            "steps": [s.to_dict() for s in self.steps],
        }


class Agent:
    """Owns the loop. One instance is safe to reuse across tasks."""

    def __init__(
        self,
        *,
        router: ModelRouter,
        settings: Any,
        registry: ToolRegistry | None = None,
        skills: SkillLoader | None = None,
        authorizer: Authorizer | None = None,
        rate_limiter: RateLimiter | None = None,
        gate: ApprovalGate | None = None,
        events: EventLog | None = None,
        memory: MemoryBundle | None = None,
        sandbox: Any | None = None,
    ) -> None:
        self.settings = settings
        self.router = router
        self.registry = registry or build_default_registry()
        self.skills = skills or SkillLoader()
        self.authorizer = authorizer or Authorizer(settings.telegram.allowed_users)
        self.rate_limiter = rate_limiter or RateLimiter(settings.telegram.max_requests_per_minute)
        self.gate = gate or ApprovalGate(allow_destructive=settings.sandbox.allow_destructive)
        self.events = events or default_log()
        self.path_guard = PathGuard(settings.workspace_root)
        self.sandbox = sandbox
        self._memory = memory
        self._running: dict[str, str] = {}
        # Tasks are serialised on purpose: this runtime is a single-user agent on a
        # 0.1-CPU free instance, and the working scratchpad is shared. Concurrent
        # runs would interleave one task's plan into another's.
        self._run_lock = asyncio.Lock()

    # -- wiring ---------------------------------------------------------

    def attach_memory(self, memory: MemoryBundle) -> None:
        self._memory = memory

    @property
    def memory(self) -> MemoryBundle:
        if self._memory is None:
            raise RuntimeError("agent has no memory bundle attached")
        return self._memory

    def describe(self) -> dict[str, Any]:
        """Live capability summary; the source of truth for 'what can you do'."""
        return {
            "tools": self.registry.names(),
            "skills": self.skills.names(),
            "model": self.router.health(),
            "running_tasks": dict(self._running),
        }

    # -- public entry point ---------------------------------------------

    async def run(
        self,
        objective: str,
        *,
        principal: Principal | None = None,
        channel: str = "cli",
        approval_token: str | None = None,
        resume_task_id: str | None = None,
    ) -> RunResult:
        """Execute one objective to completion, block, or budget exhaustion."""
        principal = principal or Principal("system", "local")
        objective = (objective or "").strip()
        if not objective:
            raise ValueError("objective must not be empty")

        # --- gate 1: authorisation and rate limit (fail closed) --------
        self.authorizer.check(principal)
        self.rate_limiter.check(principal)

        task_id = resume_task_id or new_id("task")
        record = (await self.memory.tasks.get(task_id)) if resume_task_id else None
        if record is None:
            record = TaskRecord(
                id=task_id,
                objective=objective,
                principal=str(principal),
                channel=channel,
                status="running",
            )
        else:
            record.status = "running"
        await self.memory.tasks.save(record)

        self._running[task_id] = objective[:120]
        self.events.emit(
            "task.start",
            task_id,
            objective=objective[:400],
            principal=str(principal),
            channel=channel,
        )

        started = time.monotonic()
        deadline = started + self.settings.agent.max_seconds
        result = RunResult(task_id=task_id, objective=objective)

        try:
            async with self._run_lock:
                # A fresh objective gets a fresh scratchpad unless we are resuming.
                # Done under the lock so a queued task cannot clobber a running one.
                if not resume_task_id:
                    self.memory.working = WorkingState(objective=objective)
                await self._run_loop(
                    objective=objective,
                    task_id=task_id,
                    principal=principal,
                    channel=channel,
                    approval_token=approval_token,
                    deadline=deadline,
                    result=result,
                )
        except ApprovalRequired as exc:
            result.status = "blocked"
            result.needs_approval = exc.message
            result.answer = (
                f"I stopped because this action needs your explicit approval:\n\n{exc.message}"
            )
            result.stop_reason = "approval_required"
            self.events.emit("task.blocked", task_id, reason="approval_required", detail=exc.message)
        except (PermissionDenied, RateLimited) as exc:
            result.status = "failed"
            result.error = exc.message
            result.answer = f"I could not start this task: {exc.message}"
            result.stop_reason = exc.code
            self.events.emit("task.refused", task_id, code=exc.code, detail=exc.message)
        except AgentError as exc:
            result.status = "failed"
            result.error = exc.message
            result.answer = f"The task failed: {exc.message}"
            result.stop_reason = exc.code
            self.events.emit("task.error", task_id, code=exc.code, detail=exc.message)
        except Exception as exc:  # noqa: BLE001 - a task must never crash the process
            log.exception("unhandled failure in agent loop")
            result.status = "failed"
            result.error = f"{type(exc).__name__}: {exc}"
            result.answer = f"The task failed unexpectedly: {result.error}"
            result.stop_reason = "unhandled_error"
            self.events.emit("task.error", task_id, code="unhandled_error", detail=str(exc)[:400])

        result.duration_ms = int((time.monotonic() - started) * 1000)

        # --- persist the outcome ---------------------------------------
        # A budget-exhausted run is recorded as 'running' on purpose: it was
        # interrupted, not concluded, so it stays visible in `unfinished()` and
        # the operator can resume it after a restart.
        record.status = {
            "completed": "completed",
            "failed": "failed",
            "blocked": "blocked",
            "budget_exceeded": "running",
        }.get(result.status, "failed")
        record.result = result.answer[:4000]
        record.error = result.error
        record.artifacts = list(result.artifacts)
        record.steps_taken = len(result.steps)
        record.tool_calls = [
            call.get("name", "") for step in result.steps for call in step.tool_calls
        ]
        record.finished_at = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
        await self.memory.tasks.save(record)

        self._running.pop(task_id, None)
        self.events.emit(
            "task.finish",
            task_id,
            status=result.status,
            stop_reason=result.stop_reason,
            steps=len(result.steps),
            duration_ms=result.duration_ms,
            artifacts=result.artifacts,
        )
        return result

    # -- the loop --------------------------------------------------------

    async def _run_loop(
        self,
        *,
        objective: str,
        task_id: str,
        principal: Principal,
        channel: str,
        approval_token: str | None,
        deadline: float,
        result: RunResult,
    ) -> None:
        memory = self.memory
        working: WorkingState = memory.working
        if not working.objective:
            working.objective = objective
        conversation: Conversation = memory.conversation

        recalled = await memory.long_term.recall(objective, limit=6)
        facts = [f"{e.text}" for e in recalled]

        system_prompt = build_system_prompt(
            workspace=str(self.path_guard.root),
            skills_index=self.skills.index(),
            tool_index=self._tool_index(),
            persisted_facts=facts,
            started_at=datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        )

        unfinished = await memory.tasks.unfinished()
        hint = None
        others = [t for t in unfinished if t.get("id") != task_id]
        if others:
            hint = (
                "Note: these earlier tasks were interrupted by a restart and never finished — "
                "mention them only if relevant: "
                + "; ".join(f"{t.get('id')}: {str(t.get('objective'))[:80]}" for t in others[:3])
            )

        conversation.append({"role": "system", "content": system_prompt})
        conversation.append(
            {
                "role": "user",
                "content": build_objective_message(
                    objective, channel=channel, principal=str(principal), unfinished_hint=hint
                ),
            }
        )

        ctx = ToolContext(
            workspace_root=self.path_guard.root,
            path_guard=self.path_guard,
            sandbox=self.sandbox,
            state=memory.store,
            memory=memory,
            skills=self.skills,
            gate=self.gate,
            settings=self.settings,
            task_id=task_id,
            principal=str(principal),
            approval_token=approval_token,
            extra={"router": self.router, "registry": self.registry},
        )

        tool_specs = self.registry.specs()
        failure_counts: dict[str, int] = {}
        nudges: list[str] = []
        max_steps = self.settings.agent.max_steps

        for step in range(1, max_steps + 1):
            elapsed = int(time.monotonic() - (deadline - self.settings.agent.max_seconds))
            if time.monotonic() >= deadline:
                result.status = "budget_exceeded"
                result.stop_reason = "wall_clock_budget"
                result.answer = self._budget_answer(result, "wall-clock budget")
                return

            working.step = step
            context_note = conversation.context_note()
            if step == max_steps:
                nudges.append(build_loop_break_nudge(max_steps))
            elif max_steps - step == 3:
                nudges.append(build_budget_nudge(3))

            conversation.append(
                {
                    "role": "user",
                    "content": build_step_context(
                        working_state=working.to_prompt(),
                        step=step,
                        max_steps=max_steps,
                        elapsed_seconds=elapsed,
                        max_seconds=self.settings.agent.max_seconds,
                        context_note=context_note,
                        nudges=list(nudges),
                    ),
                }
            )
            nudges.clear()

            step_record = StepRecord(index=step, started_at=time.monotonic())
            self.events.emit("step.begin", task_id, step=step, elapsed_s=elapsed)

            # ---- PLAN / ACT: one model decision ----------------------
            try:
                response = await self.router.chat(conversation.messages, tools=tool_specs)
            except ModelUnavailableError as exc:
                result.status = "failed"
                result.error = f"no usable model: {exc.message}"
                result.answer = (
                    "I could not run: no model in the configured chain is currently usable. "
                    f"Detail: {exc.message}"
                )
                result.stop_reason = "model_unavailable"
                self.events.emit("model.failed", task_id, step=step, detail=exc.message)
                return

            step_record.model = response.model
            step_record.usage = dict(response.usage or {})
            result.usage["prompt_tokens"] = result.usage.get("prompt_tokens", 0) + int(
                (response.usage or {}).get("prompt_tokens", 0)
            )
            result.usage["completion_tokens"] = result.usage.get("completion_tokens", 0) + int(
                (response.usage or {}).get("completion_tokens", 0)
            )
            self.events.emit(
                "model.call",
                task_id,
                step=step,
                model=response.model,
                tool_calls=len(response.tool_calls),
                usage=response.usage,
            )

            # ---- COMPLETE: a turn with no tool calls is the final answer --
            if not response.wants_tools:
                answer = (response.content or "").strip()
                step_record.assistant_text = answer
                step_record.duration_ms = int((time.monotonic() - step_record.started_at) * 1000)
                # Record the step either way, so the transcript shows the turn
                # the model spent even when it produced nothing usable.
                result.steps.append(step_record)
                if not answer:
                    nudges.append(
                        "Your last message was empty. Either call a tool or write the final answer."
                    )
                    self.events.emit("step.end", task_id, step=step, final=False, empty=True)
                    continue
                result.status = "completed"
                result.answer = answer
                result.stop_reason = "final_answer"
                conversation.append({"role": "assistant", "content": answer})
                self.events.emit("step.end", task_id, step=step, final=True)
                return

            conversation.append(
                {
                    "role": "assistant",
                    "content": response.content or "",
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {"name": call.name, "arguments": call.arguments},
                        }
                        for call in response.tool_calls
                    ],
                }
            )
            step_record.assistant_text = response.content or ""

            # ---- ACT: execute the requested tools ---------------------
            outcomes = await self._execute_calls(response.tool_calls, ctx, task_id, step)

            for call, tool_result in outcomes:
                step_record.tool_calls.append(
                    {
                        "name": call.name,
                        "arguments": redactor().scrub_deep(call.arguments),
                        "ok": tool_result.ok,
                        "code": tool_result.code,
                        "duration_ms": tool_result.duration_ms,
                    }
                )
                conversation.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.name,
                        "content": tool_result.to_message(),
                    }
                )

                # ---- EVALUATE / ADAPT -------------------------------
                if tool_result.ok:
                    failure_counts.pop(call.name, None)
                    if call.name == "write_artifact":
                        path = str((tool_result.content or {}).get("path", ""))
                        if path and path not in result.artifacts:
                            result.artifacts.append(path)
                else:
                    if tool_result.code == ApprovalRequired.code:
                        raise ApprovalRequired(tool_result.error or f"{call.name} needs approval")
                    key = f"{call.name}:{(tool_result.error or '')[:120]}"
                    failure_counts[key] = failure_counts.get(key, 0) + 1
                    if failure_counts[key] >= MAX_IDENTICAL_FAILURES:
                        nudges.append(
                            build_failure_nudge(
                                call.name, failure_counts[key], tool_result.error or ""
                            )
                        )

            step_record.duration_ms = int((time.monotonic() - step_record.started_at) * 1000)
            result.steps.append(step_record)
            self.events.emit(
                "step.end",
                task_id,
                step=step,
                tools=[c["name"] for c in step_record.tool_calls],
                duration_ms=step_record.duration_ms,
            )

        # Ran out of steps without a final answer.
        result.status = "budget_exceeded"
        result.stop_reason = "step_budget"
        result.answer = self._budget_answer(result, f"step budget ({max_steps} steps)")

    # -- helpers ---------------------------------------------------------

    async def _execute_calls(self, calls, ctx: ToolContext, task_id: str, step: int):
        """Execute this turn's tool calls.

        Independent calls run concurrently; that is safe because every tool is
        confined to the workspace and the registry caps its own timeout. Calls
        that touch the same file would be ordered by the model in separate
        turns, which is the honest way to express a dependency.
        """
        async def one(call):
            self.events.emit(
                "tool.call", task_id, step=step, tool=call.name,
                args=self.registry.get(call.name).preview(call.arguments)
                if call.name in self.registry.names() else {},
            )
            outcome = await self.registry.execute(call.name, call.arguments, ctx)
            self.events.emit(
                "tool.result", task_id, step=step, tool=call.name,
                ok=outcome.ok, code=outcome.code, duration_ms=outcome.duration_ms,
            )
            return call, outcome

        return list(await asyncio.gather(*(one(c) for c in calls)))

    def _tool_index(self) -> str:
        catalog = self.registry.catalog()
        by_category: dict[str, list[dict[str, Any]]] = {}
        for item in catalog:
            by_category.setdefault(item["category"], []).append(item)
        lines: list[str] = []
        for category in sorted(by_category):
            lines.append(f"### {category}")
            for item in sorted(by_category[category], key=lambda i: i["name"]):
                flag = " [privileged]" if item["privileged"] else ""
                lines.append(f"- {item['name']}{flag}: {item['description']}")
        return "\n".join(lines)

    @staticmethod
    def _budget_answer(result: RunResult, which: str) -> str:
        done = [f"- step {s.index}: " + ", ".join(c["name"] for c in s.tool_calls) for s in result.steps[-6:]]
        tail = "\n".join(done) if done else "(no tool calls were made)"
        return (
            f"I ran out of my {which} before finishing the objective.\n\n"
            f"Recent activity:\n{tail}\n\n"
            f"Artifacts produced so far: {', '.join(result.artifacts) or 'none'}.\n"
            f"Ask me to continue and I will resume from the recorded state."
        )
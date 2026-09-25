"""State tools — how the agent remembers things across steps and restarts.

These are the tools that make long-horizon work possible:

* ``working_memory`` — edit the plan / facts / blockers for the *current* task
* ``long_term_memory`` — remember or recall facts across tasks and restarts
* ``task_state`` — inspect and update the durable task record
* ``write_artifact`` — declare a file as a deliverable and record its URL
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Mapping

from ..errors import ToolError
from ..memory import TaskRecord
from ..registry import Tool, ToolContext, prop, schema

log = logging.getLogger(__name__)


class WorkingMemoryTool(Tool):
    name = "working_memory"
    category = "state"
    description = (
        "Read or update your scratchpad for the current task: the plan, verified facts, "
        "blockers and notes. This is external, durable memory — use it to keep orientation "
        "on long tasks and after context is trimmed. Actions: 'read' | 'set_plan' | "
        "'complete_step' | 'add_fact' | 'add_blocker' | 'clear_blocker' | 'add_note' | "
        "'set_finish_condition'."
    )

    @property
    def parameters(self) -> dict[str, Any]:
        return schema(
            {
                "action": prop(
                    "string",
                    "What to do.",
                    enum=[
                        "read", "set_plan", "complete_step", "add_fact",
                        "add_blocker", "clear_blocker", "add_note", "set_finish_condition",
                    ],
                ),
                "value": prop("string", "The text to add (for add_*/set_* actions).", default=""),
                "step_index": prop("integer", "1-based plan step to mark done.", minimum=1),
                "plan": prop("array", "Full list of plan steps (for set_plan)."),
            },
            required=["action"],
            description="Manage the current task scratchpad.",
        )

    async def run(self, args: Mapping[str, Any], ctx: ToolContext) -> dict[str, Any]:
        working = ctx.memory.working
        action = args["action"]
        value = str(args.get("value") or "").strip()

        if action == "read":
            pass
        elif action == "set_plan":
            plan = args.get("plan") or []
            if isinstance(plan, str):
                plan = [line.strip() for line in plan.splitlines() if line.strip()]
            if not isinstance(plan, (list, tuple)):
                raise ToolError("plan must be a list of strings")
            working.plan = [str(p).strip() for p in plan if str(p).strip()][:15]
            ctx.memory.working = working
            await self._persist(ctx)
        elif action == "complete_step":
            index = args.get("step_index")
            if not isinstance(index, int) or index < 1 or index > len(working.plan):
                raise ToolError(
                    f"step_index must be between 1 and {len(working.plan)} "
                    f"(the current plan has {len(working.plan)} steps)"
                )
            working.plan[index - 1] = f"[done] {working.plan[index - 1].replace('[done]', '').strip()}"
            ctx.memory.working = working
            await self._persist(ctx)
        elif action == "add_fact":
            if not value:
                raise ToolError("value is required for add_fact")
            working.facts.append(value)
            working.facts = working.facts[-25:]
            ctx.memory.working = working
            await self._persist(ctx)
        elif action == "add_blocker":
            if not value:
                raise ToolError("value is required for add_blocker")
            working.blockers.append(value)
            ctx.memory.working = working
            await self._persist(ctx)
        elif action == "clear_blocker":
            if value:
                working.blockers = [b for b in working.blockers if value.lower() not in b.lower()]
            else:
                working.blockers = []
            ctx.memory.working = working
            await self._persist(ctx)
        elif action == "add_note":
            if not value:
                raise ToolError("value is required for add_note")
            working.notes.append(value)
            working.notes = working.notes[-12:]
            ctx.memory.working = working
            await self._persist(ctx)
        elif action == "set_finish_condition":
            if not value:
                raise ToolError("value is required for set_finish_condition")
            working.finish_condition = value
            ctx.memory.working = working
            await self._persist(ctx)
        else:
            raise ToolError(f"unknown action: {action}")

        return {
            "action": action,
            "state": working.to_dict(),
            "rendered": working.to_prompt(),
        }

    @staticmethod
    async def _persist(ctx: ToolContext) -> None:
        if ctx.task_id and ctx.task_id != "system":
            record = await ctx.memory.tasks.get(ctx.task_id)
            if record is not None:
                record.plan = list(ctx.memory.working.plan)
                await ctx.memory.tasks.save(record)


class LongTermMemoryTool(Tool):
    name = "long_term_memory"
    category = "state"
    description = (
        "Persist or retrieve knowledge that should outlive this task and survive restarts — "
        "user preferences, environment facts, lessons learned. Actions: 'remember' | 'recall' | "
        "'forget' | 'stats'. Keep entries short and factual."
    )

    @property
    def parameters(self) -> dict[str, Any]:
        return schema(
            {
                "action": prop("string", "Operation to perform.", enum=["remember", "recall", "forget", "stats"]),
                "text": prop("string", "The fact to store, or the search query when recalling.", default=""),
                "kind": prop(
                    "string",
                    "Classification of the memory.",
                    enum=["fact", "preference", "lesson", "artifact", "summary"],
                    default="fact",
                ),
                "importance": prop("integer", "1 (trivia) to 5 (critical).", minimum=1, maximum=5, default=3),
                "entry_id": prop("string", "Entry id to forget.", default=""),
                "limit": prop("integer", "Maximum entries to return when recalling.", minimum=1, maximum=50, default=10),
            },
            required=["action"],
            description="Manage long-term memory.",
        )

    async def run(self, args: Mapping[str, Any], ctx: ToolContext) -> dict[str, Any]:
        memory = ctx.memory.long_term
        action = args["action"]

        if action == "remember":
            text = str(args.get("text") or "").strip()
            if not text:
                raise ToolError("text is required for remember")
            entry = await memory.remember(
                text,
                kind=args.get("kind", "fact"),
                task_id=ctx.task_id,
                importance=int(args.get("importance", 3)),
            )
            return {"stored": entry.to_dict(), "total": await memory.count()}

        if action == "recall":
            entries = await memory.recall(str(args.get("text") or ""), int(args.get("limit", 10)))
            return {
                "query": args.get("text") or "",
                "count": len(entries),
                "entries": [e.to_dict() for e in entries],
            }

        if action == "forget":
            entry_id = str(args.get("entry_id") or "").strip()
            if not entry_id:
                raise ToolError("entry_id is required for forget")
            return {"forgotten": await memory.forget(entry_id)}

        if action == "stats":
            entries = await memory.recall("", limit=1000)
            by_kind: dict[str, int] = {}
            for entry in entries:
                by_kind[entry.kind] = by_kind.get(entry.kind, 0) + 1
            return {"total": len(entries), "by_kind": by_kind}

        raise ToolError(f"unknown action: {action}")


class TaskStateTool(Tool):
    name = "task_state"
    category = "state"
    description = (
        "Inspect and update the durable record of the current task — its status, progress "
        "log, artifacts and result. Also lists unfinished tasks from previous runs so "
        "interrupted work can be resumed. Actions: 'read' | 'progress' | 'set_status' | "
        "'add_artifact' | 'list_recent' | 'list_unfinished'."
    )

    @property
    def parameters(self) -> dict[str, Any]:
        return schema(
            {
                "action": prop(
                    "string",
                    "Operation to perform.",
                    enum=["read", "progress", "set_status", "add_artifact", "list_recent", "list_unfinished"],
                ),
                "value": prop("string", "Progress note, status value, or artifact path.", default=""),
                "status": prop(
                    "string",
                    "New status when using set_status.",
                    enum=["pending", "running", "completed", "failed", "blocked"],
                ),
                "limit": prop("integer", "How many records to return.", minimum=1, maximum=50, default=10),
            },
            required=["action"],
            description="Inspect or update the durable task record.",
        )

    async def run(self, args: Mapping[str, Any], ctx: ToolContext) -> dict[str, Any]:
        tasks = ctx.memory.tasks
        action = args["action"]

        if action == "list_recent":
            return {"tasks": await tasks.recent(int(args.get("limit", 10)))}

        if action == "list_unfinished":
            return {"tasks": await tasks.unfinished()}

        record = await tasks.get(ctx.task_id) if ctx.task_id != "system" else None
        if record is None:
            record = TaskRecord(id=ctx.task_id, objective=ctx.memory.working.objective, principal=ctx.principal)

        if action == "read":
            return {"task": record.to_dict()}

        if action == "progress":
            value = str(args.get("value") or "").strip()
            if not value:
                raise ToolError("value is required for progress")
            record.progress.append(f"[{time.strftime('%H:%M:%S')}] {value}")
            record.progress = record.progress[-40:]
            record.steps_taken += 1
            await tasks.save(record)
            return {"task_id": record.id, "steps_taken": record.steps_taken, "progress": record.progress[-5:]}

        if action == "set_status":
            status = args.get("status")
            if not status:
                raise ToolError("status is required for set_status")
            record.status = status
            if status in ("completed", "failed"):
                record.finished_at = record.updated_at
            if args.get("value"):
                record.result = str(args["value"])
            await tasks.save(record)
            return {"task_id": record.id, "status": record.status}

        if action == "add_artifact":
            value = str(args.get("value") or "").strip()
            if not value:
                raise ToolError("value is required for add_artifact")
            if value not in record.artifacts:
                record.artifacts.append(value)
            await tasks.save(record)
            return {"task_id": record.id, "artifacts": record.artifacts}

        raise ToolError(f"unknown action: {action}")


class ArtifactWriteTool(Tool):
    name = "write_artifact"
    category = "state"
    description = (
        "Write a deliverable file and record it on the task so it is reported as a produced "
        "artifact. Use this (rather than write_file) for anything the user is meant to receive. "
        "Files live under the workspace 'artifacts/' directory unless a path is given."
    )

    @property
    def parameters(self) -> dict[str, Any]:
        return schema(
            {
                "name": prop("string", "File name, e.g. 'report.md' or 'data/results.csv'."),
                "content": prop("string", "The file content."),
                "description": prop("string", "One line describing what this artifact is.", default=""),
            },
            required=["name", "content"],
            description="Produce a deliverable artifact.",
        )

    async def run(self, args: Mapping[str, Any], ctx: ToolContext) -> dict[str, Any]:
        raw = str(args["name"]).strip().lstrip("/")
        if not raw or ".." in raw:
            raise ToolError("artifact name must be a simple relative path")
        target = ctx.path_guard.resolve(str(Path("artifacts") / raw))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(args["content"], encoding="utf-8")
        size = target.stat().st_size
        relative = ctx.path_guard.relative(target)

        record = await ctx.memory.tasks.get(ctx.task_id) if ctx.task_id != "system" else None
        if record is not None:
            if relative not in record.artifacts:
                record.artifacts.append(relative)
            await ctx.memory.tasks.save(record)

        return {
            "path": relative,
            "bytes": size,
            "description": args.get("description", ""),
            "verified_exists": target.exists(),
            "recorded_on_task": record is not None,
        }

"""Memory and state.

Five *different* things are often collapsed into the word "memory". Keeping
them separate is what makes long tasks and restarts survivable:

===================  ==========================  ===============================
Layer                Lifetime                    Backed by
===================  ==========================  ===============================
conversation         one task run                in-process list (bounded)
task state           across restarts             StateStore (durable)
long-term memory     forever                     StateStore (durable)
workspace files      until redeploy              filesystem
artifacts            forever                     workspace + uploaded URL
===================  ==========================  ===============================

Nothing here writes a secret: documents are redacted before they are stored.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Iterable

from .persistence import StateStore
from .redaction import redactor

log = logging.getLogger(__name__)

MEMORY_KEY = "agent.memory"
TASKS_KEY = "agent.tasks"
PROFILE_KEY = "agent.profile"

MAX_MEMORY_ENTRIES = 500
MAX_TASK_RECORDS = 50


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def new_id(prefix: str = "task") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# --------------------------------------------------------------------------
# Task state
# --------------------------------------------------------------------------
@dataclass
class TaskRecord:
    """Durable record of one objective, including its execution trace."""

    id: str
    objective: str
    status: str = "pending"  # pending | running | completed | failed | blocked
    principal: str = "system"
    channel: str = "cli"
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    finished_at: str | None = None
    steps_taken: int = 0
    tool_calls: list[str] = field(default_factory=list)
    plan: list[str] = field(default_factory=list)
    progress: list[str] = field(default_factory=list)
    result: str | None = None
    error: str | None = None
    artifacts: list[str] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)

    def touch(self) -> None:
        self.updated_at = _now_iso()

    def to_dict(self) -> dict[str, Any]:
        return redactor().scrub_deep(asdict(self))


class TaskStore:
    """Durable task registry. Survives restarts and spin-downs."""

    def __init__(self, store: StateStore) -> None:
        self._store = store

    async def _load(self) -> dict[str, Any]:
        data = await self._store.get(TASKS_KEY, {}) or {}
        return data if isinstance(data, dict) else {}

    async def save(self, record: TaskRecord) -> None:
        record.touch()
        data = await self._load()
        data[record.id] = record.to_dict()
        # Keep the registry bounded.
        if len(data) > MAX_TASK_RECORDS:
            ordered = sorted(
                data.items(), key=lambda kv: kv[1].get("updated_at", ""), reverse=True
            )[:MAX_TASK_RECORDS]
            data = dict(ordered)
        await self._store.set(TASKS_KEY, data)

    async def get(self, task_id: str) -> TaskRecord | None:
        data = await self._load()
        raw = data.get(task_id)
        if not raw:
            return None
        fields = {f for f in TaskRecord.__dataclass_fields__}
        return TaskRecord(**{k: v for k, v in raw.items() if k in fields})

    async def recent(self, limit: int = 10) -> list[dict[str, Any]]:
        data = await self._load()
        ordered = sorted(data.values(), key=lambda r: r.get("updated_at", ""), reverse=True)
        return ordered[:limit]

    async def unfinished(self) -> list[dict[str, Any]]:
        """Tasks interrupted by a restart — the agent can offer to resume them."""
        data = await self._load()
        return [r for r in data.values() if r.get("status") in ("pending", "running")]


# --------------------------------------------------------------------------
# Long-term memory
# --------------------------------------------------------------------------
@dataclass
class MemoryEntry:
    id: str
    kind: str  # fact | preference | lesson | artifact | summary
    text: str
    tags: list[str] = field(default_factory=list)
    task_id: str | None = None
    created_at: str = field(default_factory=_now_iso)
    importance: int = 3  # 1-5

    def to_dict(self) -> dict[str, Any]:
        return redactor().scrub_deep(asdict(self))


class LongTermMemory:
    """Small, durable, searchable store of things worth remembering.

    Search is deliberately lexical rather than vector-based: it needs no extra
    dependency, no embedding cost, and is good enough at this scale. The
    interface is narrow so a vector backend can be dropped in later.
    """

    def __init__(self, store: StateStore) -> None:
        self._store = store
        self._cache: list[MemoryEntry] | None = None

    async def _all(self) -> list[MemoryEntry]:
        if self._cache is not None:
            return self._cache
        raw = await self._store.get(MEMORY_KEY, []) or []
        entries: list[MemoryEntry] = []
        fields = set(MemoryEntry.__dataclass_fields__)
        for item in raw if isinstance(raw, list) else []:
            try:
                entries.append(MemoryEntry(**{k: v for k, v in item.items() if k in fields}))
            except TypeError:
                continue
        self._cache = entries
        return entries

    async def _persist(self, entries: list[MemoryEntry]) -> None:
        if len(entries) > MAX_MEMORY_ENTRIES:
            entries = sorted(
                entries, key=lambda e: (e.importance, e.created_at), reverse=True
            )[:MAX_MEMORY_ENTRIES]
        self._cache = entries
        await self._store.set(MEMORY_KEY, [e.to_dict() for e in entries])

    async def remember(
        self,
        text: str,
        *,
        kind: str = "fact",
        tags: Iterable[str] = (),
        task_id: str | None = None,
        importance: int = 3,
    ) -> MemoryEntry:
        text = redactor().scrub(text.strip())
        if not text:
            raise ValueError("cannot remember an empty string")
        entries = list(await self._all())
        # De-duplicate near-identical entries rather than accumulating noise.
        for entry in entries:
            if entry.text.lower() == text.lower():
                entry.importance = max(entry.importance, importance)
                await self._persist(entries)
                return entry
        entry = MemoryEntry(
            id=new_id("mem"),
            kind=kind,
            text=text,
            tags=[t for t in tags if t],
            task_id=task_id,
            importance=max(1, min(5, importance)),
        )
        entries.append(entry)
        await self._persist(entries)
        return entry

    async def recall(self, query: str = "", limit: int = 10) -> list[MemoryEntry]:
        entries = await self._all()
        if not query.strip():
            return sorted(entries, key=lambda e: (e.importance, e.created_at), reverse=True)[:limit]
        terms = [t for t in query.lower().replace(",", " ").split() if len(t) > 2]
        scored: list[tuple[int, MemoryEntry]] = []
        for entry in entries:
            haystack = f"{entry.text} {' '.join(entry.tags)} {entry.kind}".lower()
            score = sum(haystack.count(term) for term in terms)
            if score:
                scored.append((score + entry.importance, entry))
        scored.sort(key=lambda pair: (-pair[0], pair[1].created_at))
        return [e for _, e in scored[:limit]]

    async def forget(self, entry_id: str) -> bool:
        entries = list(await self._all())
        kept = [e for e in entries if e.id != entry_id]
        if len(kept) == len(entries):
            return False
        await self._persist(kept)
        return True

    async def count(self) -> int:
        return len(await self._all())


# --------------------------------------------------------------------------
# Conversation context
# --------------------------------------------------------------------------
class Conversation:
    """Bounded rolling transcript for one task.

    When it overflows, the oldest turns are *summarised* into a single line
    rather than dropped, so the agent does not silently lose an earlier
    decision it needs later.
    """

    def __init__(self, max_messages: int = 24, max_chars: int = 90_000) -> None:
        self.max_messages = max_messages
        self.max_chars = max_chars
        self._messages: list[dict[str, Any]] = []
        self._dropped_summary: list[str] = []

    def append(self, message: dict[str, Any]) -> None:
        self._messages.append(redactor().scrub_deep(dict(message)))
        self._trim()

    def extend(self, messages: Iterable[dict[str, Any]]) -> None:
        for message in messages:
            self.append(message)

    def _size(self) -> int:
        return sum(len(json.dumps(m, default=str)) for m in self._messages)

    def _trim(self) -> None:
        # Never split an assistant tool_call from its tool responses: drop the
        # oldest complete exchange instead of orphaning a tool result.
        while len(self._messages) > self.max_messages or self._size() > self.max_chars:
            if len(self._messages) <= 2:
                break
            removed: list[dict[str, Any]] = []
            while self._messages and len(self._messages) > 2:
                candidate = self._messages[0]
                removed.append(self._messages.pop(0))
                if candidate.get("role") == "assistant":
                    # also drop the tool results that followed
                    while self._messages and self._messages[0].get("role") == "tool":
                        removed.append(self._messages.pop(0))
                    break
            if not removed:
                break
            self._dropped_summary.append(self._summarise(removed))
            self._dropped_summary = self._dropped_summary[-6:]

    @staticmethod
    def _summarise(messages: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for message in messages:
            role = message.get("role")
            if role == "assistant" and message.get("tool_calls"):
                names = [
                    (c.get("function") or {}).get("name", "?") for c in message["tool_calls"]
                ]
                parts.append(f"called {', '.join(names)}")
            elif role == "tool":
                text = str(message.get("content", ""))[:160]
                parts.append(f"result: {text}")
            else:
                text = str(message.get("content", ""))[:200]
                if text:
                    parts.append(f"{role}: {text}")
        return " | ".join(parts)

    @property
    def messages(self) -> list[dict[str, Any]]:
        return list(self._messages)

    def context_note(self) -> str | None:
        """A compact recap of what scrolled out of the window."""
        if not self._dropped_summary:
            return None
        return "Earlier in this task (summarised): " + " ;; ".join(self._dropped_summary)

    def __len__(self) -> int:
        return len(self._messages)


# --------------------------------------------------------------------------
# Working memory passed to the model each step
# --------------------------------------------------------------------------
@dataclass
class WorkingState:
    """The agent's editable scratchpad, separate from the transcript."""

    objective: str = ""
    finish_condition: str = ""
    plan: list[str] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    step: int = 0

    def to_prompt(self) -> str:
        lines: list[str] = [f"OBJECTIVE: {self.objective}"]
        if self.finish_condition:
            lines.append(f"DONE WHEN: {self.finish_condition}")
        if self.plan:
            lines.append("PLAN:")
            for i, item in enumerate(self.plan, 1):
                mark = "x" if item.startswith("[done]") else " "
                lines.append(f"  [{mark}] {i}. {item.replace('[done]', '').strip()}")
        if self.facts:
            lines.append("FACTS LEARNED:")
            lines.extend(f"  - {f}" for f in self.facts[-12:])
        if self.blockers:
            lines.append("BLOCKERS:")
            lines.extend(f"  - {b}" for b in self.blockers[-6:])
        if self.notes:
            lines.append("NOTES:")
            lines.extend(f"  - {n}" for n in self.notes[-6:])
        return redactor().scrub("\n".join(lines))

    def to_dict(self) -> dict[str, Any]:
        return redactor().scrub_deep(asdict(self))


@dataclass
class MemoryBundle:
    """Handed to every tool and to the loop, so nothing reaches for a global."""

    store: StateStore
    tasks: TaskStore
    long_term: LongTermMemory
    conversation: Conversation
    working: WorkingState
    events: Any = None

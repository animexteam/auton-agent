"""Observability event log.

A task must be debuggable after the fact without exposing credentials. Every
significant thing the agent does becomes a structured event:

    task.start -> step.begin -> model.call -> tool.call -> tool.result ->
    step.end -> task.finish

Events are kept in a bounded in-memory ring buffer (cheap, per-process) and
appended to a JSONL file so they can be persisted through the state store.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .redaction import redactor


@dataclass
class Event:
    kind: str
    task_id: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["ts_iso"] = (
            datetime.fromtimestamp(self.ts, tz=timezone.utc).isoformat(timespec="milliseconds")
        )
        return payload


class EventLog:
    """Thread-safe bounded event buffer with optional JSONL spill."""

    def __init__(self, capacity: int = 2000, path: Path | None = None) -> None:
        self._events: deque[Event] = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._path = path
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, kind: str, task_id: str | None = None, **data: Any) -> Event:
        safe = redactor().scrub_deep(data)
        event = Event(kind=kind, task_id=task_id, data=dict(safe) if isinstance(safe, dict) else {"value": safe})
        with self._lock:
            self._events.append(event)
            if self._path is not None:
                try:
                    with self._path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(event.to_dict(), default=str, ensure_ascii=False) + "\n")
                except OSError:
                    # Observability must never break the agent.
                    pass
        return event

    def recent(self, limit: int = 100, task_id: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            items: Iterable[Event] = list(self._events)
        if task_id is not None:
            items = [e for e in items if e.task_id == task_id]
        return [e.to_dict() for e in list(items)[-limit:]]

    def counts(self) -> dict[str, int]:
        with self._lock:
            items = list(self._events)
        out: dict[str, int] = {}
        for event in items:
            out[event.kind] = out.get(event.kind, 0) + 1
        return out

    def clear(self) -> None:
        with self._lock:
            self._events.clear()


_log: EventLog | None = None


def configure(path: Path | None = None, capacity: int = 2000) -> EventLog:
    global _log
    _log = EventLog(capacity=capacity, path=path)
    return _log


def log() -> EventLog:
    global _log
    if _log is None:
        _log = EventLog()
    return _log

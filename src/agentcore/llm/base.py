"""Provider-neutral model interface.

The agent loop only ever talks to :class:`LLMProvider`. Swapping Ollama Cloud
for another vendor means writing one new subclass — nothing in the loop, the
tool registry, or the interfaces changes.
"""

from __future__ import annotations

import abc
import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

Message = dict[str, Any]


@dataclass
class ToolCall:
    """A single tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: Mapping[str, Any], index: int = 0) -> "ToolCall":
        fn = raw.get("function") or {}
        name = fn.get("name") or raw.get("name") or ""
        args = fn.get("arguments", raw.get("arguments", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                # Keep the raw string so the tool layer can report a precise error.
                args = {"__raw__": args}
        if not isinstance(args, dict):
            args = {"value": args}
        return cls(id=str(raw.get("id") or f"call_{index}"), name=str(name), arguments=dict(args))


@dataclass
class LLMResponse:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    model: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    finish_reason: str | None = None

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class LLMProvider(abc.ABC):
    """Minimal contract every model backend must satisfy."""

    name: str = "provider"

    @abc.abstractmethod
    async def chat(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        model: str | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        """Return one assistant turn for the given conversation."""

    async def list_models(self) -> list[str]:
        """Names the configured credential can actually reach."""
        return []

    async def aclose(self) -> None:
        return None

    def available(self) -> tuple[bool, str]:
        """(usable, reason) — used by the self-report to stay truthful."""
        return True, "ok"


def normalize_messages(messages: Iterable[Message]) -> list[Message]:
    """Deep-copy messages and drop provider-hostile empty parts.

    Keeps a valid OpenAI/Ollama-shaped transcript while making sure we never
    send an assistant turn with neither content nor tool calls.
    """
    out: list[Message] = []
    for msg in messages:
        item = dict(msg)
        role = item.get("role")
        content = item.get("content")
        if content is not None and not isinstance(content, str):
            item["content"] = json.dumps(content, default=str, ensure_ascii=False)
        if role == "assistant":
            calls = item.get("tool_calls")
            if calls:
                item["tool_calls"] = [
                    {
                        "type": "function",
                        "function": {
                            "name": (c.get("function") or {}).get("name", ""),
                            "arguments": (c.get("function") or {}).get("arguments", {}),
                        },
                    }
                    for c in calls
                ]
            elif not item.get("content"):
                item["content"] = "(no content)"
        if role == "tool":
            item.setdefault("content", "")
            item.pop("tool_calls", None)
        out.append(item)
    return out

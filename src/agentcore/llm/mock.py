"""Deterministic scripted provider used by the test suite.

Tests must never depend on a live model, so the loop is exercised against a
scripted provider that replays a fixed sequence of turns and records what it
was asked. It also enforces the same transcript invariants as the real
provider, so a malformed transcript fails in tests too.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .base import LLMProvider, LLMResponse, ToolCall, normalize_messages


class MockProvider(LLMProvider):
    name = "mock"

    def __init__(self, script: Sequence[Mapping[str, Any]] | None = None) -> None:
        self._script: list[Mapping[str, Any]] = list(script or [])
        self.calls: list[dict[str, Any]] = []
        self._cursor = 0

    def push(self, turn: Mapping[str, Any]) -> None:
        self._script.append(turn)

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        model: str | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        normalised = normalize_messages(messages)
        self.calls.append(
            {"messages": normalised, "tool_names": [t["function"]["name"] for t in (tools or [])]}
        )
        if self._cursor >= len(self._script):
            return LLMResponse(content="(script exhausted)", model=model or "mock")
        turn = self._script[self._cursor]
        self._cursor += 1

        if "error" in turn:
            raise turn["error"]

        calls = [
            ToolCall(
                id=tc.get("id", f"call_{i}"),
                name=tc["name"],
                arguments=tc.get("arguments", {}),
            )
            for i, tc in enumerate(turn.get("tool_calls", []))
        ]
        return LLMResponse(
            content=turn.get("content", ""),
            tool_calls=calls,
            model=model or "mock",
            usage=turn.get("usage", {}),
        )


def read_file_turn(path: str, call_id: str = "c1") -> dict[str, Any]:
    return {"content": "", "tool_calls": [{"id": call_id, "name": "read_file", "arguments": {"path": path}}]}


def final_turn(text: str) -> dict[str, Any]:
    return {"content": text}

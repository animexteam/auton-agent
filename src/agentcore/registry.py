"""Tool registry — typed contracts, validation, and structured errors.

A tool is a small object with three parts:

* ``spec``      — a JSON-schema declaration the model sees when choosing
* ``run(args)`` — the typed implementation
* ``preview``   — a short redacted string for the event log / UI

Tools never raise raw exceptions out of the registry: failures become
structured ``{"error": true, "code": ..., "message": ...}`` payloads the model
can read and act on. That is what lets the loop adapt instead of crashing.

Tool selection is *semantic* (the model reads the specs), never a keyword
switch table.

Execution is guarded by:
  * argument validation against the declared schema
  * a per-call timeout
  * a deny-list for clearly destructive shell patterns
"""

from __future__ import annotations

import abc
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from .errors import (
    AgentError,
    ApprovalRequired,
    PermissionDenied,
    ToolError,
    ToolNotFound,
    ToolTimeout,
    ToolValidationError,
)
from .redaction import redactor

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Minimal JSON-schema validation
# --------------------------------------------------------------------------
_TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "object": dict,
    "array": list,
}


def validate_arguments(schema: Mapping[str, Any], args: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and coerce tool arguments. Raises ToolValidationError.

    Intentionally small and dependency-free: it covers the subset of JSON
    Schema this registry actually emits (types, enums, required, defaults and
    numeric bounds), which keeps the container lean.
    """
    if not isinstance(args, Mapping):
        raise ToolValidationError(f"arguments must be an object, got {type(args).__name__}")

    props: dict[str, Any] = dict(schema.get("properties") or {})
    required: list[str] = list(schema.get("required") or [])
    extra_allowed = bool(schema.get("additionalProperties", False))

    cleaned: dict[str, Any] = {}
    problems: list[str] = []

    unknown = [k for k in args if k not in props]
    if unknown and not extra_allowed:
        problems.append(f"unknown argument(s): {', '.join(sorted(unknown))}")

    for key, spec in props.items():
        if key not in args or args[key] is None:
            if "default" in spec:
                cleaned[key] = spec["default"]
            elif key in required:
                problems.append(f"missing required argument '{key}'")
            continue

        value = args[key]
        expected = spec.get("type")
        if expected:
            py = _TYPE_MAP.get(expected)
            if py is not None:
                if expected in ("integer", "number") and isinstance(value, bool):
                    problems.append(f"'{key}' must be a {expected}, got boolean")
                    continue
                if not isinstance(value, py):
                    # Tolerate the common LLM habit of passing numbers as strings.
                    if isinstance(value, str) and expected in ("integer", "number"):
                        try:
                            value = int(float(value)) if expected == "integer" else float(value)
                        except (TypeError, ValueError):
                            problems.append(f"'{key}' must be a {expected}, got {value!r}")
                            continue
                    elif isinstance(value, str) and expected == "boolean":
                        low = value.strip().lower()
                        if low in ("true", "1", "yes"):
                            value = True
                        elif low in ("false", "0", "no"):
                            value = False
                        else:
                            problems.append(f"'{key}' must be a boolean, got {value!r}")
                            continue
                    else:
                        problems.append(
                            f"'{key}' must be a {expected}, got {type(value).__name__}"
                        )
                        continue

        enum = spec.get("enum")
        if enum and value not in enum:
            problems.append(f"'{key}' must be one of {enum}, got {value!r}")
            continue

        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if "minimum" in spec and value < spec["minimum"]:
                problems.append(f"'{key}' must be >= {spec['minimum']}")
                continue
            if "maximum" in spec and value > spec["maximum"]:
                problems.append(f"'{key}' must be <= {spec['maximum']}")
                continue

        cleaned[key] = value

    if problems:
        raise ToolValidationError("; ".join(problems))
    return cleaned


def schema(
    properties: Mapping[str, Any],
    required: Iterable[str] = (),
    description: str = "",
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "type": "object",
        "properties": dict(properties),
        "required": list(required),
        "additionalProperties": False,
    }
    if description:
        out["description"] = description
    return out


def prop(kind: str, description: str, **extra: Any) -> dict[str, Any]:
    base: dict[str, Any] = {"type": kind, "description": description}
    base.update(extra)
    return base


# --------------------------------------------------------------------------
# Tool contract
# --------------------------------------------------------------------------
@dataclass
class ToolResult:
    ok: bool
    content: Any = None
    error: str | None = None
    code: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    duration_ms: int = 0

    def to_message(self, max_chars: int = 6000) -> str:
        """Render the result as the text a tool message carries back to the model."""
        import json as _json

        if self.ok:
            body = self.content
            if not isinstance(body, str):
                body = _json.dumps(body, default=str, ensure_ascii=False, indent=2)
        else:
            body = _json.dumps(
                {"error": True, "code": self.code or "tool_error", "message": self.error},
                ensure_ascii=False,
            )
        if len(body) > max_chars:
            head = body[:max_chars]
            body = f"{head}\n… [truncated {len(body) - max_chars} chars]"
        return redactor().scrub(body)


class Tool(abc.ABC):
    name: str = ""
    description: str = ""
    category: str = "general"
    #: True when the tool can alter the local system or spend money.
    privileged: bool = False
    #: True when a mistaken call could destroy data.
    destructive: bool = False
    timeout_seconds: float | None = None

    @property
    @abc.abstractmethod
    def parameters(self) -> dict[str, Any]:
        """JSON schema for the arguments."""

    @abc.abstractmethod
    async def run(self, args: Mapping[str, Any], ctx: "ToolContext") -> Any:
        """Execute. Raise AgentError subclasses for expected failures."""

    # -- helpers --------------------------------------------------------

    def spec(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def preview(self, args: Mapping[str, Any]) -> str:
        """Short, redacted, human-readable one-liner for the event log."""
        try:
            import json as _json

            rendered = _json.dumps(dict(args), default=str, ensure_ascii=False)
        except Exception:  # noqa: BLE001
            rendered = str(args)
        return redactor().scrub(rendered)[:220]


@dataclass
class ToolContext:
    """Everything a tool is allowed to reach. No globals, no ambient access."""

    workspace_root: Any  # pathlib.Path
    path_guard: Any
    sandbox: Any
    state: Any
    memory: Any
    skills: Any
    gate: Any
    settings: Any
    task_id: str = "system"
    principal: str = "system"
    approval_token: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class ToolRegistry:
    """Holds tools, exposes their specs, and executes them safely."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        if not tool.name:
            raise ValueError("tool must declare a name")
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name: {tool.name}")
        self._tools[tool.name] = tool
        return tool

    def register_all(self, tools: Iterable[Tool]) -> None:
        for tool in tools:
            self.register(tool)

    def get(self, name: str) -> Tool:
        tool = self._tools.get(name)
        if tool is None:
            raise ToolNotFound(
                f"unknown tool '{name}'. Available: {', '.join(sorted(self._tools))}"
            )
        return tool

    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self, only: Sequence[str] | None = None) -> list[dict[str, Any]]:
        selected = self._tools.values() if only is None else [self.get(n) for n in only]
        return [t.spec() for t in selected]

    def catalog(self) -> list[dict[str, Any]]:
        """Compact index for prompts and self-reports (no full schemas)."""
        return [
            {
                "name": t.name,
                "category": t.category,
                "description": t.description.split(".")[0].strip(),
                "privileged": t.privileged,
                "destructive": t.destructive,
            }
            for t in sorted(self._tools.values(), key=lambda x: (x.category, x.name))
        ]

    async def execute(
        self, name: str, args: Mapping[str, Any], ctx: ToolContext
    ) -> ToolResult:
        started = time.perf_counter()
        try:
            tool = self.get(name)
        except AgentError as exc:
            return ToolResult(ok=False, error=exc.message, code=exc.code)

        try:
            cleaned = validate_arguments(tool.parameters, dict(args or {}))
        except ToolValidationError as exc:
            return ToolResult(
                ok=False,
                error=f"invalid arguments for '{name}': {exc.message}",
                code=exc.code,
            )

        timeout = tool.timeout_seconds or getattr(ctx.settings.sandbox, "timeout_seconds", 60)
        try:
            if tool.privileged or tool.destructive:
                ctx.gate.require(
                    action=name,
                    approval_token=ctx.approval_token,
                    destructive=tool.destructive,
                )
            value = await asyncio.wait_for(tool.run(cleaned, ctx), timeout=timeout)
        except AgentError as exc:
            return ToolResult(
                ok=False,
                error=exc.message,
                code=exc.code,
                duration_ms=int((time.perf_counter() - started) * 1000),
            )
        except asyncio.TimeoutError:
            return ToolResult(
                ok=False,
                error=f"tool '{name}' exceeded its {timeout}s time budget",
                code=ToolTimeout.code,
                duration_ms=int((time.perf_counter() - started) * 1000),
            )
        except Exception as exc:  # noqa: BLE001 - never let a tool kill the loop
            log.exception("tool raised", extra={"tool": name})
            return ToolResult(
                ok=False,
                error=f"unexpected tool failure: {type(exc).__name__}: {exc}",
                code=ToolError.code,
                duration_ms=int((time.perf_counter() - started) * 1000),
            )

        return ToolResult(
            ok=True,
            content=value,
            duration_ms=int((time.perf_counter() - started) * 1000),
        )


def build_default_registry() -> ToolRegistry:
    """Assemble the standard tool set. Imported lazily to avoid cycles."""
    from .tools import build_tools

    registry = ToolRegistry()
    registry.register_all(build_tools())
    return registry

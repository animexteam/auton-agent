"""Execution tools — the agent acting on its environment.

Both tools delegate to :class:`~agentcore.sandbox.Sandbox`, which applies the
deny-list and the resource limits. Neither one touches ``asyncio`` directly,
so the guarantees live in exactly one place.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

from ..registry import Tool, ToolContext, prop, schema

log = logging.getLogger(__name__)


class RunCommandTool(Tool):
    name = "run_command"
    category = "exec"
    description = (
        "Run a shell command in the agent workspace and return stdout, stderr and the exit "
        "code. Commands run under a wall-clock timeout and memory/CPU limits. Destructive "
        "commands (filesystem wipes, reboot, killing init) need explicit approval; ordinary "
        "commands run directly. Use this for git, package managers, file inspection and any "
        "CLI work."
    )
    # Authorisation comes from the principal being on the allowlist; the blast
    # radius is bounded by the sandbox resource limits and the destructive-command
    # screening. So this is *not* a per-call privileged action — otherwise the
    # agent would need a human approval token for every `ls`, which defeats it.
    privileged = False

    @property
    def parameters(self) -> dict[str, Any]:
        return schema(
            {
                "command": prop("string", "The shell command to execute."),
                "cwd": prop("string", "Working directory relative to the workspace root.", default="."),
                "timeout_seconds": prop(
                    "integer", "Maximum wall-clock seconds for this command.", minimum=1, maximum=900
                ),
            },
            required=["command"],
            description="Execute a shell command.",
        )

    async def run(self, args: Mapping[str, Any], ctx: ToolContext) -> dict[str, Any]:
        cwd = ctx.path_guard.resolve(args.get("cwd") or ".")
        result = await ctx.sandbox.run(
            args["command"],
            cwd=cwd,
            timeout_seconds=args.get("timeout_seconds"),
            # A granted approval is what lets a destructive command through.
            approval_token=ctx.approval_token,
        )
        payload = result.to_payload()
        payload["summary"] = result.summary()
        return payload


class RunPythonTool(Tool):
    name = "run_python"
    category = "exec"
    description = (
        "Write Python source to a temporary file inside the workspace and execute it, "
        "returning the captured output. Use this for data processing, calculations and "
        "anything easier to express as a script than as a shell pipeline."
    )
    privileged = False

    @property
    def parameters(self) -> dict[str, Any]:
        return schema(
            {
                "code": prop("string", "Python source code to execute."),
                "timeout_seconds": prop(
                    "integer", "Maximum wall-clock seconds.", minimum=1, maximum=900, default=60
                ),
            },
            required=["code"],
            description="Execute a Python snippet.",
        )

    async def run(self, args: Mapping[str, Any], ctx: ToolContext) -> dict[str, Any]:
        import uuid

        script_dir = ctx.workspace_root / ".agent_scripts"
        script_dir.mkdir(parents=True, exist_ok=True)
        script = script_dir / f"snippet_{uuid.uuid4().hex[:8]}.py"
        script.write_text(args["code"], encoding="utf-8")
        try:
            result = await ctx.sandbox.run(
                ["python3", str(script)],
                cwd=ctx.workspace_root,
                timeout_seconds=args.get("timeout_seconds", 60),
                shell=False,
            )
        finally:
            # Keep the workspace clean; the script is scratch, not an artifact.
            try:
                script.unlink(missing_ok=True)
            except OSError:
                pass
        payload = result.to_payload()
        payload["summary"] = result.summary()
        return payload

"""Meta tools — self-awareness and skill loading.

The self-report tool is the honest-answer mechanism: it reads the *live*
process state (which models actually answer, which tools are registered, what
permissions are in force, what limits apply) instead of asserting capabilities
from a prompt. The system prompt forbids claiming a capability that this tool
does not report.
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import sys
import time
from typing import Any, Mapping

from ..errors import ToolError
from ..registry import Tool, ToolContext, prop, schema

log = logging.getLogger(__name__)

_PROCESS_START = time.time()


class LoadSkillTool(Tool):
    name = "load_skill"
    category = "meta"
    description = (
        "Load the full instructions for a reusable skill (a documented procedure for a class "
        "of task). You are given a one-line index of available skills; call this to pull the "
        "detailed procedure for the one you need, instead of guessing. Use action 'search' to "
        "find a skill by describing the task."
    )

    @property
    def parameters(self) -> dict[str, Any]:
        return schema(
            {
                "action": prop("string", "Operation to perform.", enum=["load", "list", "search"], default="load"),
                "name": prop("string", "Skill name to load.", default=""),
                "query": prop("string", "Task description to search for when action='search'.", default=""),
            },
            description="Load a skill's full procedure.",
        )

    async def run(self, args: Mapping[str, Any], ctx: ToolContext) -> dict[str, Any]:
        skills = ctx.skills
        action = args.get("action", "load")

        if action == "list":
            return {"skills": skills.catalog(), "index": skills.index()}

        if action == "search":
            query = str(args.get("query") or "").strip()
            if not query:
                raise ToolError("query is required for search")
            matches = skills.search(query)
            return {
                "query": query,
                "matches": [s.to_dict() for s in matches],
                "hint": "Call load_skill with action='load' and the name of the best match.",
            }

        name = str(args.get("name") or "").strip()
        if not name:
            raise ToolError("name is required to load a skill")
        body = skills.load(name)
        found = not body.startswith(f"[skill:{name.lower()}] not found")
        return {"name": name, "found": found, "content": body}


class SelfReportTool(Tool):
    name = "self_report"
    category = "meta"
    description = (
        "Report your real, current environment and capabilities: the model you are actually "
        "running on, the tools registered, the execution sandbox limits, persistence health, "
        "which integrations are configured, and what you cannot do. Call this before claiming "
        "any capability, and never state a capability this tool does not confirm."
    )

    @property
    def parameters(self) -> dict[str, Any]:
        return schema(
            {
                "section": prop(
                    "string",
                    "Which part to report.",
                    enum=["all", "model", "tools", "environment", "permissions", "limits", "integrations"],
                    default="all",
                ),
            },
            description="Report live capabilities and environment.",
        )

    async def run(self, args: Mapping[str, Any], ctx: ToolContext) -> dict[str, Any]:
        section = args.get("section", "all")
        settings = ctx.settings
        wanted = lambda key: section in ("all", key)  # noqa: E731

        report: dict[str, Any] = {}

        if wanted("model"):
            router = ctx.extra.get("router")
            if router is not None:
                report["model"] = router.health()
            else:
                report["model"] = {
                    "provider": settings.model.provider,
                    "configured_model": settings.model.primary,
                    "note": "live router health unavailable in this context",
                }

        if wanted("tools"):
            registry = ctx.extra.get("registry")
            report["tools"] = registry.catalog() if registry is not None else []

        if wanted("environment"):
            report["environment"] = {
                "python": sys.version.split()[0],
                "platform": f"{platform.system()} {platform.release()}",
                "machine": platform.machine(),
                "cpu_count": os.cpu_count(),
                "cwd": str(ctx.workspace_root),
                "workspace_writable": os.access(str(ctx.workspace_root), os.W_OK),
                "persistence_backend": settings.persistence.backend,
                "uptime_seconds": int(time.time() - _PROCESS_START),
                "available_cli": sorted(
                    name for name in ("git", "curl", "python3", "node", "npm", "jq", "rg")
                    if shutil.which(name)
                ),
            }

        if wanted("permissions"):
            report["permissions"] = {
                "sandbox_enabled": settings.sandbox.enabled,
                "allow_destructive": settings.sandbox.allow_destructive,
                "filesystem_scope": str(ctx.workspace_root),
                "filesystem_scope_note": "every file path is confined to the workspace root",
                "telegram_authorised_users": settings.telegram.allowed_users or "(none — Telegram disabled)",
                "http_api_enabled": bool(settings.api_auth_token),
                "approval_token_supplied": bool(ctx.approval_token),
            }

        if wanted("limits"):
            report["limits"] = {
                "command_timeout_seconds": settings.sandbox.timeout_seconds,
                "max_output_bytes": settings.sandbox.max_output_bytes,
                "max_memory_mb": settings.sandbox.max_memory_mb,
                "max_cpu_seconds": settings.sandbox.max_cpu_seconds,
                "agent_max_steps": settings.agent.max_steps,
                "agent_max_seconds": settings.agent.max_seconds,
                "telegram_rate_limit_per_minute": settings.telegram.max_requests_per_minute,
            }

        if wanted("integrations"):
            health: dict[str, Any] = {}
            try:
                ok, msg = await ctx.memory.store.healthy()
                health["persistence"] = {"healthy": ok, "detail": msg}
            except Exception as exc:  # noqa: BLE001
                health["persistence"] = {"healthy": False, "detail": str(exc)}
            report["integrations"] = {
                "telegram": {
                    "configured": bool(settings.telegram.bot_token),
                    "mode": settings.telegram.mode,
                },
                "github": {"configured": bool(settings.github_api_key)},
                "gist_persistence": {"configured": settings.persistence.gist_ready},
                "state": health,
            }

        report["cannot_do"] = [
            "install system packages without the operator enabling it",
            "reach anything outside the workspace filesystem scope",
            "run destructive commands unless ALLOW_DESTRUCTIVE is set or an approval token is given",
            "use an integration whose credential is not configured (see 'integrations')",
        ]
        return report

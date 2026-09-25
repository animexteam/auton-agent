"""Runtime — wires configuration, memory, tools and the agent together.

One place where the object graph is assembled, so the CLI, the HTTP service and
the tests all build the *same* agent. If a component needs to change provider
or backend, it changes here and nowhere else.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import events as events_mod
from .config import Settings, load_settings
from .events import EventLog
from .llm import ModelRouter, build_provider
from .logging_setup import configure_logging, get_logger
from .loop import Agent
from .memory import (
    Conversation,
    LongTermMemory,
    MemoryBundle,
    TaskStore,
    WorkingState,
)
from .persistence import StateStore, build_store
from .redaction import configure as configure_redaction
from .registry import build_default_registry
from .sandbox import Sandbox
from .security import ApprovalGate, Authorizer, PathGuard, RateLimiter
from .skills import SkillLoader

log = get_logger(__name__)


@dataclass
class Runtime:
    """The assembled, ready-to-use agent plus everything around it."""

    settings: Settings
    store: StateStore
    memory: MemoryBundle
    router: ModelRouter
    agent: Agent
    events: EventLog
    skills: SkillLoader
    sandbox: Sandbox

    async def aclose(self) -> None:
        await self.store.aclose()
        await self.router.aclose()


def build_runtime(
    *,
    base_dir: Path | None = None,
    settings: Settings | None = None,
    provider: Any | None = None,
) -> Runtime:
    """Assemble the runtime from configuration."""
    settings = settings or load_settings(base_dir)

    # 1. Redaction first: everything logged or stored afterwards is safe.
    configure_redaction(settings.secret_values())
    configure_logging(settings.log_level)

    # 2. Directories.
    settings.workspace_root.mkdir(parents=True, exist_ok=True)
    settings.state_root.mkdir(parents=True, exist_ok=True)

    # 3. Observability.
    event_log = events_mod.configure(path=settings.state_root / "events.jsonl")

    # 4. Persistence and the three memory layers.
    store = build_store(settings.persistence, settings.state_root)
    memory = MemoryBundle(
        store=store,
        tasks=TaskStore(store),
        long_term=LongTermMemory(store),
        conversation=Conversation(max_messages=settings.agent.session_turns * 2 + 12),
        working=WorkingState(),
        events=event_log,
    )

    # 5. Skills: built-ins plus anything the agent or the operator has written.
    skills = SkillLoader(
        extra_dirs=[
            settings.workspace_root / "skills",
            Path(__file__).resolve().parent / "skills" / "library",
        ]
    )

    # 6. Model layer.
    llm = provider or build_provider(settings.model)
    router = ModelRouter(llm, settings.model.chain)

    # 7. Execution sandbox.
    sandbox = Sandbox(settings.sandbox, settings.workspace_root)

    # 8. Security.
    authorizer = Authorizer(settings.telegram.allowed_users)
    rate_limiter = RateLimiter(settings.telegram.max_requests_per_minute)
    gate = ApprovalGate(allow_destructive=settings.sandbox.allow_destructive)

    # 9. The agent.
    agent = Agent(
        router=router,
        settings=settings,
        registry=build_default_registry(),
        skills=skills,
        authorizer=authorizer,
        rate_limiter=rate_limiter,
        gate=gate,
        events=event_log,
        memory=memory,
        sandbox=sandbox,
    )

    log.info(
        "runtime ready",
        extra={
            "model_primary": settings.model.primary,
            "model_chain": list(settings.model.chain),
            "model_configured": settings.model.configured,
            "persistence": settings.persistence.backend,
            "persistence_durable": settings.persistence.gist_ready,
            "telegram_enabled": settings.telegram.enabled,
            "telegram_authorised_users": len(settings.telegram.allowed_users),
            "tools": agent.registry.names(),
            "skills": skills.count(),
            "sandbox_enabled": settings.sandbox.enabled,
        },
    )
    return Runtime(
        settings=settings,
        store=store,
        memory=memory,
        router=router,
        agent=agent,
        events=event_log,
        skills=skills,
        sandbox=sandbox,
    )


def _find_gist_store(store: Any) -> Any:
    """Return the :class:`GistStore` inside a store, if there is one.

    The chained backend wraps a disk store and a gist store, so the gist has to be
    pulled out of the wrapper before it can be asked to provision itself.
    """
    if hasattr(store, "resolve_gist_id"):
        return store
    mirror = getattr(store, "mirror", None)
    if mirror is not None and hasattr(mirror, "resolve_gist_id"):
        return mirror
    return None


async def preflight(runtime: Runtime) -> dict[str, Any]:
    """Check what actually works before accepting traffic.

    Honest startup diagnostics: it reports the models that genuinely answer,
    whether the workspace is writable, and whether persistence is durable —
    rather than assuming the configuration is correct.
    """
    report: dict[str, Any] = {"checks": {}, "warnings": [], "ok": True}

    # model reachability (cheap probe, does not burn a full generation)
    usable: list[str] = []
    if runtime.settings.model.configured:
        probe = await runtime.router.chat([{"role": "user", "content": "ping"}])
        if probe.content or probe.tool_calls:
            usable.append(probe.model or runtime.settings.model.primary)
    if not usable:
        report["ok"] = False
        report["warnings"].append(
            "no model responded to a probe; check OLLAMA_API_KEY and MODEL_PRIMARY"
        )
    report["checks"]["models_responding"] = usable

    # workspace writability
    probe_file = runtime.settings.workspace_root / ".write_probe"
    try:
        probe_file.write_text("ok", encoding="utf-8")
        writable = probe_file.read_text(encoding="utf-8") == "ok"
        probe_file.unlink(missing_ok=True)
    except OSError as exc:
        writable = False
        report["ok"] = False
        report["warnings"].append(f"workspace is not writable: {exc}")
    report["checks"]["workspace_writable"] = writable

    # persistence durability
    try:
        ok, detail = await runtime.store.healthy()
    except Exception as exc:  # noqa: BLE001
        ok, detail = False, str(exc)
    report["checks"]["persistence"] = {"healthy": ok, "detail": detail}
    report["checks"]["durable_persistence"] = runtime.settings.persistence.gist_ready
    if not runtime.settings.persistence.gist_ready:
        report["warnings"].append(
            "GIST_API_KEY not set: state will not survive a redeploy on an ephemeral "
            "filesystem"
        )
    else:
        # Provision the state gist now rather than on the first write, so a failure
        # (bad token, no gist scope) surfaces at startup instead of mid-task.
        gist = _find_gist_store(runtime.store)
        if gist is not None:
            try:
                resolved = await gist.resolve_gist_id()
                report["checks"]["state_gist"] = resolved
            except Exception as exc:  # noqa: BLE001
                report["warnings"].append(f"gist provisioning failed: {exc}")

    # security posture
    report["checks"]["telegram_authorised_users"] = len(runtime.settings.telegram.allowed_users)
    if runtime.settings.telegram.enabled and not runtime.settings.telegram.allowed_users:
        report["warnings"].append(
            "TELEGRAM_BOT_TOKEN is set but TELEGRAM_ALLOWED_USERS is empty: "
            "every Telegram message will be denied (this is the safe default)"
        )
    report["checks"]["api_auth_configured"] = bool(runtime.settings.api_auth_token)
    report["checks"]["destructive_allowed"] = runtime.settings.sandbox.allow_destructive

    return report

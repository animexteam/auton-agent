"""HTTP surface: health, Telegram webhook, and an authenticated task API.

Design notes:

* The container never exposes an unauthenticated shell. The only privileged
  endpoints require ``API_AUTH_TOKEN``; with no token configured they are
  *disabled*, not open.
* The health endpoint deliberately leaks nothing sensitive (no hostnames,
  paths or versions beyond a boolean readiness) so it is safe to hit publicly.
* The Telegram webhook verifies its secret header in constant time before any
  work happens.
* Tasks run as background asyncio jobs; the webhook acknowledges Telegram
  immediately, because Telegram retries (and eventually disables) a webhook
  that does not answer quickly.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from .config import Settings
from .errors import PermissionDenied, RateLimited
from .interfaces.telegram import TelegramInterface
from .memory import new_id
from .runtime import Runtime, preflight
from .security import Principal, require_api_token

log = logging.getLogger(__name__)

STARTED_AT = time.time()


def create_app(runtime: Runtime, telegram: TelegramInterface | None = None) -> FastAPI:
    settings: Settings = runtime.settings
    agent = runtime.agent

    app = FastAPI(
        title="auton-agent",
        version="1.0.0",
        description="An autonomous agent exposed over Telegram and an authenticated HTTP API.",
        docs_url=None,  # no public API browser on a service that can execute commands
        redoc_url=None,
    )
    app.state.runtime = runtime
    app.state.telegram = telegram
    app.state.startup_report = {}

    # ---------------------------------------------------------------- health
    @app.get("/health")
    async def health() -> dict[str, Any]:
        """Liveness + readiness. Safe to expose publicly: no secrets, no paths."""
        return {
            "status": "ok",
            "service": "auton-agent",
            "uptime_seconds": int(time.time() - STARTED_AT),
            "ready": bool(app.state.startup_report.get("ok", False)),
            "model_configured": settings.model.configured,
            "telegram_enabled": settings.telegram.enabled,
            "durable_state": settings.persistence.gist_ready,
        }

    @app.get("/")
    async def root() -> dict[str, Any]:
        return {
            "service": "auton-agent",
            "description": "Autonomous agent. Drive it via Telegram or the authenticated API.",
            "endpoints": ["/health", "/telegram/webhook", "/tasks", "/self", "/events"],
        }

    @app.get("/ready")
    async def ready() -> JSONResponse:
        """Readiness with the preflight report (still no secrets)."""
        report = app.state.startup_report or {}
        status = 200 if report.get("ok") else 503
        return JSONResponse(report, status_code=status)

    # ---------------------------------------------------------------- auth
    def _authorize(authorization: str | None) -> Principal:
        token = None
        if authorization and authorization.lower().startswith("bearer "):
            token = authorization[7:].strip()
        try:
            require_api_token(settings.api_auth_token, token)
        except PermissionDenied as exc:
            raise HTTPException(status_code=401, detail=exc.message) from exc
        return Principal("api", "token")

    # ---------------------------------------------------------------- telegram
    @app.post("/telegram/webhook")
    async def telegram_webhook(
        request: Request,
        background: BackgroundTasks,
        x_telegram_bot_api_secret_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        if telegram is None:
            raise HTTPException(status_code=503, detail="telegram interface is not enabled")
        try:
            telegram.verify_webhook(x_telegram_bot_api_secret_token)
        except PermissionDenied as exc:
            log.warning("rejected telegram webhook", extra={"reason": exc.message})
            raise HTTPException(status_code=403, detail="forbidden") from exc

        update = await request.json()
        # Acknowledge immediately: Telegram retries a slow webhook.
        background.add_task(_safe_handle_update, telegram, update)
        return {"ok": True}

    async def _safe_handle_update(tg: TelegramInterface, update: dict[str, Any]) -> None:
        try:
            await tg.handle_update(update)
        except Exception:  # noqa: BLE001
            log.exception("background telegram handling failed")

    # ---------------------------------------------------------------- tasks
    @app.post("/tasks")
    async def create_task(
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        principal = _authorize(authorization)
        body = await request.json()
        objective = str(body.get("objective") or "").strip()
        if not objective:
            raise HTTPException(status_code=422, detail="'objective' is required")
        approval_token = body.get("approval_token")
        try:
            agent.authorizer.check(principal)
            agent.rate_limiter.check(principal)
        except (PermissionDenied, RateLimited) as exc:
            raise HTTPException(status_code=429 if "rate" in exc.code else 403, detail=exc.message)

        task_id = new_id("task")
        task = asyncio.create_task(
            agent.run(
                objective,
                principal=principal,
                channel="api",
                approval_token=approval_token,
                resume_task_id=task_id,
            )
        )
        app.state.background_tasks = getattr(app.state, "background_tasks", {})
        app.state.background_tasks[task_id] = task
        return {"task_id": task_id, "status": "running", "poll": f"/tasks/{task_id}"}

    @app.get("/tasks/{task_id}")
    async def get_task(
        task_id: str,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        _authorize(authorization)
        record = await runtime.memory.tasks.get(task_id)
        if record is None:
            raise HTTPException(status_code=404, detail="unknown task id")
        return record.to_dict()

    @app.get("/tasks")
    async def list_tasks(
        authorization: str | None = Header(default=None),
        limit: int = 10,
    ) -> dict[str, Any]:
        _authorize(authorization)
        return {"tasks": await runtime.memory.tasks.recent(max(1, min(limit, 50)))}

    # ---------------------------------------------------------------- introspection
    @app.get("/self")
    async def self_report(authorization: str | None = Header(default=None)) -> dict[str, Any]:
        """What the agent actually is right now — the honest capability report."""
        _authorize(authorization)
        return {
            "describe": agent.describe(),
            "model_health": runtime.router.health(),
            "limits": {
                "max_steps": settings.agent.max_steps,
                "max_seconds": settings.agent.max_seconds,
                "command_timeout": settings.sandbox.timeout_seconds,
                "max_memory_mb": settings.sandbox.max_memory_mb,
            },
            "security": {
                "telegram_authorised_users": len(settings.telegram.allowed_users),
                "api_auth_configured": bool(settings.api_auth_token),
                "destructive_allowed": settings.sandbox.allow_destructive,
            },
            "persistence": {
                "backend": settings.persistence.backend,
                "durable": settings.persistence.gist_ready,
            },
            "startup_report": app.state.startup_report,
        }

    @app.get("/events")
    async def get_events(
        authorization: str | None = Header(default=None),
        limit: int = 100,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        _authorize(authorization)
        return {
            "counts": runtime.events.counts(),
            "events": runtime.events.recent(max(1, min(limit, 500)), task_id=task_id),
        }

    @app.post("/approvals")
    async def create_approval(
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        """Issue a one-shot approval token for a privileged action."""
        _authorize(authorization)
        import secrets as _secrets

        body = {}
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            body = {}
        token = _secrets.token_urlsafe(24)
        agent.gate.grant(token)
        return {
            "approval_token": token,
            "scope": body.get("scope", "privileged actions"),
            "expires": "process lifetime (revoke by restarting the service)",
        }

    # ---------------------------------------------------------------- lifecycle
    # The startup preflight lives in the ASGI lifespan (see main.py), which is the
    # supported hook. It is also exposed here so a test or an embedding caller can
    # populate the readiness report without starting a server.
    async def run_preflight() -> dict[str, Any]:
        report = await preflight(runtime)
        app.state.startup_report = report
        for warning in report.get("warnings", []):
            log.warning("preflight: %s", warning)
        return report

    app.state.run_preflight = run_preflight
    app.state.runtime = runtime
    return app

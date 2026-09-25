"""Service entrypoint — `python -m agentcore.main service`.

Used by the Docker image / Render web service. Chooses the Telegram transport
from configuration:

* ``TELEGRAM_MODE=webhook`` (default in production): Telegram pushes updates to
  ``/telegram/webhook``. No polling loop, so the service is idle-cheap.
* ``TELEGRAM_MODE=polling``: a background getUpdates loop runs alongside the
  HTTP server (useful locally, or when the service has no public URL).

Either way the HTTP surface is served, so health checks and the authenticated
task API always work.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager

import uvicorn

from .interfaces.telegram import TelegramInterface
from .runtime import build_runtime, preflight
from .service import create_app

log = logging.getLogger(__name__)


async def _webhook_url_hint() -> str | None:
    return os.environ.get("RENDER_EXTERNAL_URL") or os.environ.get("PUBLIC_BASE_URL")


def main() -> None:
    runtime = build_runtime()
    settings = runtime.settings

    telegram = TelegramInterface(agent=runtime.agent, settings=settings)
    app = create_app(runtime, telegram)

    poll_task: asyncio.Task | None = None

    @asynccontextmanager
    async def lifespan(_app):
        nonlocal poll_task
        report = await preflight(runtime)
        app.state.startup_report = report
        for warning in report.get("warnings", []):
            log.warning("preflight: %s", warning)

        if telegram.client.configured():
            try:
                info = await telegram.initialise()
                log.info("telegram ready", extra=info)
                if settings.telegram.mode == "polling":
                    poll_task = asyncio.create_task(telegram.poll_forever())
                    log.info("telegram polling mode active")
                else:
                    base = await _webhook_url_hint()
                    if base:
                        secret_state = "with secret" if settings.telegram.webhook_secret else "WITHOUT secret"
                        log.info("webhook mode (%s); POST /telegram/webhook is ready at %s", secret_state, base)
                    else:
                        log.warning(
                            "webhook mode selected but RENDER_EXTERNAL_URL/PUBLIC_BASE_URL is unset; "
                            "set TELEGRAM_MODE=polling or expose a public URL"
                        )
            except Exception:  # noqa: BLE001 - the HTTP surface must still come up
                log.exception("telegram initialisation failed; continuing with HTTP only")
        else:
            log.warning("TELEGRAM_BOT_TOKEN not set: running without a Telegram interface")

        try:
            yield
        finally:
            if poll_task is not None:
                poll_task.cancel()
                await asyncio.gather(poll_task, return_exceptions=True)
            await telegram.client.aclose()
            await runtime.aclose()

    app.router.lifespan_context = lifespan

    port = int(os.environ.get("PORT", settings.port))
    log.info("starting http server", extra={"port": port})
    uvicorn.run(app, host="0.0.0.0", port=port, log_config=None, access_log=False)


if __name__ == "__main__":
    main()

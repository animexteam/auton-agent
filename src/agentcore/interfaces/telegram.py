"""Telegram interface.

Telegram is treated as one *channel*, not as the agent's foundation: the
interface layer translates Telegram updates into ``agent.run(objective)`` calls
and formats the result back. The same agent serves the CLI and the HTTP API, so
nothing about the agent's design depends on Telegram.

Two transports are supported because they suit different deployments:

* ``webhook`` — the production path on Render: Telegram pushes updates to the
  service, so the agent is not permanently polling (important on a free plan
  with limited instance hours).
* ``polling``  — ``getUpdates`` long-polling: works behind NAT and for local
  development, at the cost of a permanently running loop.

Authorisation and rate limiting are enforced *before* the agent is invoked, and
they fail closed: with no allowlist configured, nobody is allowed in.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import httpx

from ..config import Settings
from ..errors import AgentError, PermissionDenied, RateLimited
from ..loop import Agent, RunResult
from ..security import Principal

log = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org"
MAX_TELEGRAM_CHARS = 3900


@dataclass
class IncomingMessage:
    update_id: int
    chat_id: int
    user_id: str
    username: str
    text: str
    is_group: bool = False


class TelegramClient:
    """Thin wrapper over the Bot API."""

    def __init__(self, token: str | None, timeout: float = 30.0) -> None:
        self.token = token
        self._client: httpx.AsyncClient | None = None
        self._timeout = timeout

    def configured(self) -> bool:
        return bool(self.token)

    async def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _url(self, method: str) -> str:
        return f"{TELEGRAM_API}/bot{self.token}/{method}"

    async def call(self, method: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self.token:
            raise AgentError("telegram bot token is not configured")
        client = await self._http()
        try:
            resp = await client.post(self._url(method), json=payload or {})
        except httpx.HTTPError as exc:
            raise AgentError(f"telegram request failed: {exc}") from exc
        try:
            body = resp.json()
        except ValueError:
            raise AgentError(f"telegram returned non-JSON (http {resp.status_code})")
        if not body.get("ok"):
            raise AgentError(
                f"telegram {method} failed: {body.get('description', resp.status_code)}"
            )
        return body.get("result") or {}

    async def send_message(self, chat_id: int, text: str) -> None:
        for chunk in _chunks(text, MAX_TELEGRAM_CHARS):
            await self.call(
                "sendMessage",
                {
                    "chat_id": chat_id,
                    "text": chunk,
                    "disable_web_page_preview": True,
                },
            )

    async def send_typing(self, chat_id: int) -> None:
        try:
            await self.call("sendChatAction", {"chat_id": chat_id, "action": "typing"})
        except AgentError:
            pass  # cosmetic only

    async def set_webhook(self, url: str, secret: str | None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "url": url,
            "allowed_updates": ["message", "edited_message"],
            "drop_pending_updates": True,
        }
        if secret:
            payload["secret_token"] = secret
        return await self.call("setWebhook", payload)

    async def delete_webhook(self) -> dict[str, Any]:
        return await self.call("deleteWebhook", {"drop_pending_updates": False})

    async def get_webhook_info(self) -> dict[str, Any]:
        return await self.call("getWebhookInfo", {})

    async def get_me(self) -> dict[str, Any]:
        return await self.call("getMe", {})


def _chunks(text: str, size: int) -> list[str]:
    if len(text) <= size:
        return [text]
    chunks: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= size:
            chunks.append(remaining)
            break
        window = remaining[:size]
        split = max(window.rfind("\n\n"), window.rfind("\n"), window.rfind(" "))
        split = split if split > size // 2 else size
        chunks.append(remaining[:split])
        remaining = remaining[split:].lstrip()
    return chunks


def parse_update(update: dict[str, Any]) -> IncomingMessage | None:
    """Extract a text message from a Telegram update, or None."""
    message = update.get("message") or update.get("edited_message")
    if not message:
        return None
    text = (message.get("text") or "").strip()
    if not text:
        return None
    chat = message.get("chat") or {}
    sender = message.get("from") or {}
    return IncomingMessage(
        update_id=int(update.get("update_id", 0)),
        chat_id=int(chat.get("id", 0)),
        user_id=str(sender.get("id", "")),
        username=str(sender.get("username") or sender.get("first_name") or "unknown"),
        text=text,
        is_group=chat.get("type") in ("group", "supergroup"),
    )


def strip_bot_mention(text: str, bot_username: str | None) -> str:
    """In a group, only respond to messages addressed to the bot."""
    if bot_username:
        for token in (f"@{bot_username}", f"@{bot_username.lower()}"):
            if text.startswith(token):
                return text[len(token) :].strip()
    return text


class TelegramInterface:
    """Turns updates into agent runs and results back into messages."""

    def __init__(
        self,
        *,
        agent: Agent,
        settings: Settings,
        client: TelegramClient | None = None,
    ) -> None:
        self.agent = agent
        self.settings = settings
        self.client = client or TelegramClient(settings.telegram.bot_token)
        self.bot_username: str | None = None
        self._seen_updates: set[int] = set()

    async def initialise(self) -> dict[str, Any]:
        """Discover the bot identity and, in webhook mode, register the URL."""
        if not self.client.configured():
            return {"enabled": False, "reason": "TELEGRAM_BOT_TOKEN not set"}
        me = await self.client.get_me()
        self.bot_username = me.get("username")
        info: dict[str, Any] = {"enabled": True, "username": self.bot_username}
        return info

    # -- webhook mode ---------------------------------------------------

    def verify_webhook(self, provided_secret: str | None) -> None:
        from ..security import verify_webhook_secret

        verify_webhook_secret(self.settings.telegram.webhook_secret, provided_secret)

    async def handle_update(self, update: dict[str, Any]) -> dict[str, Any]:
        """Process one Telegram update. Returns a small status dict."""
        message = parse_update(update)
        if message is None:
            return {"handled": False, "reason": "no text message"}

        if message.update_id in self._seen_updates:
            return {"handled": False, "reason": "duplicate update"}
        self._seen_updates.add(message.update_id)
        if len(self._seen_updates) > 5000:
            self._seen_updates = set(list(self._seen_updates)[-2000:])

        text = strip_bot_mention(message.text, self.bot_username) if message.is_group else message.text
        principal = Principal("telegram", message.user_id)

        # Handle the built-in commands before spending any model budget.
        command = text.lower().strip()
        if command in ("/start", "/help"):
            await self.client.send_message(message.chat_id, self._help_text())
            return {"handled": True, "command": command}
        if command == "/status":
            await self.client.send_message(message.chat_id, self._status_text())
            return {"handled": True, "command": command}
        if command == "/whoami":
            await self.client.send_message(
                message.chat_id,
                f"Your Telegram user id is {message.user_id}. "
                f"(Your operator must add this id to TELEGRAM_ALLOWED_USERS for you to run tasks.)",
            )
            return {"handled": True, "command": command}
        if command == "/id":
            await self.client.send_message(message.chat_id, f"chat_id={message.chat_id}")
            return {"handled": True, "command": command}

        # Authorisation + rate limit are checked by the agent too (defence in
        # depth), but doing it here lets us reply with a clear reason.
        try:
            self.agent.authorizer.check(principal)
            self.agent.rate_limiter.check(principal)
        except (PermissionDenied, RateLimited) as exc:
            await self.client.send_message(message.chat_id, f"Not authorised: {exc.message}")
            self.agent.events.emit("telegram.denied", None, user=message.user_id, reason=exc.code)
            return {"handled": True, "denied": True, "reason": exc.code}

        await self.client.send_typing(message.chat_id)
        typing_task = asyncio.create_task(self._typing_loop(message.chat_id))
        try:
            result: RunResult = await self.agent.run(
                text, principal=principal, channel="telegram"
            )
        finally:
            typing_task.cancel()
            with_suppressed = asyncio.gather(typing_task, return_exceptions=True)
            await with_suppressed

        await self.client.send_message(message.chat_id, self.format_result(result))
        return {"handled": True, "task_id": result.task_id, "status": result.status}

    async def _typing_loop(self, chat_id: int) -> None:
        try:
            while True:
                await self.client.send_typing(chat_id)
                await asyncio.sleep(4)
        except asyncio.CancelledError:
            return

    # -- polling mode ---------------------------------------------------

    async def poll_forever(self) -> None:
        """Long-poll getUpdates. Used when a public webhook URL is not available."""
        if not self.client.configured():
            raise AgentError("cannot poll without TELEGRAM_BOT_TOKEN")
        await self.initialise()
        await self.client.delete_webhook()
        offset = 0
        log.info("telegram polling started", extra={"username": self.bot_username})
        while True:
            try:
                updates = await self.client.call(
                    "getUpdates",
                    {"offset": offset, "timeout": 25, "allowed_updates": ["message"]},
                )
            except AgentError as exc:
                log.warning("poll failed", extra={"error": exc.message})
                await asyncio.sleep(5)
                continue
            for update in updates if isinstance(updates, list) else []:
                offset = max(offset, int(update.get("update_id", 0)) + 1)
                try:
                    await self.handle_update(update)
                except Exception:  # noqa: BLE001 - never kill the poll loop
                    log.exception("failed handling update")

    # -- formatting -----------------------------------------------------

    @staticmethod
    def format_result(result: RunResult) -> str:
        icon = {
            "completed": "✅",
            "failed": "❌",
            "blocked": "⏸",
            "budget_exceeded": "⏳",
        }.get(result.status, "•")
        lines = [f"{icon} {result.status.replace('_', ' ')}"]
        if result.answer:
            lines.append("")
            lines.append(result.answer)
        if result.artifacts:
            lines.append("")
            lines.append("Artifacts: " + ", ".join(result.artifacts[:8]))
        lines.append("")
        lines.append(
            f"task {result.task_id} · {len(result.steps)} steps · {result.duration_ms / 1000:.1f}s"
        )
        return "\n".join(lines)

    def _help_text(self) -> str:
        return (
            "I'm an autonomous agent. Send me an objective and I'll work it out myself — "
            "researching, writing files, running commands and verifying the result.\n\n"
            "Commands:\n"
            "/status — my current configuration and limits\n"
            "/whoami — your Telegram user id (needed for the allowlist)\n"
            "/help — this message\n\n"
            "Anything else you send is treated as a task to accomplish."
        )

    def _status_text(self) -> str:
        info = self.agent.describe()
        models = info.get("model", {})
        return (
            f"Model chain: {', '.join(models.get('models', [{}])[0].get('name', '?') and [m['name'] for m in models.get('models', [])])}\n"
            f"Tools: {', '.join(info.get('tools', []))}\n"
            f"Skills: {len(info.get('skills', []))}\n"
            f"Max steps per task: {self.settings.agent.max_steps}\n"
            f"Command timeout: {self.settings.sandbox.timeout_seconds}s\n"
            f"Workspace: {self.settings.workspace_root}\n"
            f"Durable state: {'yes (Gist)' if self.settings.persistence.gist_ready else 'disk only'}\n"
            f"Destructive commands: {'allowed' if self.settings.sandbox.allow_destructive else 'blocked'}"
        )

#!/usr/bin/env python3
"""Prove the Telegram path end to end without needing the bot token.

The bot token lives in the deployment's environment, not in the sandbox, and
Telegram will not accept a forged `from` field on a normal message — so the
honest way to test the interface layer is to drive `TelegramInterface` directly
with a synthetic update and a recording stub client. That exercises the real
routing, auth, agent loop and reply-formatting code; only the transport is
faked.

Run:  python scripts/verify_telegram_path.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

_tmp = Path(tempfile.mkdtemp(prefix="tg-verify-"))
os.environ["WORKSPACE_ROOT"] = str(_tmp / "workspace")
os.environ["STATE_ROOT"] = str(_tmp / "state")
os.environ["PERSISTENCE_BACKEND"] = "disk"
os.environ.pop("GIST_API_KEY", None)
os.environ.pop("TELEGRAM_BOT_TOKEN", None)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agentcore.config import Settings, load_settings  # noqa: E402
from agentcore.interfaces.telegram import TelegramInterface  # noqa: E402
from agentcore.runtime import build_runtime  # noqa: E402


class RecordingClient:
    """Stands in for the Bot API. Records exactly what would be sent."""

    def __init__(self) -> None:
        self.sent: list[tuple[int, str]] = []
        self.actions: list[str] = []

    def configured(self) -> bool:
        return True

    async def send_message(self, chat_id: int, text: str) -> None:
        self.sent.append((chat_id, text))

    async def send_typing(self, chat_id: int) -> None:
        self.actions.append("typing")

    async def get_me(self) -> dict:
        return {"username": "iProAiBot", "first_name": "iPro Ai"}

    async def aclose(self) -> None:
        return None


_UPDATE_SEQ = [0]


def update(text: str, user_id: int, chat_id: int | None = None) -> dict:
    # Telegram update ids must be unique and increasing. The interface dedupes on
    # them (correctly), so reusing one silently swallows the message.
    _UPDATE_SEQ[0] += 1
    return {
        "update_id": 100000 + _UPDATE_SEQ[0],
        "message": {
            "message_id": 5,
            "chat": {"id": chat_id if chat_id is not None else user_id, "type": "private"},
            "from": {"id": user_id, "username": "operator", "first_name": "Operator"},
            "text": text,
        },
    }


async def main() -> int:
    base = load_settings()
    # The operator in this test is the real allowlisted id.
    allowed = base.telegram.allowed_users[0]
    settings = Settings(
        model=base.model,
        telegram=type(base.telegram)(
            bot_token="test-token",
            mode="webhook",
            webhook_secret=base.telegram.webhook_secret,
            allowed_users=(allowed,),
            max_requests_per_minute=30,
        ),
        persistence=base.persistence,
        sandbox=base.sandbox,
        agent=base.agent,
        workspace_root=base.workspace_root,
        state_root=base.state_root,
        api_auth_token=base.api_auth_token,
    )

    runtime = build_runtime(settings=settings)
    client = RecordingClient()
    tg = TelegramInterface(agent=runtime.agent, settings=settings, client=client)
    tg.bot_username = "iProAiBot"

    results = {}

    # ---- 1. a real objective, delivered through the Telegram interface ----
    print("=" * 72)
    print("TELEGRAM PATH — real objective through the interface layer")
    print("=" * 72)
    obj = (
        "Search the web and tell me the current date today, in one short line. "
        "Use web_search, do not answer from memory."
    )
    out = await tg.handle_update(update(obj, int(allowed)))
    print(f"handle_update -> {json.dumps(out)}")
    print(f"replies sent  : {len(client.sent)}")
    for chat_id, text in client.sent:
        print(f"--- reply to chat {chat_id} ---")
        print(text[:900])
    results["1_objective_handled"] = bool(out.get("handled"))
    results["2_agent_ran_and_replied"] = len(client.sent) >= 1
    reply_text = client.sent[-1][1] if client.sent else ""
    results["3_reply_contains_current_date"] = "2026" in reply_text
    print(f"typing indicators sent: {len(client.actions)}")

    # ---- 2. built-in commands -------------------------------------------------
    print()
    print("=" * 72)
    print("BUILT-IN COMMANDS")
    print("=" * 72)
    client.sent.clear()
    await tg.handle_update(update("/whoami", int(allowed)))
    print(f"/whoami  -> {client.sent[-1][1]!r}")
    client.sent.clear()
    await tg.handle_update(update("/id", int(allowed)))
    print(f"/id      -> {client.sent[-1][1]!r}")
    client.sent.clear()
    await tg.handle_update(update("/help", int(allowed)))
    print(f"/help    -> {client.sent[-1][1][:160]!r}...")
    client.sent.clear()
    await tg.handle_update(update("/status", int(allowed)))
    status_text = client.sent[-1][1]
    print(f"/status  -> {status_text[:400]!r}")
    results["4_status_reports_new_model"] = "gpt-oss:120b" in status_text

    # ---- 3. an unauthorised user is refused ------------------------------------
    print()
    print("=" * 72)
    print("AUTHORISATION — a non-allowlisted user")
    print("=" * 72)
    client.sent.clear()
    out = await tg.handle_update(update("do something for me", 999999999))
    print(f"handle_update -> {json.dumps(out)}")
    print(f"replies sent  : {len(client.sent)}")
    if client.sent:
        print(f"reply         : {client.sent[-1][1]!r}")
    results["5_unknown_user_denied"] = out.get("denied") is True
    results["6_unknown_user_did_not_run_agent"] = out.get("task_id") is None

    await runtime.aclose()

    print()
    print("=" * 72)
    print("VERDICT")
    print("=" * 72)
    print(json.dumps(results, indent=2))
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

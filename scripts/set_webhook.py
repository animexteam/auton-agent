#!/usr/bin/env python3
"""Register, inspect or remove the Telegram webhook for a deployed auton-agent.

Why this is its own script
--------------------------
Telegram permits exactly ONE webhook URL per bot. Registration is therefore a
deliberate operational act, not something that should happen implicitly inside
application code. Keeping it here has a second benefit: the running service
never needs the public URL baked in as configuration -- it only has to serve
``POST /telegram/webhook`` and verify the secret header. The same image works on
any host.

How the secret fits in
----------------------
``TELEGRAM_WEBHOOK_SECRET`` is registered with Telegram *and* held by the
service. Telegram echoes it in the ``X-Telegram-Bot-Api-Secret-Token`` header on
every delivery; ``service.py`` compares it in constant time and answers 403 on a
mismatch, so a forged request never reaches the agent. The two values must be
identical, which is why this script reads the secret from the same environment
the service was deployed with.

Usage
-----
    python scripts/set_webhook.py --url https://auton-agent.onrender.com
    python scripts/set_webhook.py --url https://auton-agent.onrender.com --drop-pending
    python scripts/set_webhook.py --status
    python scripts/set_webhook.py --delete
    python scripts/set_webhook.py                                  # uses RENDER_EXTERNAL_URL

Secrets are read from the environment (or ``.env``) and never printed -- only a
length plus a short SHA-256 fingerprint, so the output is safe to paste into a
report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx

TELEGRAM_API = "https://api.telegram.org"

#: The route the service exposes. Must match service.py.
WEBHOOK_PATH = "/telegram/webhook"

#: Telegram will consider a webhook "recently errored" if the service takes
#: longer than this to answer. The handler acknowledges immediately and runs the
#: task in the background, so this should never be hit.
ANSWER_TIMEOUT_HINT_SECONDS = 10

#: Updates the agent understands. Restricting them stops Telegram queueing
#: channel posts, polls and other noise the agent would only discard.
ALLOWED_UPDATES = ["message", "edited_message"]


# --------------------------------------------------------------------------
# pure helpers (unit-tested)
# --------------------------------------------------------------------------
def build_webhook_url(base: str) -> str:
    """Turn a public base URL into the full webhook URL.

    Tolerates a scheme-less host, a trailing slash, or a base that already
    includes the path, so it is safe to pass whatever the host reports.
    """
    base = (base or "").strip().rstrip("/")
    if not base:
        raise ValueError("a public base URL is required (pass --url)")
    if not base.startswith(("http://", "https://")):
        base = "https://" + base
    if base.endswith(WEBHOOK_PATH):
        return base
    return base + WEBHOOK_PATH


def fingerprint(value: str | None) -> str:
    """A safe stand-in for a secret: presence, length, and a short digest."""
    if not value:
        return "(unset)"
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return f"set (len={len(value)}, sha256:{digest})"


def summarise_webhook_info(info: dict[str, Any]) -> str:
    """Human-readable, secret-free summary of getWebhookInfo."""
    url = info.get("url") or "(none registered)"
    lines = [f"url              : {url}"]
    lines.append(f"pending_updates  : {info.get('pending_update_count', 0)}")
    # Key the check off last_error_message, which is what we actually display.
    # Telegram normally sends last_error_date too, but treating the date as the
    # signal would report "none" for an error that has no timestamp -- exactly
    # the false negative that makes a broken webhook look healthy.
    if info.get("last_error_message"):
        when = info.get("last_error_date")
        suffix = f", at {when}" if when else ""
        lines.append(
            f"last_error       : {info.get('last_error_message')} "
            f"(code {info.get('last_error_code')}{suffix})"
        )
    else:
        lines.append("last_error       : none")
    lines.append(f"allowed_updates  : {', '.join(info.get('allowed_updates') or []) or '(default)'}")
    if info.get("max_connections") is not None:
        lines.append(f"max_connections  : {info.get('max_connections')}")
    return "\n".join(lines)


def load_dotenv() -> None:
    """Load ``.env`` from the cwd or its parent. Never overwrites real env."""
    for candidate in (Path(".env"), Path("../.env")):
        if not candidate.exists():
            continue
        for raw in candidate.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
        return


# --------------------------------------------------------------------------
# Telegram calls
# --------------------------------------------------------------------------
def _call(token: str, method: str, payload: dict[str, Any] | None = None) -> Any:
    resp = httpx.post(
        f"{TELEGRAM_API}/bot{token}/{method}",
        json=payload or {},
        timeout=40.0,
    )
    try:
        body = resp.json()
    except ValueError:
        raise SystemExit(f"telegram {method}: non-JSON reply (http {resp.status_code})")
    if not body.get("ok"):
        raise SystemExit(
            f"telegram {method} failed: {body.get('description', resp.status_code)}"
        )
    return body.get("result")


def service_health(base_url: str) -> dict[str, Any] | None:
    """Best-effort peek at the deployed service's /health (no auth needed)."""
    try:
        resp = httpx.get(build_webhook_url(base_url).removesuffix(WEBHOOK_PATH) + "/health", timeout=90.0)
    except httpx.HTTPError as exc:
        print(f"  health probe failed: {exc}")
        return None
    try:
        return resp.json()
    except ValueError:
        return None


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------
def cmd_status(token: str) -> int:
    info = _call(token, "getWebhookInfo")
    print("Registered webhook:")
    print(summarise_webhook_info(info))
    return 0


def cmd_delete(token: str) -> int:
    _call(token, "deleteWebhook", {"drop_pending_updates": False})
    print("Webhook removed.")
    return 0


def cmd_register(args: argparse.Namespace, token: str, secret: str | None) -> int:
    url = build_webhook_url(args.url)

    me = _call(token, "getMe")
    print(f"bot              : @{me.get('username')} ({me.get('first_name')})")
    print(f"webhook url      : {url}")
    print(f"webhook secret   : {fingerprint(secret)}")

    # Pre-flight: registering a webhook at a service that has no bot token would
    # make every delivery fail with 503. Check first, unless told not to.
    if not args.no_preflight:
        health = service_health(args.url)
        print(f"service /health  : {json.dumps(health) if health else 'unreachable'}")
        if health is None:
            print("  !! service did not answer /health; continuing anyway (it may be cold-starting)")
        elif not health.get("telegram_enabled"):
            message = (
                "telegram_enabled is FALSE on the service: the env push has not "
                "applied yet, so deliveries would be rejected with 503. "
                "Re-run scripts/push_env.py and wait for the deploy to finish, "
                "or pass --no-preflight to override."
            )
            if not args.force:
                print(f"  !! {message}")
                return 2
            print(f"  !! {message} (overridden by --force)")

    payload: dict[str, Any] = {
        "url": url,
        "allowed_updates": ALLOWED_UPDATES,
        # False by default: anything the bot has already been sent is delivered
        # rather than silently discarded. Pass --drop-pending to discard it.
        "drop_pending_updates": bool(args.drop_pending),
    }
    if secret:
        payload["secret_token"] = secret

    _call(token, "setWebhook", payload)
    print("setWebhook       : ok")

    info = _call(token, "getWebhookInfo")
    print("\nAfter registration:")
    print(summarise_webhook_info(info))

    if info.get("last_error_message"):
        print("\nWARNING: Telegram reports a delivery error -- the service is not accepting updates.")
        return 1
    print("\nWebhook is registered and Telegram reports no delivery errors.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=os.environ.get("RENDER_EXTERNAL_URL") or os.environ.get("PUBLIC_BASE_URL"),
                        help="public base URL of the deployed service (or RENDER_EXTERNAL_URL)")
    parser.add_argument("--status", action="store_true", help="show getWebhookInfo and exit")
    parser.add_argument("--delete", action="store_true", help="remove the webhook and exit")
    parser.add_argument("--drop-pending", action="store_true",
                        help="discard updates Telegram queued while no webhook was set")
    parser.add_argument("--no-preflight", action="store_true", help="skip the /health check")
    parser.add_argument("--force", action="store_true", help="register even if /health says telegram_enabled is false")
    args = parser.parse_args()

    load_dotenv()

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    secret = os.environ.get("TELEGRAM_WEBHOOK_SECRET")

    if not token:
        print("TELEGRAM_BOT_TOKEN is not set (env or .env). Nothing to do.")
        return 1

    if args.status:
        return cmd_status(token)
    if args.delete:
        return cmd_delete(token)

    if not secret:
        print(
            "TELEGRAM_WEBHOOK_SECRET is not set: the webhook will be registered WITHOUT "
            "a secret, so any caller who knows the URL can forge an update. Set it and "
            "redeploy before relying on this in production."
        )
    return cmd_register(args, token, secret)


if __name__ == "__main__":
    sys.exit(main())

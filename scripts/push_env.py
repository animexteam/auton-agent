#!/usr/bin/env python3
"""Push the full environment to both deployed services and trigger a deploy.

Why this exists separately from `deploy_render.py env`: Render's service-creation
payload does not reliably persist `envVars` (observed: a service created with 16
env vars reported 0 via the API). Setting them explicitly after creation is the
reliable path, and it is what this script does.

Secrets are read from the environment and never printed — only names and lengths.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import httpx

API = "https://api.render.com/v1"

SERVICES = {
    1: "srv-darcl0m0tbcc73b2fhtg",
    2: "srv-darcl1btqb8s73evji8g",
}

# Non-secret configuration.
PLAIN = {
    "MODEL_PROVIDER": "ollama_cloud",
    "OLLAMA_BASE_URL": "https://ollama.com",
    "MODEL_PRIMARY": "gpt-oss:120b",
    "MODEL_FALLBACKS": "nemotron-3-ultra,gpt-oss:20b,nemotron-3-nano:30b,nemotron-3-super,gemma4:31b",
    "MODEL_TIMEOUT_SECONDS": "240",
    "TELEGRAM_MODE": "webhook",
    "PERSISTENCE_BACKEND": "chained",
    "SANDBOX_ENABLED": "true",
    "SANDBOX_TIMEOUT_SECONDS": "60",
    "SANDBOX_MAX_MEMORY_MB": "512",
    "SANDBOX_MAX_CPU_SECONDS": "120",
    "SANDBOX_MAX_OUTPUT_BYTES": "200000",
    "SANDBOX_MAX_PROCESSES": "128",
    "AGENT_MAX_STEPS": "40",
    "AGENT_MAX_SECONDS": "900",
    "LOG_LEVEL": "INFO",
    "LOG_JSON": "true",
    "WORKSPACE_ROOT": "/app/workspace",
    "STATE_ROOT": "/app/.agentstate",
    "AGENT_ENVIRONMENT_NOTE": (
        "Render free instance: 0.1 CPU, 512 MB RAM, ephemeral disk, Debian/Linux "
        "container, no persistent storage (durable state is mirrored to a private GitHub Gist)."
    ),
}

# Secrets: deployed name -> environment variables that may supply it, in order
# of preference.
#
# The alias matters: GIST_API_KEY falls back to GITHUB_API_KEY, mirroring
# config.py, which does the same. Render's PUT /env-vars REPLACES the whole set,
# so a secret that is silently skipped here would be DELETED from the running
# service -- which would quietly break durable state on an ephemeral filesystem.
SECRET_SOURCES: dict[str, tuple[str, ...]] = {
    "OLLAMA_API_KEY": ("OLLAMA_API_KEY",),
    "GIST_API_KEY": ("GIST_API_KEY", "GITHUB_API_KEY"),
    "API_AUTH_TOKEN": ("API_AUTH_TOKEN",),
    "TELEGRAM_WEBHOOK_SECRET": ("TELEGRAM_WEBHOOK_SECRET",),
    "TELEGRAM_BOT_TOKEN": ("TELEGRAM_BOT_TOKEN",),
    "TELEGRAM_ALLOWED_USERS": ("TELEGRAM_ALLOWED_USERS",),
}


def load_dotenv() -> None:
    path = Path(".env")
    if not path.exists():
        path = Path("../.env")
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def build_env() -> list[dict[str, str]]:
    env = dict(PLAIN)
    missing = []
    for target, sources in SECRET_SOURCES.items():
        value = None
        for source in sources:
            value = os.environ.get(source)
            if value:
                break
        if value:
            env[target] = value
        else:
            missing.append(target)
    if missing:
        print(f"  not set locally (skipped): {', '.join(missing)}")
        # A skipped secret is not harmless: PUT /env-vars replaces the whole
        # set, so it would be removed from the running service.
        print("  !! anything listed above will be REMOVED from the service unless it is already set there")
    print(f"  pushing {len(env)} variables:")
    for key in sorted(env):
        value = env[key]
        kind = "SECRET" if key in SECRET_SOURCES else "plain"
        print(f"    {key:30s} [{kind}] len={len(value)}")
    return [{"key": k, "value": v} for k, v in env.items()]


def main() -> int:
    load_dotenv()
    env_vars = build_env()
    ok = True
    for account, service_id in SERVICES.items():
        key = os.environ.get(f"RENDER_API_KEY_{account}")
        if not key:
            print(f"[account {account}] no API key; skipped")
            continue
        headers = {"Authorization": f"Bearer {key}", "Accept": "application/json"}
        print(f"\n[account {account}] setting env on {service_id}")
        resp = httpx.put(
            f"{API}/services/{service_id}/env-vars", headers=headers, json=env_vars, timeout=60
        )
        print(f"  PUT env-vars -> http {resp.status_code}")
        if resp.status_code >= 400:
            print(f"  {resp.text[:300]}")
            ok = False
            continue

        # Read back the key NAMES only -- never the values -- so a silently
        # dropped variable is caught here instead of at runtime.
        # limit=100 is required: Render paginates this endpoint at 20 items by
        # default, so a plain GET reports the surplus keys as "missing".
        check = httpx.get(
            f"{API}/services/{service_id}/env-vars?limit=100", headers=headers, timeout=60
        )
        if check.status_code < 400:
            try:
                landed = {
                    item.get("envVar", item).get("key")
                    for item in check.json()
                    if isinstance(item, dict)
                }
            except (ValueError, AttributeError):
                landed = set()
            expected = {item["key"] for item in env_vars}
            absent = sorted(expected - landed)
            print(f"  read back {len(landed)} keys; missing: {absent if absent else 'none'}")
            if absent:
                ok = False
        else:
            print(f"  read-back failed -> http {check.status_code}")

        deploy = httpx.post(
            f"{API}/services/{service_id}/deploys",
            headers=headers,
            json={"clearCache": "clear"},
            timeout=60,
        )
        print(f"  trigger deploy -> http {deploy.status_code}")
        if deploy.status_code < 400:
            print(f"  deploy id: {deploy.json().get('id')}")
        else:
            print(f"  {deploy.text[:300]}")
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

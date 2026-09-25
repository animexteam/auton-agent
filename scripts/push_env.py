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
    "MODEL_PRIMARY": "glm-5.2",
    "MODEL_FALLBACKS": "gpt-oss:20b,nemotron-3-nano:30b,gemma4:31b",
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

# Secrets: name -> environment variable holding the value.
SECRET_SOURCES = {
    "OLLAMA_API_KEY": "OLLAMA_API_KEY",
    "GIST_API_KEY": "GIST_API_KEY",
    "API_AUTH_TOKEN": "API_AUTH_TOKEN",
    "TELEGRAM_WEBHOOK_SECRET": "TELEGRAM_WEBHOOK_SECRET",
    "TELEGRAM_BOT_TOKEN": "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_ALLOWED_USERS": "TELEGRAM_ALLOWED_USERS",
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
    for target, source in SECRET_SOURCES.items():
        value = os.environ.get(source)
        if value:
            env[target] = value
        else:
            missing.append(target)
    if missing:
        print(f"  not set locally (skipped): {', '.join(missing)}")
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

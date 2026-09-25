#!/usr/bin/env python3
"""Deploy and manage auton-agent on Render, programmatically.

Usage
-----
    python scripts/deploy_render.py plan
    python scripts/deploy_render.py deploy --account 1 --repo https://github.com/<user>/auton-agent
    python scripts/deploy_render.py deploy-both --repo https://github.com/<user>/auton-agent
    python scripts/deploy_render.py status --account 1
    python scripts/deploy_render.py logs --account 1
    python scripts/deploy_render.py env --account 1 --set KEY=VALUE [--set K2=V2 ...]
    python scripts/deploy_render.py webhook --account 1 --public-url https://<svc>.onrender.com
    python scripts/deploy_render.py suspend --account 1
    python scripts/deploy_render.py resume --account 1
    python scripts/deploy_render.py delete --account 1 --yes

Why this script exists
----------------------
The Render dashboard is fine for a human, but "create and manage the project
programmatically" means the service definition has to live in code. This script
is that definition, and it is idempotent: running `deploy` twice updates the
existing service instead of creating a duplicate.

Secrets are never hard-coded. They are read from the environment (or a local
.env), and the *names* — never the values — are printed.

Free-plan facts this script encodes
-----------------------------------
  * free web services spin down after 15 minutes idle, so a Telegram *webhook*
    is used rather than long-polling;
  * the filesystem is ephemeral and free services cannot mount a disk, so durable
    state is mirrored to a Gist;
  * 750 instance-hours per workspace per month, shared across free services.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx

RENDER_API = "https://api.render.com/v1"
SERVICE_NAME = "auton-agent"

#: Environment variables the deployed service needs. Values come from the local
#: environment; only the NAMES are ever printed.
SECRET_KEYS = (
    "OLLAMA_API_KEY",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_WEBHOOK_SECRET",
    "TELEGRAM_ALLOWED_USERS",
    "API_AUTH_TOKEN",
    "GIST_ID",
    "GIST_API_KEY",
)

PLAIN_ENV = {
    "MODEL_PROVIDER": "ollama_cloud",
    "OLLAMA_BASE_URL": "https://ollama.com",
    "MODEL_PRIMARY": "gpt-oss:120b",
    "MODEL_FALLBACKS": "nemotron-3-ultra,gpt-oss:20b,nemotron-3-nano:30b,nemotron-3-super,gemma4:31b",
    "TELEGRAM_MODE": "webhook",
    "PERSISTENCE_BACKEND": "chained",
    "SANDBOX_ENABLED": "true",
    "SANDBOX_TIMEOUT_SECONDS": "60",
    "SANDBOX_MAX_MEMORY_MB": "512",
    "AGENT_MAX_STEPS": "40",
    "AGENT_MAX_SECONDS": "900",
    "LOG_LEVEL": "INFO",
}


# --------------------------------------------------------------------------
# tiny .env loader (no dependency)
# --------------------------------------------------------------------------
def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def api_key(account: int) -> str:
    key = os.environ.get(f"RENDER_API_KEY_{account}")
    if not key:
        sys.exit(
            f"error: RENDER_API_KEY_{account} is not set. "
            f"Export it or put it in .env before running this script."
        )
    return key


class Render:
    """Minimal, typed wrapper over the Render REST API."""

    def __init__(self, key: str) -> None:
        self._client = httpx.Client(
            base_url=RENDER_API,
            headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
            timeout=60.0,
        )

    def close(self) -> None:
        self._client.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        resp = self._client.request(method, path, **kwargs)
        if resp.status_code >= 400:
            raise SystemExit(
                f"render api {method} {path} failed: http {resp.status_code} {resp.text[:400]}"
            )
        if not resp.content:
            return None
        return resp.json()

    # -- owners / services ---------------------------------------------

    def owners(self) -> list[dict[str, Any]]:
        data = self._request("GET", "/owners", params={"limit": 20}) or []
        return [item.get("owner", item) for item in data]

    def services(self) -> list[dict[str, Any]]:
        data = self._request("GET", "/services", params={"limit": 100}) or []
        return [item.get("service", item) for item in data]

    def find_service(self, name: str = SERVICE_NAME) -> dict[str, Any] | None:
        for service in self.services():
            if service.get("name") == name:
                return service
        return None

    def create_service(
        self,
        *,
        owner_id: str,
        repo_url: str,
        branch: str = "main",
        name: str = SERVICE_NAME,
        dockerfile_path: str = "./Dockerfile",
    ) -> dict[str, Any]:
        payload = {
            "type": "web_service",
            "name": name,
            "ownerId": owner_id,
            "repo": repo_url,
            "branch": branch,
            "autoDeploy": "yes",
            "serviceDetails": {
                "env": "docker",
                "plan": "free",
                "region": "oregon",
                "healthCheckPath": "/health",
                "dockerfilePath": dockerfile_path,
                "envVars": _env_list(),
            },
        }
        return self._request("POST", "/services", json=payload)

    def update_env(self, service_id: str, env_vars: dict[str, str]) -> None:
        self._request(
            "PUT",
            f"/services/{service_id}/env-vars",
            json=[{"key": k, "value": v} for k, v in env_vars.items()],
        )

    def get_env(self, service_id: str) -> list[dict[str, Any]]:
        return self._request("GET", f"/services/{service_id}/env-vars") or []

    def trigger_deploy(self, service_id: str) -> Any:
        return self._request("POST", f"/services/{service_id}/deploys", json={"clearCache": "do_not_clear"})

    def deploys(self, service_id: str, limit: int = 5) -> list[dict[str, Any]]:
        data = self._request("GET", f"/services/{service_id}/deploys", params={"limit": limit}) or []
        return [item.get("deploy", item) for item in data]

    def logs(self, owner_id: str, service_id: str, limit: int = 100) -> list[dict[str, Any]]:
        try:
            data = self._request(
                "GET",
                f"/logs",
                params={"ownerId": owner_id, "resource": service_id, "limit": limit},
            )
        except SystemExit as exc:
            return [{"error": str(exc)}]
        return data.get("logs", data) if isinstance(data, dict) else data

    def suspend(self, service_id: str) -> Any:
        return self._request("POST", f"/services/{service_id}/suspend")

    def resume(self, service_id: str) -> Any:
        return self._request("POST", f"/services/{service_id}/resume")

    def delete(self, service_id: str) -> Any:
        return self._request("DELETE", f"/services/{service_id}")


def _env_list(override: dict[str, str] | None = None) -> list[dict[str, str]]:
    """Build the env-var list, taking secret VALUES from the environment."""
    env: dict[str, str] = dict(PLAIN_ENV)
    missing: list[str] = []
    for key in SECRET_KEYS:
        value = os.environ.get(key)
        if value:
            env[key] = value
        else:
            missing.append(key)
    if override:
        env.update(override)
    if missing:
        print(f"  note: not set locally, so not deployed: {', '.join(missing)}")
    print("  env vars deployed (names only): " + ", ".join(sorted(env)))
    return [{"key": k, "value": v} for k, v in env.items()]


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------
def cmd_plan(args: argparse.Namespace) -> int:
    print("auton-agent -> Render deployment plan")
    for account in (1, 2):
        key = os.environ.get(f"RENDER_API_KEY_{account}")
        if not key:
            print(f"  account {account}: RENDER_API_KEY_{account} not set (skipped)")
            continue
        client = Render(key)
        try:
            owners = client.owners()
            who = ", ".join(f"{o.get('name')} <{o.get('email')}>" for o in owners[:3])
            existing = [s for s in client.services() if s.get("type") == "web_service"]
            print(f"  account {account}: {who}")
            print(f"    existing web services: {len(existing)}")
            for service in existing[:6]:
                plan = (service.get("serviceDetails") or {}).get("plan", "?")
                print(f"      - {service.get('name')} [{plan}] id={service.get('id')}")
            print(f"    will use free plan; durable state via Gist (free disks unsupported)")
        finally:
            client.close()
    return 0


def _resolve_repo(args: argparse.Namespace) -> str:
    repo = args.repo or os.environ.get("RENDER_REPO_URL")
    if not repo:
        sys.exit("error: pass --repo https://github.com/<user>/auton-agent (or set RENDER_REPO_URL)")
    if repo.endswith(".git"):
        repo = repo[:-4]
    return repo


def _deploy_one(account: int, repo: str, extra_env: dict[str, str], branch: str) -> dict[str, Any]:
    key = api_key(account)
    client = Render(key)
    try:
        owners = client.owners()
        if not owners:
            sys.exit(f"account {account}: no owner found for this API key")
        owner_id = os.environ.get(f"RENDER_OWNER_ID_{account}") or owners[0]["id"]
        print(f"[account {account}] owner={owners[0].get('name')} id={owner_id}")

        existing = client.find_service()
        if existing:
            service = existing
            print(f"[account {account}] service exists: {service['id']} — updating env vars")
            client.update_env(service["id"], dict(PLAIN_ENV))
            client.trigger_deploy(service["id"])
        else:
            print(f"[account {account}] creating web service '{SERVICE_NAME}' from {repo}")
            service = client.create_service(
                owner_id=owner_id, repo_url=repo, branch=branch
            )
        return {"account": account, "owner_id": owner_id, "service": service}
    finally:
        client.close()


def cmd_deploy(args: argparse.Namespace) -> int:
    repo = _resolve_repo(args)
    result = _deploy_one(args.account, repo, {}, args.branch)
    service = result["service"]
    service_id = service.get("id")
    print(f"\nservice id : {service_id}")
    print(f"service url: {service.get('serviceDetails', {}).get('url') or service.get('url')}")
    print("\nNow set the secrets in the Render dashboard (or re-run with `env`), then:")
    print(f"  python {sys.argv[0]} status --account {args.account}")
    return 0


def cmd_deploy_both(args: argparse.Namespace) -> int:
    repo = _resolve_repo(args)
    results = []
    for account in (1, 2):
        if not os.environ.get(f"RENDER_API_KEY_{account}"):
            print(f"[account {account}] API key not set — switched off, skipping")
            continue
        results.append(_deploy_one(account, repo, {}, args.branch))
    print("\n=== SUMMARY ===")
    for item in results:
        service = item["service"]
        print(f"account {item['account']}: {service.get('id')} "
              f"{service.get('serviceDetails', {}).get('url') or service.get('url')}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    key = api_key(args.account)
    client = Render(key)
    try:
        service = client.find_service()
        if not service:
            print(f"account {args.account}: no service named {SERVICE_NAME}")
            return 1
        details = service.get("serviceDetails") or {}
        print(f"name       : {service.get('name')}")
        print(f"id         : {service.get('id')}")
        print(f"plan       : {details.get('plan')}")
        print(f"region     : {details.get('region')}")
        print(f"suspended  : {service.get('suspended')}")
        print(f"autoDeploy : {service.get('autoDeploy')}")
        print(f"url        : {details.get('url') or service.get('url')}")
        defines = client.deploys(service["id"], limit=3)
        print("recent deploys:")
        for deploy in defines:
            print(f"  - {deploy.get('id')} {deploy.get('status')} "
                  f"commit={str(deploy.get('commit', {}).get('id'))[:8]}")
        return 0
    finally:
        client.close()


def cmd_logs(args: argparse.Namespace) -> int:
    key = api_key(args.account)
    client = Render(key)
    try:
        service = client.find_service()
        if not service:
            sys.exit(f"account {args.account}: service not found")
        owners = client.owners()
        owner_id = owners[0]["id"]
        entries = client.logs(owner_id, service["id"], limit=args.lines)
        for entry in entries:
            if isinstance(entry, dict) and "error" in entry:
                print(entry["error"])
            else:
                print(json.dumps(entry, default=str)[:400])
        return 0
    finally:
        client.close()


def cmd_env(args: argparse.Namespace) -> int:
    key = api_key(args.account)
    client = Render(key)
    try:
        service = client.find_service()
        if not service:
            sys.exit(f"account {args.account}: service not found")
        if args.set:
            env = {}
            for item in args.set:
                if "=" not in item:
                    sys.exit(f"error: --set expects KEY=VALUE, got {item!r}")
                k, _, v = item.partition("=")
                env[k.strip()] = v.strip()
            client.update_env(service["id"], env)
            print(f"updated {len(env)} env var(s): {', '.join(sorted(env))}")
            client.trigger_deploy(service["id"])
            print("triggered a deploy to pick up the change")
        else:
            current = client.get_env(service["id"])
            print("configured env vars (names shown; secret values masked):")
            for item in current:
                var = item.get("envVar", item)
                key_name = var.get("key")
                masked = "***" if key_name in SECRET_KEYS else var.get("value")
                print(f"  {key_name} = {masked}")
        return 0
    finally:
        client.close()


def cmd_webhook(args: argparse.Namespace) -> int:
    """Register the Telegram webhook against the deployed URL."""
    import urllib.parse
    import urllib.request

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        sys.exit("error: TELEGRAM_BOT_TOKEN is not set locally")
    url = args.public_url or os.environ.get("RENDER_EXTERNAL_URL")
    if not url:
        sys.exit("error: pass --public-url https://<service>.onrender.com")
    target = f"{url.rstrip('/')}/telegram/webhook"

    body = {"url": target, "allowed_updates": ["message", "edited_message"],
            "drop_pending_updates": True}
    secret = os.environ.get("TELEGRAM_WEBHOOK_SECRET")
    if secret:
        body["secret_token"] = secret
    payload = urllib.parse.urlencode(body).encode()
    request = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/setWebhook", data=payload
    )
    with urllib.request.urlopen(request, timeout=30) as resp:
        result = json.loads(resp.read())
    print(f"setWebhook -> {json.dumps(result)}")
    print(f"webhook target: {target}")
    return 0 if result.get("ok") else 1


def cmd_suspend(args: argparse.Namespace) -> int:
    client = Render(api_key(args.account))
    try:
        service = client.find_service()
        if not service:
            sys.exit("service not found")
        print(f"suspend -> {client.suspend(service['id'])}")
        return 0
    finally:
        client.close()


def cmd_resume(args: argparse.Namespace) -> int:
    client = Render(api_key(args.account))
    try:
        service = client.find_service()
        if not service:
            sys.exit("service not found")
        print(f"resume -> {client.resume(service['id'])}")
        return 0
    finally:
        client.close()


def cmd_delete(args: argparse.Namespace) -> int:
    if not args.yes:
        sys.exit("refusing to delete without --yes")
    client = Render(api_key(args.account))
    try:
        service = client.find_service()
        if not service:
            sys.exit("service not found")
        client.delete(service["id"])
        print(f"deleted {service['id']}")
        return 0
    finally:
        client.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage auton-agent on Render")
    parser.add_argument("--env-file", default=".env", help="path to a .env file to load")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("plan", help="show what would be deployed, per account")

    p_deploy = sub.add_parser("deploy", help="create or update the service on one account")
    p_deploy.add_argument("--account", type=int, choices=(1, 2), default=1)
    p_deploy.add_argument("--repo", default=None)
    p_deploy.add_argument("--branch", default="main")

    p_both = sub.add_parser("deploy-both", help="deploy to both Render accounts")
    p_both.add_argument("--repo", default=None)
    p_both.add_argument("--branch", default="main")

    p_status = sub.add_parser("status", help="show service status and recent deploys")
    p_status.add_argument("--account", type=int, choices=(1, 2), default=1)

    p_logs = sub.add_parser("logs", help="fetch recent service logs")
    p_logs.add_argument("--account", type=int, choices=(1, 2), default=1)
    p_logs.add_argument("--lines", type=int, default=100)

    p_env = sub.add_parser("env", help="list or set environment variables")
    p_env.add_argument("--account", type=int, choices=(1, 2), default=1)
    p_env.add_argument("--set", action="append", help="KEY=VALUE (repeatable)")

    p_hook = sub.add_parser("webhook", help="register the Telegram webhook")
    p_hook.add_argument("--account", type=int, choices=(1, 2), default=1)
    p_hook.add_argument("--public-url", default=None)

    for name, help_text in (("suspend", "suspend the service"), ("resume", "resume the service")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--account", type=int, choices=(1, 2), default=1)

    p_delete = sub.add_parser("delete", help="delete the service")
    p_delete.add_argument("--account", type=int, choices=(1, 2), default=1)
    p_delete.add_argument("--yes", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv(Path(args.env_file))
    table = {
        "plan": cmd_plan,
        "deploy": cmd_deploy,
        "deploy-both": cmd_deploy_both,
        "status": cmd_status,
        "logs": cmd_logs,
        "env": cmd_env,
        "webhook": cmd_webhook,
        "suspend": cmd_suspend,
        "resume": cmd_resume,
        "delete": cmd_delete,
    }
    return table[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())

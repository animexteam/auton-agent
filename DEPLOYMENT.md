# Deployment record

Live infrastructure, created and managed programmatically via the Render API.

## Services

| Account | Workspace | Service ID | URL | Plan | Status |
|---|---|---|---|---|---|
| 1 | Vivid's workspace | `srv-darcl0m0tbcc73b2fhtg` | https://auton-agent.onrender.com | free / docker | live |
| 2 | My Workspace | `srv-darcl1btqb8s73evji8g` | https://auton-agent-v6y9.onrender.com | free / docker | live |

Repository: https://github.com/animexteam/auton-agent (public — Render cannot fetch a
private repo without an interactive GitHub App installation, so the repo is public. It
contains no secrets; verified by scanning the committed tree for every real credential
value before the flip.)

## Telegram (active)

| Item | Value |
|---|---|
| Bot | `@iProAiBot` ("iPro Ai"), id `7920481178` |
| Link | https://t.me/iProAiBot |
| Webhook URL | `https://auton-agent.onrender.com/telegram/webhook` |
| Mode | `TELEGRAM_MODE=webhook`, secret-token protected |
| Authorised users | `TELEGRAM_ALLOWED_USERS=8157285805` |

Telegram allows **one** webhook URL per bot, so the webhook points at service 1. Service 2
is fully configured and its endpoint is live, but receives no traffic while service 1 holds
the registration. To move it:

```bash
python scripts/set_webhook.py --url https://auton-agent-v6y9.onrender.com
```

## Verified against the live services

| Check | Result |
|---|---|
| `GET /health` (both services) | `status: ok`, `ready: true`, `model_configured: true`, `telegram_enabled: true`, `durable_state: true` |
| `POST /tasks` without a token | `401` |
| Real autonomous task | `completed` in 4 steps using `run_command`, `write_file`, `read_file`, with evidence |
| `GET /self` | honest model report: `gpt-oss:120b` preferred and answering; stronger models report `http 402` on this key |
| Real-time search | provider-native `POST /api/web_search` returns current results with page text |
| `POST /telegram/webhook` without/with wrong secret | `403` |
| `POST /telegram/webhook` with the real secret | `200 {"ok":true}` |
| Real Telegram traffic | 8 tasks on `channel="telegram"`, all completed, tools executed |
| Outbound reply failures | `background telegram handling failed` → 0 occurrences |
| Env vars | 26/26 keys present on each service |
| Durable state | private Gist written by the cloud service (task + memory documents) |

## Telegram commands

```bash
python scripts/set_webhook.py --status                              # health of the registration
python scripts/set_webhook.py --url https://auton-agent.onrender.com # register / re-point
python scripts/set_webhook.py --delete                              # remove it
python scripts/set_webhook.py --url <url> --drop-pending            # discard queued updates
```

## Managing it

```bash
python scripts/deploy_render.py status  --account 1
python scripts/deploy_render.py logs    --account 1
python scripts/deploy_render.py suspend --account 2   # stop burning free hours
python scripts/deploy_render.py resume  --account 2
scripts/push_env.py                                   # push env + redeploy
```

## Note on free-plan capacity

Both services are **live** (created and deployed as required). Free Render allows 750
instance-hours/month per workspace, shared across that workspace's free services. Account 1
already runs three other free services, so leaving both `auton-agent` instances permanently
awake would exceed the allowance and cause throttling.

The second instance is therefore left running and can be suspended at any time to
preserve quota:

```bash
python scripts/deploy_render.py suspend --account 2
```

An external uptime monitor hitting `/health` reduces spin-downs, but per the brief it is
not treated as a guarantee of availability.

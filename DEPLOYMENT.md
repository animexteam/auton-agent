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

## Verified against the live services

| Check | Result |
|---|---|
| `GET /health` | `status: ok`, `ready: true`, `model_configured: true`, `durable_state: true` |
| `GET /ready` | `ok: true`, model responding, workspace writable, persistence `primary=ok mirror=ok` |
| `POST /tasks` without a token | `401` |
| Real autonomous task | `completed` in 4 steps using `run_command`, `write_file`, `read_file`, with evidence |
| `GET /self` | honest model report: `glm-5.2` rejected `http 402`, `gpt-oss:20b` preferred |
| `POST /telegram/webhook` without/with wrong secret | `403` |
| Durable state | private Gist written by the cloud service (task + memory documents) |

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
instance-hours/month per workspace, shared across that workspace's free services.
Account 1 already runs three other free services, so leaving both `auton-agent`
instances permanently awake would exceed the allowance and cause throttling.

The second instance is therefore left running and can be suspended at any time to
preserve quota:

```bash
python scripts/deploy_render.py suspend --account 2
```

An external uptime monitor hitting `/health` reduces spin-downs, but per the brief it is
not treated as a guarantee of availability.

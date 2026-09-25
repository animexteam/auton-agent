# auton-agent

A lightweight autonomous AI agent that runs as a Telegram bot on **Render's free tier**.

It is not a chatbot with a few glued-on functions. It receives an objective, decides how
to achieve it, uses tools, inspects what actually happened, adapts when things break, and
keeps going until the job is done or it hits a genuine boundary.

```
UNDERSTAND -> PLAN -> ACT -> OBSERVE -> EVALUATE -> ADAPT -> ACT AGAIN -> VERIFY -> COMPLETE
```

---

## What it can actually do

Verified end-to-end, not aspirational:

| Capability | How it is provided |
|---|---|
| Reason over multi-step objectives | Agent loop with tool-calling against Ollama Cloud |
| Run shell commands | Hardened sandbox: rlimits, timeout, isolated env, destructive-command screening |
| Read / write / search files | Path-guarded to the workspace |
| Execute code | `run_python` — script written to a temp file, run under the same limits, cleaned up |
| Research the web | `web_search`, `fetch_url`, `download_file` with untrusted-content fencing |
| Remember across restarts | Disk + private GitHub **Gist** mirror; the agent creates its own gist |
| Keep task state across steps | Working memory, scratchpad, plan, progress log, checkpoints |
| Load domain know-how | 11 skills, progressive disclosure — index always visible, body on demand |
| Report on itself honestly | `self_report` reads the live process, not a hardcoded description |
| Survive redeploys | State recovered from the Gist after the local disk is wiped |

## Architecture

```
                Telegram  ──┐
                            ├──►  interfaces  ──►  Agent (loop.py)  ──►  ModelRouter ──► Ollama Cloud
                HTTP / CLI ─┘                          │                                GLM 5.2 -> fallbacks
                                                       │
                          ┌────────────────────────────┼────────────────────────────┐
                          ▼                            ▼                            ▼
                    ToolRegistry                  MemoryBundle                  Observability
                  (17 tools, typed             (working / task /             (event log,
                   contracts, gate)              long-term / conv)             redacted logs)
                                                       │
                                                       ▼
                                                  StateStore
                                          disk  ◄─►  GitHub Gist
```

Design decisions worth stating, because they are deliberate rather than default:

**No framework.** No LangChain, no AutoGen, no agent SDK. The loop is ~600 readable lines.
On a 0.1-CPU instance, a framework's import cost and abstraction layers buy nothing and cost
cold-start seconds.

**Ollama Cloud, not self-hosted.** The task forbids self-hosting an LLM on Render, and free
tier could not run one anyway. The model layer is one module behind a `ModelProvider`
interface — swapping to any OpenAI-compatible endpoint is a single new file.

**Webhook, not polling.** A free Render service spins down after ~15 minutes idle. A webhook
wakes it on demand; long-polling would require an always-on process and burn the 750-hour
monthly allowance.

**Gist for durability, disk for speed.** A free service gets no persistent disk and its
filesystem is wiped on every deploy. Writes hit local disk (fast, always works) and are
mirrored to a private Gist. `PERSISTENCE_BACKEND` swaps the strategy without touching call sites.

**Tool results are data, never instructions.** Web content is wrapped in
`<untrusted-content>` markers and the system prompt tells the model that anything inside is
evidence to evaluate, never a command to obey. This is the prompt-injection boundary.

## Repository layout

```
src/agentcore/
  loop.py            the agentic loop — the only place that reasons about steps
  prompt.py          system prompt + untrusted-content fencing
  registry.py        tool contracts, validation, structured errors, approval gate
  sandbox.py         subprocess execution with rlimits and command screening
  security.py        authz, rate limiting, path guard, approval tokens, webhook verify
  persistence.py     StateStore: Disk / Gist / Chained
  memory.py          the five kinds of state, kept distinct on purpose
  runtime.py         wiring + startup preflight (reports what genuinely works)
  service.py         FastAPI: /health /ready /tasks /self /events /approvals /telegram/webhook
  main.py, cli.py    service and CLI entrypoints
  llm/               provider-neutral model interface + Ollama Cloud + router
  tools/             fs, exec, web, state, meta
  skills/            10 built-in skills + a library/ directory for on-disk ones
scripts/deploy_render.py   create/manage the Render service programmatically
tests/                     91 tests
```

## Quick start

```bash
pip install -r requirements.txt
pip install pytest pytest-asyncio        # test-only
cp .env.example .env                     # then fill in the values
python3 -m pytest -q                     # 91 passed
```

Run a task locally:

```bash
export PYTHONPATH=src && set -a && . .env && set +a
python3 -m agentcore.cli run "Count the Python files under src/ and write the number to count.txt"
python3 -m agentcore.cli self            # what the agent actually is right now
python3 -m agentcore.cli serve           # HTTP service
```

## Deployment

Programmatic, idempotent, and safe to re-run:

```bash
python scripts/deploy_render.py plan
python scripts/deploy_render.py deploy-both --repo https://github.com/<user>/auton-agent
python scripts/deploy_render.py status --account 1
python scripts/deploy_render.py webhook --account 1 --public-url https://<svc>.onrender.com
python scripts/deploy_render.py suspend --account 2      # stop burning free hours
```

`render.yaml` is the dashboard-driven equivalent.

## Operating within Render Free

Free tier is best-effort, not guaranteed infrastructure. What is done about it:

| Constraint | Mitigation |
|---|---|
| Spins down after ~15 min idle, ~1 min cold start | Webhook interface; the agent's state is external so a cold start loses nothing |
| Ephemeral filesystem | Gist mirror; on boot the agent re-attaches to its existing state gist |
| 750 instance-hours/month, shared per workspace | `suspend`/`resume` commands; only the primary account needs to run continuously |
| 512 MB RAM, 0.1 CPU | No framework, 5 runtime dependencies, single worker, CPU/memory rlimits on every child process |
| No persistent disks | Gist; workspace artefacts are also copied into durable state when referenced |

**Cold starts are handled explicitly:** the webhook endpoint acknowledges Telegram within
milliseconds and runs the task in the background, so a cold start never causes a retry storm.

## Security posture

- **Allowlist.** Only Telegram user IDs in `TELEGRAM_ALLOWED_USERS` may use the agent.
  Empty allowlist = everyone denied. This is the safe default, not an oversight.
- **API auth.** `/tasks`, `/self`, `/events`, `/approvals` require `Authorization: Bearer`.
  `/health` and `/ready` are public and leak nothing (no secrets, no host paths).
- **Approval boundary.** Genuinely destructive commands (`rm -rf /`, `mkfs`, `reboot`) stop
  and return `blocked` with a request for approval. A fork bomb is refused outright and no
  token can unlock it. Ordinary commands (`ls`, `git`, `pytest`, `curl`) run without a
  per-call prompt — otherwise the shell would be useless.
- **Resource limits.** Every child process gets CPU, memory, process-count and file-size
  rlimits, a wall-clock timeout and a hard cap on captured output.
- **Path confinement.** All file tools resolve through a `PathGuard` that rejects traversal
  and absolute escapes.
- **Secret hygiene.** A central redactor scrubs every registered secret from logs, event
  records, tool results and error messages. Secrets are never placed in a model prompt.
- **Least privilege.** The agent gets no Render or GitHub master credentials. Its Gist token
  is the only external credential it holds, and only because durable state needs it.

## Known limitations

Stated plainly, because the agent is required to be accurate about itself:

- **GLM 5.2 is configured but not reachable on this Ollama Cloud key.** The tier does not
  include it, so the router serves `gpt-oss:20b` and reports which model actually answered.
  With an entitled key, GLM 5.2 is used automatically with no code change.
- **No sub-agent fan-out yet.** The architecture has the seam (`Registry`, isolated task
  scope) but parallel workers are not implemented; on 0.1 CPU they would not pay off.
- **No browser automation.** HTTP fetching only; no JS-rendered pages.
- **Cold starts are ~1 minute** on free tier. Inherent to the plan.
- **Single-user by design.** Tasks are serialised under a run lock; this is a personal agent,
  not a shared service.

## Credential configuration

Every credential the system genuinely needs, with values left blank, is listed in
**[CREDENTIALS.md](CREDENTIALS.md)**.

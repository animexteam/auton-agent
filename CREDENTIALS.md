========================================
CREDENTIALS / SECRETS
========================================

Every credential the finished system genuinely requires.
All values are intentionally blank. Nothing here is invented.

Store them as environment variables (Render dashboard, or a local `.env`).
`.env` is gitignored — no secret ever enters source control.


MODEL PROVIDER  (required)

OLLAMA_API_KEY=
    Ollama Cloud API key — the agent's reasoning layer.
    Get it from https://ollama.com/settings/keys


TELEGRAM  (required to use the bot)

TELEGRAM_BOT_TOKEN=
    Bot token from @BotFather. Without it the agent still runs and serves its
    HTTP API; only the Telegram channel is disabled.

TELEGRAM_ALLOWED_USERS=
    Comma-separated numeric Telegram user IDs permitted to use the agent.
    Get your own id from @userinfobot. LEAVE THIS EMPTY AND EVERY MESSAGE IS
    DENIED — that is the safe default. This is the primary access control.

TELEGRAM_WEBHOOK_SECRET=
    Optional but recommended. Any random string. Telegram echoes it back on every
    webhook call so forged requests are rejected.
    (Generate: openssl rand -hex 32)


PERSISTENT STORAGE  (required — the filesystem is ephemeral)

GIST_API_KEY=
    GitHub token with the `gist` scope, used for the durable state mirror.
    A classic token with ONLY `gist` is sufficient for the running agent.

GIST_ID=
    LEAVE BLANK. The agent discovers its own state gist on boot, or creates one,
    and caches the id. Set this only to re-attach a specific gist after a wipe.
    (The task explicitly asks that the gist not be hardcoded.)


HTTP API  (required to drive the agent programmatically)

API_AUTH_TOKEN=
    Any long random string. Clients send: Authorization: Bearer <value>
    (Generate: openssl rand -hex 32)


GITHUB  (required for source code and initial repo push — NOT for the running agent)

GITHUB_API_KEY=
    Token with `repo` + `gist` scopes. Used by scripts/deploy_render.py and the
    one-time `git push`. Withheld from the agent at runtime (least privilege).

GITHUB_USERNAME=
    The account that owns the repository.


RENDER  (required for deployment only — NOT for the running agent)

RENDER_API_KEY_1=
    Render API key for the primary account.
    Dashboard -> Account Settings -> API Keys.

RENDER_API_KEY_2=
    Render API key for the second account. The system deploys to both.


========================================
OTHER REQUIRED CONFIGURATION
========================================
Non-secret settings with working defaults. Change only if you have a reason to.

MODEL_PRIMARY=glm-5.2
    Defaults to GLM 5.2 as specified. If the account is not entitled to it the
    router falls back automatically — see MODEL_FALLBACKS. No code change needed.

MODEL_FALLBACKS=gpt-oss:20b,nemotron-3-nano:30b,gemma4:31b
    Tried in order when the primary is unavailable or not entitled.

MODEL_PROVIDER=ollama_cloud
OLLAMA_BASE_URL=https://ollama.com

PERSISTENCE_BACKEND=chained
    chained = disk (fast) + Gist (durable). Use "disk" only where state loss on
    restart is acceptable.

TELEGRAM_MODE=webhook
    webhook for Render (a webhook wakes a spun-down free instance).
    poll only for local development.

TELEGRAM_MAX_REQUESTS_PER_MINUTE=6
    Per-user rate limit. Protects the free model quota and the small CPU.

AGENT_MAX_STEPS=40
AGENT_MAX_SECONDS=900
    Iteration and wall-clock ceilings. These are what stop a runaway loop.

AGENT_ENVIRONMENT_NOTE=
    One line describing the runtime, shown to the model so its self-awareness is
    grounded in reality.

SANDBOX_ENABLED=true
SANDBOX_TIMEOUT_SECONDS=60
SANDBOX_MAX_MEMORY_MB=512
SANDBOX_MAX_CPU_SECONDS=120
SANDBOX_MAX_OUTPUT_BYTES=200000
SANDBOX_MAX_PROCESSES=128
    Execution limits applied to every child process.

ALLOW_DESTRUCTIVE=false
    Keep false. When true, destructive commands (rm -rf /, mkfs, reboot) run
    without asking. Fork bombs are refused either way.

WORKSPACE_ROOT=./workspace
STATE_ROOT=./.agentstate
SKILLS_PATH=
    Colon-separated extra directories to scan for skill files.

LOG_LEVEL=INFO
LOG_JSON=true
    Logs are JSON and always pass through the secret redactor.


========================================
ON THE RENDER FREE PLAN
========================================
- Web services spin down after ~15 minutes without inbound traffic and take about
  a minute to wake. The webhook design makes this tolerable: Telegram's request
  wakes the service, the endpoint acknowledges immediately, and the task runs in
  the background. State is external, so a cold start loses nothing.
- The filesystem is ephemeral and free services cannot mount a disk. This is
  exactly why durable state is mirrored to a Gist.
- 750 instance-hours per workspace per month, shared across free services. Two
  always-on services would exceed this, which is why the second account's service
  is deployed but can be suspended with:
      python scripts/deploy_render.py suspend --account 2
- Free tier is best-effort, not guaranteed 24/7. An uptime monitor can reduce
  spin-downs but must not be treated as a guarantee of availability.

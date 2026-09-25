# Progress / state

Living record of what is built, verified, and known-broken.
Updated as work happens — not a changelog, a status board.

## Status: working, tested, deployed

## Verified in this environment (real output, not assumed)

| Check | Result |
|---|---|
| Test suite | `91 passed` |
| Real multi-step run (files) | wrote `proof.txt`, read it back, reported verified content |
| Real multi-step run (shell) | ran `python --version`, counted files, wrote `facts.md` |
| HTTP API auth | no token -> `401`; with token -> task created |
| End-to-end via HTTP | task reached `completed`, result + tool trace persisted |
| Durable state | local state **deleted**, service restarted, all 6 tasks recovered from the Gist |
| Gist self-provisioning | agent created its own private gist, then re-attached to it (no duplicate) |
| Startup preflight | `/ready` reports model, workspace, persistence, auth posture |
| Approval boundary | destructive command -> `blocked`; ordinary command -> runs |

## Bugs found and fixed during development

Kept because they explain the current design.

1. **Relative-import depth** — root modules used `..`, two-level modules used `.`
   *Cause:* written before ever running. *Fix:* correct depth everywhere; run tests earlier.
2. **Broken triple-quote in `prompt.py`** — module failed to import.
3. **`needs_approval: bool` vs `str`** — the field carries a reason string.
4. **Every shell command demanded approval** — `RunCommandTool` was marked `privileged`,
   so the agent halted for a human before each `ls`.
   *Fix:* two-tier screening — ordinary runs, destructive asks, catastrophic refused.
5. **A granted approval token could not unlock a destructive command** — the gate was
   checked but the token was never passed to the sandbox.
   *Fix:* thread `approval_token` into `Sandbox.run`; always-blocked tier still enforced.
6. **Health check probed the Gist before its id existed** — reported a false "not
   accessible" on a fresh deployment.
   *Fix:* `resolve_gist_id()` before probing.
7. **Empty assistant turns were not recorded** — the step vanished from the trace.
8. **Budget-exhausted tasks were marked `failed`** — they are *interrupted*, so they are
   now recorded as `running` to stay resumable.

## Environment findings

- **Model entitlement, measured.** Walking the catalogue model by model showed only
  `gpt-oss:120b`, `nemotron-3-ultra`, `nemotron-3-super`, `nemotron-3-nano:30b`,
  `gemma4:31b` and `gpt-oss:20b` answer on this key. `glm-5.3`, `glm-5.2`, `kimi-k3`,
  `deepseek-v4-pro:0813`, `minimax-m3` and `mistral-large-3:675b` all return **HTTP 402**
  ("this model is not included in your free usage").
  *A first parallel probe reported `429 too many concurrent requests` for most models — that
  is the account's concurrency ceiling, not an entitlement answer, so the probe had to be
  re-run sequentially. Reporting the 429 as "not entitled" would have been wrong.*
- **Real-time search is a provider endpoint.** `POST /api/web_search` returns current
  results together with extracted page content; it is now the primary search path.
- Free-tier Render: spins down after ~15 min idle, ~1 min cold start, ephemeral disk,
  no persistent disks, 750 instance-hours/month per workspace.

## Next steps (if work continues)

- Add sub-agent fan-out for genuinely parallel research.
- Add a Redis or Postgres `StateStore` backend for higher write rates.
- Add browser automation for JS-rendered pages.
- Wire an uptime monitor against `/health` to reduce free-tier spin-downs.

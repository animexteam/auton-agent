# Architecture

Why the system is built this way. Decisions are stated with their reasoning and their
costs, so a future maintainer can tell a deliberate choice from an accident.

## 1. The agentic loop

`loop.py` is the heart. One objective in, one `RunResult` out.

```
run(objective)
  ├─ authorise principal          → deny before doing any work
  ├─ rate limit                   → per-user, protects the model quota
  ├─ create TaskRecord            → durable, so a restart can resume
  ├─ acquire run lock             → tasks are serialised (see below)
  └─ _run_loop:
       each iteration:
         build messages (system + task + digest + scratchpad + recent turns)
         router.chat(tools)                → ACT
         if no tool calls: final answer → COMPLETE
         else: execute each tool           → OBSERVE
               feed results back as tool messages
               record into working memory  → EVALUATE
         budget check (steps, wall clock)
```

**Why a hand-written loop and not a framework.** The loop is where the interesting
reasoning lives, and it is ~600 lines. A framework adds abstraction between the model and
the tools — precisely the surface where a bug is hardest to see. On a 0.1-CPU instance,
framework imports also cost cold-start seconds on every activation. Neither trade is worth it.

**Why tasks are serialised.** One `asyncio.Lock` per agent. The runtime is a single-user
agent on 512 MB with a shared scratchpad; two concurrent runs would interleave one task's
plan into another's. Serialising is the honest design for this deployment size, and the
lock is a one-line change to revisit if that ever stops being true.

**Why budgets are hard ceilings.** `AGENT_MAX_STEPS` and `AGENT_MAX_SECONDS` are enforced
inside the loop, not merely suggested to the model. An agent with an infinite loop and a
metered API key is a bill, not a feature.

## 2. Model layer

```
ModelProvider (interface)
   └── OllamaCloudProvider   POST /api/chat with tools
ModelRouter
   ├── ordered chain: primary → fallbacks
   ├── retry with backoff on transient errors
   └── health map: marks a model dead after N consecutive failures
```

The loop only ever sees `ModelResponse` — content, tool calls, token usage. Adding an
OpenAI-compatible provider means adding one file, not touching the loop.

**Entitlement is attempted, never assumed.** The task specifies GLM 5.2, so it heads the
chain. This account's tier does not include it; the router discovers that at runtime and
serves the next model that answers. `router.health()` and the `self_report` tool both
report which model actually responded — the agent never claims to be using a model it
could not reach.

## 3. Tool contracts

Every tool declares a typed parameter schema and returns a `ToolResult`:

```python
@dataclass
class ToolResult:
    ok: bool
    content: Any = None
    error: str | None = None
    code: str | None = None      # machine-readable: "timeout", "not_found", ...
    meta: dict = field(default_factory=dict)
    duration_ms: int = 0
```

Two rules make this earn its keep:

1. **A tool never raises into the loop.** Exceptions are caught in the registry and
   converted into `ToolResult(ok=False, code=...)`. The loop can therefore always reason
   about what happened instead of unwinding.
2. **Errors are structured, so recovery is a decision, not a guess.** `code` lets the agent
   distinguish `timeout` (retry smaller) from `validation_error` (fix the arguments) from
   `permission_denied` (stop and ask).

**No keyword→function routing.** Tool descriptions are written for semantic selection —
each states *when* to use it and, where relevant, when not to. The model chooses.

The 17 tools: `run_command`, `run_python`, `read_file`, `write_file`, `edit_file`,
`list_dir`, `search_files`, `fetch_url`, `web_search`, `download_file`, `working_memory`,
`task_state`, `long_term_memory`, `load_skill`, `self_report`, `write_artifact`, plus the
skills index.

## 4. Skills — progressive disclosure

The problem: putting every domain's instructions in the system prompt makes it enormous,
expensive, and *worse* at following any single part of it.

The solution: the agent always sees a one-line index of available skills and their trigger
conditions; the full body is pulled in only via `load_skill` when relevant.

```markdown
---
name: web-research
description: Use when a task needs facts, docs or prices from the live web.
---
## Steps
1. ...
```

Ten built-in skills ship with the code (task-planning, web-research, shell-and-code,
debugging, verification, git-github, deployment, data-processing, security,
context-management). Operators and the agent itself can drop more into
`<workspace>/skills` or `SKILLS_PATH`; the loader is tolerant — a malformed skill is
skipped with a warning rather than breaking startup.

This is why the system prompt can stay small while the system still grows.

## 5. Memory — five distinct things

The word "memory" hides five requirements with different lifetimes. They are separate types:

| Kind | Lifetime | Backed by |
|---|---|---|
| Conversation | current task | `Conversation` (bounded, trimmed) |
| Working / scratchpad | current task | `WorkingState` — plan, facts, progress |
| Task state | across restarts | `TaskStore` → `StateStore` |
| Long-term memory | forever | `LongTermMemory` → `StateStore` |
| Artefacts | forever | workspace files + durable state |

Separating them is what allows correct behaviour for each: the conversation is trimmed to
protect the context window, while task state is never trimmed because it *is* the record.

### Context management

Long tasks would otherwise grow the prompt without bound. The loop sends:

- the objective and the durable plan (not the whole history),
- a compact digest of completed steps,
- the scratchpad facts the agent chose to record,
- only the most recent turns verbatim.

The scratchpad is the agent's own mechanism for pushing important state out of the context
window and into durable storage, so it can continue after context reduction or a restart.

## 6. Persistence — surviving an ephemeral disk

```
StateStore (interface)
   ├── DiskStore      atomic writes (temp + rename)
   ├── GistStore      one private Gist, one file per document
   └── ChainedStore   disk primary + gist mirror
```

`ChainedStore` writes to disk synchronously (fast, always succeeds) and the mirror
best-effort: a Gist failure is logged, never fatal, because losing durability must not break
the running task.

**The agent owns its gist.** `GIST_ID` is intentionally blank in configuration and the task
requires it not to be hardcoded. On boot `resolve_gist_id()`:

1. uses a configured id if present;
2. else the id cached in `STATE_ROOT` from a previous boot;
3. else **finds** the account's existing gist carrying the `auton-agent durable state`
   marker;
4. else **creates** a private one.

Step 3 before 4 is what makes this idempotent: after a redeploy wipes the cache, the agent
re-attaches to its existing state instead of scattering duplicates.

Why a Gist and not a real database: free Render has no persistent disk, and the task's
state is small documents written at human pace. A single gist keeps it to one request per
write and needs no extra service or credential. A generic high-frequency filesystem is
explicitly *not* what this is used for.

## 7. Execution sandbox

`Sandbox.run()` launches a subprocess with:

- **resource limits** via `preexec_fn` + `setrlimit`: CPU seconds, address space, process
  count, file size;
- **a wall-clock timeout**, after which the whole process group is killed;
- **a sparse child environment** — the parent's secrets are not inherited;
- **output caps** so a runaway `yes` cannot exhaust memory;
- **command screening** in two tiers.

The two-tier screening is a deliberate correction of a first-cut design:

| Tier | Examples | Behaviour |
|---|---|---|
| Catastrophic | fork bomb | `PermissionDenied` — no approval can unlock it |
| Destructive | `rm -rf /`, `mkfs`, `reboot` | `ApprovalRequired` — stops and asks a human |
| Ordinary | `ls`, `git`, `pytest`, `curl` | runs |

The first version marked the shell tool `privileged`, which made the agent halt for a human
token before *every* `ls`. That is not security, it is an unusable agent. Authorisation for
running commands comes from the principal being on the allowlist; the blast radius is
contained by the resource limits. Approval is reserved for actions whose *effect* is
irreversible.

## 8. Security boundaries

- **Authentication** — Telegram allowlist (empty = deny all); `Bearer` token on the API.
- **Authorisation** — `Authorizer` per principal; the loop refuses unauthorised work before
  spending a single model token.
- **Rate limiting** — per-user token bucket; protects both the model quota and the CPU.
- **Webhook verification** — constant-time compare of Telegram's secret header.
- **Path confinement** — `PathGuard` resolves every path and rejects escapes.
- **Secret isolation** — a central `Redactor` scrubs registered secrets from logs, events,
  tool results and error strings. Secrets are never put in a prompt.
- **Untrusted content** — fetched web content is wrapped in `<untrusted-content>` markers
  and the system prompt states that anything inside is evidence, never an instruction.
- **Least privilege** — the agent holds no Render or repository credential. Its only
  external credential is the gist token, needed for durable state.

## 9. Portability

Render is the current provider, not a dependency. `render.yaml` and
`scripts/deploy_render.py` are the only Render-aware files. The application itself needs:
a process that runs `python -m agentcore.main service`, a `PORT`, and outbound HTTPS. That
is satisfied by any container host or a plain VPS (`docker build && docker run`). Nothing in
`agentcore/` imports a hosting SDK.

## 10. Observability

- **Structured JSON logs**, every entry through the redactor.
- **An event log** (`task.start`, `step.end`, `tool.call`, `tool.result`, `task.blocked`,
  `task.complete`) exposed at `GET /events`, so a run can be reconstructed after the fact.
- **`GET /self`** — the honest self-report: which model answered, which tools are registered,
  what the limits are, what failed recently.
- **Startup preflight** — probes model reachability, workspace writability and persistence
  health, and returns them from `/ready`. A misconfiguration is visible at boot rather than
  discovered mid-task.

## Costs accepted

Honest accounting of what this design gives up:

- **Serialised tasks** — no concurrency. Correct for a single-user 512 MB agent.
- **No sub-agents** — the seam exists; on 0.1 CPU parallelism would not pay off.
- **No browser automation** — HTTP fetching only, so JS-rendered pages are out of reach.
- **A single Gist for state** — fine at human pace, wrong for high-frequency writes.
- **Cold starts** — ~1 minute on free tier; inherent to the plan, mitigated by external state.

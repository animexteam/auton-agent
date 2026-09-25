---
name: self-modification
description: Use when the task is to change the agent's own code, configuration, or skills — fixing a bug in agentcore, adding a tool, or editing a skill file.
---

## When to use

The objective is to improve, fix or extend *this agent itself*: a defect in
`agentcore`, a new tool, a new or corrected skill file, or a configuration change.

## Before changing anything

1. Read the file you intend to change. Never edit from memory; the on-disk text is
   the only truth.
2. Find every caller: `search_files` for the symbol across `src/` and `tests/`.
3. Identify the invariant you must not break. The non-negotiables in this codebase
   are: secrets never reach logs/prompts/results, the path guard confines file
   access to the workspace, and every tool returns a structured error rather than
   raising through the loop.

## Procedure

1. Change the *smallest* surface that fixes the problem. Prefer editing an existing
   module over adding a new one.
2. Add or update a test that fails before the change and passes after. A fix with no
   test is not a fix.
3. Run the suite: `python3 -m pytest -q`. It must be green.
4. If the change touches the loop, tools or security, run a real end-to-end task
   afterwards — unit tests do not prove the agent still behaves.

## Hard limits

- You may modify project code, skills and configuration.
- You must **not** grant yourself new infrastructure credentials, loosen the
  approval gate, or disable the sandbox. If a task appears to require that, stop
  and report it as blocked — it is an approval boundary, not a bug.
- Never commit a secret. `.env` is gitignored on purpose.

## Verification

State the file changed, the test added, and the command output proving the suite is
green. If you cannot show that, the work is not done.

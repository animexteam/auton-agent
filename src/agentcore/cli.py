"""CLI entrypoint — `python -m agentcore.cli <command>`.

Lets the same agent be driven entirely locally, with no HTTP and no Telegram.
This is also the interface the end-to-end smoke test uses.

Commands:
    run "<objective>"     execute one objective and print the transcript
    chat                  interactive session
    doctor                preflight: report what actually works
    tools                 list registered tools
    skills                list available skills
    self                  full capability/self report
    tasks                 recent durable task records
    events                recent observability events
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from .loop import RunResult
from .runtime import Runtime, build_runtime, preflight
from .security import Principal


def _print_result(result: RunResult, verbose: bool) -> None:
    print("\n" + "=" * 68)
    print(f"STATUS : {result.status}  ({result.stop_reason})")
    print(f"TASK   : {result.task_id}")
    print(f"TIME   : {result.duration_ms / 1000:.1f}s in {len(result.steps)} steps")
    if result.usage:
        print(f"TOKENS : {result.usage}")
    if result.artifacts:
        print(f"ARTIFACTS: {', '.join(result.artifacts)}")
    print("-" * 68)
    print(result.answer or "(no final answer)")
    print("=" * 68)

    if verbose:
        print("\nTRANSCRIPT")
        for step in result.steps:
            print(f"\n[step {step.index}] model={step.model} {step.duration_ms}ms")
            if step.assistant_text:
                print(f"  think: {step.assistant_text[:300]}")
            for call in step.tool_calls:
                flag = "ok" if call["ok"] else f"FAILED({call.get('code')})"
                print(f"  -> {call['name']} [{flag}] {call['duration_ms']}ms")
                print(f"     args: {json.dumps(call['arguments'], default=str)[:200]}")


async def _cmd_run(runtime: Runtime, args: argparse.Namespace) -> int:
    result = await runtime.agent.run(
        args.objective,
        principal=Principal("cli", "local"),
        channel="cli",
        approval_token=args.approval_token,
    )
    _print_result(result, args.verbose)
    return 0 if result.status == "completed" else 2


async def _cmd_chat(runtime: Runtime, args: argparse.Namespace) -> int:
    print("auton-agent interactive session. Type 'exit' to quit.")
    while True:
        try:
            line = input("\nobjective> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if line.lower() in ("exit", "quit", ":q"):
            return 0
        if not line:
            continue
        result = await runtime.agent.run(line, principal=Principal("cli", "local"), channel="cli")
        _print_result(result, args.verbose)


async def _cmd_doctor(runtime: Runtime, args: argparse.Namespace) -> int:
    report = await preflight(runtime)
    print(json.dumps(report, indent=2, default=str))
    health = runtime.router.health()
    print("\nMODEL HEALTH")
    print(json.dumps(health, indent=2, default=str))
    return 0 if report.get("ok") else 1


async def _cmd_tools(runtime: Runtime, args: argparse.Namespace) -> int:
    for item in runtime.agent.registry.catalog():
        flags = " [privileged]" if item["privileged"] else ""
        print(f"{item['category']:12} {item['name']:20}{flags}\n             {item['description']}")
    return 0


async def _cmd_skills(runtime: Runtime, args: argparse.Namespace) -> int:
    for skill in runtime.skills.catalog():
        print(f"{skill['name']:22} ({skill['source']}, {skill['chars']} chars)\n    {skill['description']}")
    return 0


async def _cmd_self(runtime: Runtime, args: argparse.Namespace) -> int:
    print(json.dumps(runtime.agent.describe(), indent=2, default=str))
    return 0


async def _cmd_tasks(runtime: Runtime, args: argparse.Namespace) -> int:
    print(json.dumps(await runtime.memory.tasks.recent(args.limit), indent=2, default=str))
    return 0


async def _cmd_events(runtime: Runtime, args: argparse.Namespace) -> int:
    print(json.dumps(runtime.events.recent(args.limit), indent=2, default=str))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agentcore", description="auton-agent CLI")
    parser.add_argument("--base-dir", default=None, help="working directory for state and workspace")
    parser.add_argument("--verbose", action="store_true", help="print the full step transcript")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run one objective")
    p_run.add_argument("objective", help="what to accomplish")
    p_run.add_argument("--approval-token", default=None)

    sub.add_parser("chat", help="interactive session")

    p_doctor = sub.add_parser("doctor", help="preflight environment check")
    p_tools = sub.add_parser("tools", help="list tools")
    p_skills = sub.add_parser("skills", help="list skills")
    p_self = sub.add_parser("self", help="capability report")
    p_tasks = sub.add_parser("tasks", help="recent task records")
    p_tasks.add_argument("--limit", type=int, default=10)
    p_events = sub.add_parser("events", help="recent events")
    p_events.add_argument("--limit", type=int, default=30)
    return parser


async def _dispatch(runtime: Runtime, args: argparse.Namespace) -> int:
    table = {
        "run": _cmd_run,
        "chat": _cmd_chat,
        "doctor": _cmd_doctor,
        "tools": _cmd_tools,
        "skills": _cmd_skills,
        "self": _cmd_self,
        "tasks": _cmd_tasks,
        "events": _cmd_events,
    }
    handler = table[args.command]
    try:
        return await handler(runtime, args)
    finally:
        await runtime.aclose()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runtime = build_runtime(base_dir=Path(args.base_dir) if args.base_dir else None)
    return asyncio.run(_dispatch(runtime, args))


if __name__ == "__main__":
    sys.exit(main())

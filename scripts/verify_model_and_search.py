#!/usr/bin/env python3
"""End-to-end proof for the model switch and the real-time search path.

Three claims are checked here, each with a real network call — nothing is
asserted from configuration:

1. the newly selected primary model actually answers;
2. it actually emits tool calls (with well-formed arguments);
3. real-time information actually reaches the agent, demonstrated by running the
   *real* agent loop on an objective whose answer cannot be known from training
   data, and printing the search results the agent itself retrieved.

Run:  python scripts/verify_model_and_search.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

# Isolate this run: workspace + state in a temp dir, and no gist mirror, so the
# verification cannot touch the deployed agent's durable state.
_tmp = Path(tempfile.mkdtemp(prefix="verify-"))
os.environ["WORKSPACE_ROOT"] = str(_tmp / "workspace")
os.environ["STATE_ROOT"] = str(_tmp / "state")
os.environ["PERSISTENCE_BACKEND"] = "disk"
os.environ.pop("GIST_API_KEY", None)
os.environ.pop("TELEGRAM_BOT_TOKEN", None)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agentcore.config import load_settings  # noqa: E402
from agentcore.llm import ModelRouter, build_provider  # noqa: E402

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }
]


def banner(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}", flush=True)


async def main() -> int:
    settings = load_settings()

    banner("CONFIGURED CHAIN (read from the live config object)")
    print(json.dumps(
        {
            "primary": settings.model.primary,
            "chain": list(settings.model.chain),
            "api_key_present": bool(settings.model.api_key),
            "base_url": settings.model.base_url,
        },
        indent=2,
    ))

    provider = build_provider(settings.model)
    router = ModelRouter(provider, settings.model.chain)

    # ---- 1. does the primary model actually answer? ---------------------
    banner("1. REAL CHAT CALL — does the model answer?")
    reply = await router.chat(
        [{"role": "user", "content": "Reply with exactly: ALIVE"}],
    )
    print(f"model_that_answered : {reply.model}")
    print(f"content             : {reply.content!r}")
    print(f"usage               : {reply.usage}")
    claim1 = bool(reply.content)

    # ---- 2. tool calling -------------------------------------------------
    banner("2. REAL TOOL-CALLING CALL — does it emit a valid tool call?")
    tool_reply = await router.chat(
        [{"role": "user", "content": "What is the weather in Jakarta right now? Use the web_search tool."}],
        tools=TOOLS,
    )
    calls = [
        {"name": c.name, "arguments": c.arguments, "id": c.id} for c in tool_reply.tool_calls
    ]
    print(f"model_that_answered : {tool_reply.model}")
    print(f"tool_calls          : {json.dumps(calls, indent=2)}")
    claim2 = bool(calls) and all(c["name"] for c in calls)
    print(f"tool_calling_works  : {claim2}")

    # ---- 3. real-time search --------------------------------------------
    banner("3. NATIVE REAL-TIME SEARCH — provider endpoint through the router")
    probe_query = "what is today's date and any major world news today"
    found = await router.search(probe_query, max_results=3)
    print(f"query       : {probe_query}")
    print(f"results     : {len(found)}")
    for item in found:
        print(f"  - {item.title[:90]}")
        print(f"    {item.url[:100]}")
        print(f"    content[:300]: {(item.content or '')[:300]!r}")
    claim3 = len(found) > 0 and any((i.content or "").strip() for i in found)
    print(f"returns_live_content : {claim3}")

    # ---- 4. the whole agent loop, on an objective needing current data ----
    banner("4. FULL AGENT LOOP — objective that depends on CURRENT information")
    from agentcore.runtime import build_runtime  # imported late: needs the env set above

    runtime = build_runtime(settings=settings)
    objective = (
        "Find the exact current date today and one real news headline from today. "
        "You must search the web — do not use any date you might remember. "
        "Then write both into current_info.txt in the workspace."
    )
    result = await runtime.agent.run(objective, channel="cli")
    print(f"status      : {result.status}")
    print(f"stop_reason : {result.stop_reason}")
    print(f"steps       : {len(result.steps)}")
    for step in result.steps:
        used = ", ".join(f"{c['name']}(ok={c['ok']})" for c in step.tool_calls) or "(no tools)"
        print(f"  step {step.index}: model={step.model} tools={used}")
    print(f"answer:\n{result.answer[:1200]}")
    print(f"artifacts   : {result.artifacts}")

    # The search results the agent itself saw — this is the real-time evidence.
    seen = Path(settings.workspace_root)
    print(f"\nworkspace files: {[p.name for p in seen.glob('*')]}")
    for name in ("current_info.txt",):
        path = seen / name
        if path.exists():
            print(f"--- {name} ---\n{path.read_text(encoding='utf-8')[:800]}")

    used_search = any(
        c["name"] == "web_search" and c["ok"] for s in result.steps for c in s.tool_calls
    )
    claim4 = used_search
    print(f"\nagent_really_used_web_search : {claim4}")

    await runtime.aclose()

    banner("VERDICT")
    verdict = {
        "1_model_answers": claim1,
        "2_tool_calling": claim2,
        "3_native_search_and_content": claim3,
        "4_agent_used_real_time_search": claim4,
    }
    print(json.dumps(verdict, indent=2))
    return 0 if all(verdict.values()) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

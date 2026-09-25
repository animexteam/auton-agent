"""The agentic loop: multi-step execution, observation, adaptation, budgets."""

from __future__ import annotations

import pytest

from agentcore.config import AgentConfig, Settings
from agentcore.errors import ApprovalRequired, ModelUnavailableError, PermissionDenied
from agentcore.llm.mock import MockProvider
from agentcore.llm.router import ModelRouter
from agentcore.loop import Agent
from agentcore.memory import (
    Conversation,
    LongTermMemory,
    MemoryBundle,
    TaskStore,
    WorkingState,
)
from agentcore.persistence import DiskStore
from agentcore.registry import ToolRegistry
from agentcore.sandbox import Sandbox
from agentcore.security import ApprovalGate, Authorizer, RateLimiter, Principal
from agentcore.skills import SkillLoader
from agentcore.tools import build_tools


def build_agent(settings: Settings, provider: MockProvider, registry: ToolRegistry | None = None) -> Agent:
    settings.workspace_root.mkdir(parents=True, exist_ok=True)
    settings.state_root.mkdir(parents=True, exist_ok=True)
    store = DiskStore(settings.state_root)
    memory = MemoryBundle(
        store=store,
        tasks=TaskStore(store),
        long_term=LongTermMemory(store),
        conversation=Conversation(),
        working=WorkingState(),
    )
    registry = registry or ToolRegistry()
    if registry is None or not registry.names():
        registry = ToolRegistry()
        registry.register_all(build_tools())

    return Agent(
        router=ModelRouter(provider, [settings.model.primary, *settings.model.fallbacks]),
        settings=settings,
        registry=registry,
        skills=SkillLoader(),
        authorizer=Authorizer(settings.telegram.allowed_users),
        rate_limiter=RateLimiter(100),
        gate=ApprovalGate(allow_destructive=False),
        memory=memory,
        sandbox=Sandbox(settings.sandbox, settings.workspace_root),
    )


# --------------------------------------------------------------------------
# core loop behaviour
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_single_turn_completion(tmp_settings):
    provider = MockProvider([{"content": "The answer is 42."}])
    agent = build_agent(tmp_settings, provider)
    result = await agent.run("What is the answer?", principal=Principal("cli", "t"))

    assert result.status == "completed"
    assert "42" in result.answer
    assert len(result.steps) == 1
    assert result.steps[0].tool_calls == []


@pytest.mark.asyncio
async def test_multi_step_tool_use_and_observation(tmp_settings):
    """The loop must act, read the result, and act again — not one-shot."""
    provider = MockProvider(
        [
            # step 1: write a file
            {
                "content": "",
                "tool_calls": [
                    {"id": "c1", "name": "write_file", "arguments": {"path": "note.txt", "content": "hello agent"}}
                ],
            },
            # step 2: read it back (proves it saw the previous result)
            {
                "content": "",
                "tool_calls": [{"id": "c2", "name": "read_file", "arguments": {"path": "note.txt"}}],
            },
            # step 3: finish
            {"content": "Wrote and verified note.txt containing 'hello agent'."},
        ]
    )
    agent = build_agent(tmp_settings, provider)
    result = await agent.run("Create note.txt and verify it.", principal=Principal("cli", "t"))

    assert result.status == "completed"
    assert len(result.steps) == 3
    assert result.steps[0].tool_calls[0]["name"] == "write_file"
    assert result.steps[1].tool_calls[0]["name"] == "read_file"
    # The transcript that reached the model must contain the real tool output.
    tool_messages = [m for call in provider.calls for m in call["messages"] if m.get("role") == "tool"]
    assert any("hello agent" in str(m.get("content")) for m in tool_messages)


@pytest.mark.asyncio
async def test_adapts_after_tool_failure(tmp_settings):
    """A failing tool must produce a structured observation the model reacts to."""
    provider = MockProvider(
        [
            # step 1: read a file that does not exist -> tool error
            {"content": "", "tool_calls": [{"id": "c1", "name": "read_file", "arguments": {"path": "missing.txt"}}]},
            # step 2: adapt by writing it instead
            {
                "content": "Not found, so creating it.",
                "tool_calls": [
                    {"id": "c2", "name": "write_file", "arguments": {"path": "missing.txt", "content": "created"}}
                ],
            },
            {"content": "Created missing.txt after the read failed."},
        ]
    )
    agent = build_agent(tmp_settings, provider)
    result = await agent.run("Read missing.txt", principal=Principal("cli", "t"))

    assert result.status == "completed"
    assert result.steps[0].tool_calls[0]["ok"] is False
    assert result.steps[1].tool_calls[0]["ok"] is True
    # The structured error reached the model as a tool message.
    contents = [
        str(m.get("content"))
        for call in provider.calls
        for m in call["messages"]
        if m.get("role") == "tool"
    ]
    assert any('"error": true' in c for c in contents)


@pytest.mark.asyncio
async def test_identical_failures_trigger_strategy_nudge(tmp_settings):
    """Three identical failures must inject an explicit correction, not loop blindly."""
    turn = {"content": "", "tool_calls": [{"id": "c", "name": "read_file", "arguments": {"path": "nope.txt"}}]}
    provider = MockProvider([turn, turn, turn, turn, {"content": "giving up on that path"}])
    agent = build_agent(tmp_settings, provider)
    result = await agent.run("Read nope.txt", principal=Principal("cli", "t"))

    assert result.status == "completed"
    later_prompts = "\n".join(
        str(m.get("content")) for call in provider.calls for m in call["messages"]
    )
    assert "has now failed" in later_prompts


@pytest.mark.asyncio
async def test_step_budget_stops_the_loop(tmp_settings):
    """The loop must terminate on the step budget with a useful report."""
    settings = Settings(**{**tmp_settings.__dict__, "agent": AgentConfig(max_steps=3, max_seconds=60)})
    forever = {"content": "", "tool_calls": [{"id": "c", "name": "list_dir", "arguments": {"path": "."}}]}
    provider = MockProvider([forever, forever, forever, forever])
    agent = build_agent(settings, provider)
    result = await agent.run("Loop forever", principal=Principal("cli", "t"))

    assert result.status == "budget_exceeded"
    assert result.stop_reason == "step_budget"
    assert len(result.steps) == 3
    assert "ran out of my step budget" in result.answer


@pytest.mark.asyncio
async def test_multi_tool_call_in_one_turn(tmp_settings):
    """Independent calls in one assistant turn are all executed."""
    provider = MockProvider(
        [
            {
                "content": "",
                "tool_calls": [
                    {"id": "a", "name": "write_file", "arguments": {"path": "a.txt", "content": "A"}},
                    {"id": "b", "name": "write_file", "arguments": {"path": "b.txt", "content": "B"}},
                ],
            },
            {"content": "wrote both"},
        ]
    )
    agent = build_agent(tmp_settings, provider)
    result = await agent.run("Write two files", principal=Principal("cli", "t"))

    assert result.status == "completed"
    assert len(result.steps[0].tool_calls) == 2
    assert (tmp_settings.workspace_root / "a.txt").exists()
    assert (tmp_settings.workspace_root / "b.txt").exists()


@pytest.mark.asyncio
async def test_empty_assistant_turn_is_nudged(tmp_settings):
    provider = MockProvider([{"content": "   "}, {"content": "real answer"}])
    agent = build_agent(tmp_settings, provider)
    result = await agent.run("Answer me", principal=Principal("cli", "t"))
    assert result.status == "completed"
    assert result.answer == "real answer"
    assert len(result.steps) == 2


@pytest.mark.asyncio
async def test_artifact_is_recorded_on_the_task(tmp_settings):
    provider = MockProvider(
        [
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "name": "write_artifact",
                        "arguments": {"name": "report.md", "content": "# Report\n\nDone."},
                    }
                ],
            },
            {"content": "Produced report.md"},
        ]
    )
    agent = build_agent(tmp_settings, provider)
    result = await agent.run("Write a report", principal=Principal("cli", "t"))

    assert result.artifacts == ["artifacts/report.md"]
    assert (tmp_settings.workspace_root / "artifacts" / "report.md").read_text().startswith("# Report")
    record = await agent.memory.tasks.get(result.task_id)
    assert "artifacts/report.md" in record.artifacts


# --------------------------------------------------------------------------
# durability and recovery
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_task_survives_a_new_agent_instance(tmp_settings):
    """The durable record must be readable by a fresh process (restart safety)."""
    provider = MockProvider([{"content": "done"}])
    agent = build_agent(tmp_settings, provider)
    result = await agent.run("Persist this", principal=Principal("cli", "t"))

    # Simulate a restart: rebuild everything from the same state root.
    fresh = build_agent(tmp_settings, MockProvider([]))
    record = await fresh.memory.tasks.get(result.task_id)
    assert record is not None
    assert record.status == "completed"
    assert record.objective == "Persist this"


@pytest.mark.asyncio
async def test_unfinished_task_is_visible_for_resume(tmp_settings):
    settings = Settings(**{**tmp_settings.__dict__, "agent": AgentConfig(max_steps=1, max_seconds=60)})
    forever = {"content": "", "tool_calls": [{"id": "c", "name": "list_dir", "arguments": {"path": "."}}]}
    agent = build_agent(settings, MockProvider([forever, forever]))
    result = await agent.run("Interrupted work", principal=Principal("cli", "t"))
    assert result.status == "budget_exceeded"

    fresh = build_agent(settings, MockProvider([]))
    unfinished = await fresh.memory.tasks.unfinished()
    assert any(t["id"] == result.task_id for t in unfinished)


# --------------------------------------------------------------------------
# security gates
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_telegram_user_not_on_allowlist_is_denied(tmp_settings):
    agent = build_agent(tmp_settings, MockProvider([{"content": "should not run"}]))
    with pytest.raises(PermissionDenied):
        await agent.run("do something", principal=Principal("telegram", "999"), channel="telegram")


@pytest.mark.asyncio
async def test_empty_allowlist_denies_everyone(tmp_settings):
    settings = Settings(**{**tmp_settings.__dict__, "telegram": tmp_settings.telegram.__class__(allowed_users=())})
    agent = build_agent(settings, MockProvider([{"content": "no"}]))
    with pytest.raises(PermissionDenied):
        await agent.run("hi", principal=Principal("telegram", "42"), channel="telegram")


@pytest.mark.asyncio
async def test_destructive_command_needs_approval(tmp_settings):
    """A destructive command must stop the run and ask, rather than proceeding."""
    provider = MockProvider(
        [{"content": "", "tool_calls": [{"id": "c", "name": "run_command", "arguments": {"command": "rm -rf /"}}]}]
    )
    agent = build_agent(tmp_settings, provider)
    # No approval granted anywhere by default.
    agent.gate = ApprovalGate(allow_destructive=False)
    result = await agent.run("wipe the disk", principal=Principal("cli", "t"))
    assert result.status == "blocked"
    assert result.needs_approval


@pytest.mark.asyncio
async def test_ordinary_command_completes_without_an_approval_token(tmp_settings):
    """The agent must be able to actually use its shell on an ordinary command."""
    provider = MockProvider(
        [
            {"content": "", "tool_calls": [{"id": "c", "name": "run_command", "arguments": {"command": "echo hi"}}]},
            {"content": "ran it"},
        ]
    )
    agent = build_agent(tmp_settings, provider)
    result = await agent.run("run a command", principal=Principal("cli", "t"))
    assert result.status == "completed"
    assert result.steps[0].tool_calls[0]["ok"] is True


@pytest.mark.asyncio
async def test_approval_token_permits_the_destructive_action(tmp_settings):
    provider = MockProvider(
        [
            {"content": "", "tool_calls": [{"id": "c", "name": "run_command", "arguments": {"command": "rm -rf /"}}]},
            {"content": "cleared it"},
        ]
    )
    agent = build_agent(tmp_settings, provider)
    agent.gate.grant("tok-1")
    result = await agent.run(
        "clear the disk", principal=Principal("cli", "t"), approval_token="tok-1"
    )
    # The token unlocks the gate; the command itself then runs (and fails harmlessly).
    # `needs_approval` holds the reason string when set, so "not blocked" is None.
    assert result.needs_approval is None
    assert result.steps[0].tool_calls[0]["ok"] is True


@pytest.mark.asyncio
async def test_model_failure_produces_an_honest_failure(tmp_settings):
    # Both models in the chain must be exhausted before the router gives up.
    provider = MockProvider(
        [{"error": ModelUnavailableError("all models down")}] * 4
    )
    agent = build_agent(tmp_settings, provider)
    result = await agent.run("anything", principal=Principal("cli", "t"))
    assert result.status == "failed"
    assert "model" in (result.error or "").lower()
    # It must not claim success.
    assert result.artifacts == []


@pytest.mark.asyncio
async def test_router_falls_back_to_the_second_model(tmp_settings):
    """A permanently-rejected primary must not fail the task while a backup exists."""
    provider = MockProvider(
        [{"error": ModelUnavailableError("primary not entitled")}, {"content": "backup answered"}]
    )
    agent = build_agent(tmp_settings, provider)
    result = await agent.run("anything", principal=Principal("cli", "t"))
    assert result.status == "completed"
    assert "backup answered" in result.answer
    # The failing model is quarantined; the healthy one becomes preferred.
    health = agent.router.health()
    assert health["fallbacks"] >= 1
    assert health["preferred"] == tmp_settings.model.fallbacks[0]


@pytest.mark.asyncio
async def test_working_memory_scratchpad_persists_across_steps(tmp_settings):
    provider = MockProvider(
        [
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "name": "working_memory",
                        "arguments": {"action": "set_plan", "plan": ["gather", "write", "verify"]},
                    }
                ],
            },
            {
                "content": "",
                "tool_calls": [
                    {"id": "c2", "name": "working_memory", "arguments": {"action": "add_fact", "value": "found 3 sources"}}
                ],
            },
            {"content": "done"},
        ]
    )
    agent = build_agent(tmp_settings, provider)
    result = await agent.run("Plan something", principal=Principal("cli", "t"))
    assert result.status == "completed"

    # The scratchpad content must appear in the prompts sent on later steps.
    final_prompt = str(provider.calls[-1]["messages"][-1].get("content"))
    assert "gather" in final_prompt
    assert "found 3 sources" in final_prompt


@pytest.mark.asyncio
async def test_recalled_memory_is_injected_into_the_system_prompt(tmp_settings):
    provider = MockProvider([{"content": "ok"}])
    agent = build_agent(tmp_settings, provider)
    await agent.memory.long_term.remember("The user prefers concise reports.", kind="preference")

    await agent.run("Write me a report", principal=Principal("cli", "t"))
    system_prompt = str(provider.calls[0]["messages"][0].get("content"))
    assert "concise reports" in system_prompt

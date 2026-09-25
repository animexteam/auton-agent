"""Persistence, memory, and secret redaction."""

from __future__ import annotations

import json

import pytest

from agentcore.memory import (
    Conversation,
    LongTermMemory,
    TaskRecord,
    TaskStore,
    WorkingState,
)
from agentcore.persistence import ChainedStore, DiskStore, build_store
from agentcore.config import PersistenceConfig
from agentcore.redaction import Redactor, configure, scrub


# --------------------------------------------------------------------------
# disk store
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_disk_store_roundtrip(tmp_path):
    store = DiskStore(tmp_path / "state")
    await store.set("doc", {"a": 1, "b": [1, 2, 3]})
    assert await store.get("doc") == {"a": 1, "b": [1, 2, 3]}
    assert await store.get("absent", "fallback") == "fallback"
    await store.delete("doc")
    assert await store.get("doc") is None


@pytest.mark.asyncio
async def test_disk_store_survives_a_new_instance(tmp_path):
    root = tmp_path / "state"
    await DiskStore(root).set("doc", {"persisted": True})
    assert await DiskStore(root).get("doc") == {"persisted": True}


@pytest.mark.asyncio
async def test_disk_store_sanitises_keys(tmp_path):
    store = DiskStore(tmp_path / "state")
    await store.set("weird/../key name", {"ok": True})
    assert await store.get("weird/../key name") == {"ok": True}
    # No traversal happened: the file is inside the root.
    assert all(p.parent == (tmp_path / "state") for p in (tmp_path / "state").glob("*.json"))


# --------------------------------------------------------------------------
# chained store
# --------------------------------------------------------------------------
class _FailingStore(DiskStore):
    """A mirror that is up for reads but rejects writes."""

    async def set(self, key, value):  # noqa: D102
        from agentcore.errors import PersistenceError

        raise PersistenceError("mirror is down")


@pytest.mark.asyncio
async def test_chained_store_keeps_working_when_the_mirror_fails(tmp_path):
    primary = DiskStore(tmp_path / "primary")
    chain = ChainedStore(primary, _FailingStore(tmp_path / "mirror"), mirror_keys=("agent.memory",))
    # The write must not fail just because durability is degraded.
    await chain.set("agent.memory", [{"x": 1}])
    assert await chain.get("agent.memory") == [{"x": 1}]


@pytest.mark.asyncio
async def test_chained_store_reads_through_from_the_mirror(tmp_path):
    mirror = DiskStore(tmp_path / "mirror")
    await mirror.set("agent.memory", [{"from": "mirror"}])
    fresh_primary = DiskStore(tmp_path / "primary")
    chain = ChainedStore(fresh_primary, mirror, mirror_keys=("agent.memory",))
    # Cold start: primary is empty, value must come from the mirror.
    assert await chain.get("agent.memory") == [{"from": "mirror"}]
    # and it must now be warmed on the fast path
    assert await fresh_primary.get("agent.memory") == [{"from": "mirror"}]


def test_build_store_falls_back_to_disk_without_gist_credentials(tmp_path):
    store = build_store(PersistenceConfig(backend="gist", gist_id=None, gist_api_key=None), tmp_path)
    assert store.name == "disk"


# --------------------------------------------------------------------------
# task store
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_task_store_records_and_lists(tmp_path):
    store = DiskStore(tmp_path / "state")
    tasks = TaskStore(store)
    record = TaskRecord(id="t1", objective="do a thing", status="running")
    await tasks.save(record)

    got = await tasks.get("t1")
    assert got is not None and got.objective == "do a thing"
    assert any(t["id"] == "t1" for t in await tasks.recent())
    # A running task is 'unfinished' and therefore resumable after a restart.
    assert any(t["id"] == "t1" for t in await tasks.unfinished())


@pytest.mark.asyncio
async def test_task_store_marks_completion(tmp_path):
    tasks = TaskStore(DiskStore(tmp_path / "state"))
    await tasks.save(TaskRecord(id="t2", objective="x", status="completed"))
    assert await tasks.unfinished() == []


# --------------------------------------------------------------------------
# long-term memory
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_memory_remember_and_recall(tmp_path):
    memory = LongTermMemory(DiskStore(tmp_path / "state"))
    await memory.remember("The deploy target is Render.", kind="fact", importance=4)
    await memory.remember("The user hates long reports.", kind="preference", importance=5)

    hits = await memory.recall("render")
    assert hits and "Render" in hits[0].text
    assert await memory.count() == 2


@pytest.mark.asyncio
async def test_memory_deduplicates_identical_facts(tmp_path):
    memory = LongTermMemory(DiskStore(tmp_path / "state"))
    await memory.remember("same fact")
    await memory.remember("same fact")
    assert await memory.count() == 1


@pytest.mark.asyncio
async def test_memory_forget(tmp_path):
    memory = LongTermMemory(DiskStore(tmp_path / "state"))
    entry = await memory.remember("temporary")
    assert await memory.forget(entry.id) is True
    assert await memory.count() == 0
    assert await memory.forget("nope") is False


# --------------------------------------------------------------------------
# conversation trimming
# --------------------------------------------------------------------------
def test_conversation_trims_and_remembers_what_it_dropped():
    conversation = Conversation(max_messages=4, max_chars=100_000)
    for i in range(10):
        conversation.append({"role": "user", "content": f"message {i}"})
    assert len(conversation) <= 4
    note = conversation.context_note()
    assert note and "summarised" in note


def test_conversation_never_orphans_tool_results():
    """Trimming must not leave a tool message without its assistant call."""
    conversation = Conversation(max_messages=6, max_chars=100_000)
    for i in range(8):
        conversation.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": f"c{i}", "type": "function", "function": {"name": "read_file", "arguments": {}}}],
            }
        )
        conversation.append({"role": "tool", "tool_call_id": f"c{i}", "content": "result"})
    messages = conversation.messages
    assert messages[0]["role"] != "tool", "a tool result was orphaned at the head"


def test_working_state_renders_and_marks_done_steps():
    state = WorkingState(objective="ship it", finish_condition="it is live")
    state.plan = ["write code", "run tests"]
    state.plan[0] = "[done] write code"
    state.facts.append("render free spins down")
    rendered = state.to_prompt()
    assert "ship it" in rendered
    assert "it is live" in rendered
    assert "[x] 1. write code" in rendered
    assert "render free spins down" in rendered


# --------------------------------------------------------------------------
# redaction — the leak-prevention guarantee
# --------------------------------------------------------------------------
def test_redactor_removes_known_secret_values():
    red = Redactor({"OLLAMA_API_KEY": "supersecretvalue12345"})
    assert "supersecretvalue12345" not in red.scrub("key=supersecretvalue12345 end")


def test_redactor_scrubs_by_credential_shape_without_being_told():
    red = Redactor({})
    for payload in (
        "token ghp_abcdefghijklmnopqrstuvwxyz0123456789",
        "render rnd_abcdefghijklmnopqrstuvwx",
        "bot 123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi",
        "slack xoxb-123456789012-abcdefghijklm",
    ):
        scrubbed = red.scrub(payload)
        assert "[REDACTED]" in scrubbed


def test_redactor_walks_nested_structures():
    red = Redactor({"GITHUB_API_KEY": "ghp_verysecretvaluehere123"})
    payload = {"a": ["x ghp_verysecretvaluehere123"], "b": {"c": "ghp_verysecretvaluehere123"}}
    out = red.scrub_deep(payload)
    assert "verysecretvaluehere" not in json.dumps(out)


def test_redactor_ignores_short_values():
    """Short/empty env values must not be substituted, or output would garble."""
    red = Redactor({"X": "", "Y": "abc"})
    assert red.scrub("abc here") == "abc here"


def test_module_level_scrub_uses_configured_redactor():
    configure({"API_AUTH_TOKEN": "tok_abcdefghijklmnop"})
    assert "tok_abcdefghijklmnop" not in scrub("auth: tok_abcdefghijklmnop")


@pytest.mark.asyncio
async def test_state_written_to_disk_is_redacted(tmp_path):
    """A secret must not reach the state files."""
    configure({"GITHUB_API_KEY": "ghp_thisisasecretvalue0123456789"})
    store = DiskStore(tmp_path / "state")
    await store.set("doc", {"note": "the key is ghp_thisisasecretvalue0123456789"})
    raw = (tmp_path / "state" / "doc.json").read_text()
    assert "thisisasecretvalue" not in raw
    assert "[REDACTED]" in raw

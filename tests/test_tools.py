"""Tool contracts: validation, confinement, limits, redaction."""

from __future__ import annotations

import pytest

from agentcore.config import SandboxConfig
from agentcore.errors import ApprovalRequired, PermissionDenied, ToolValidationError
from agentcore.memory import Conversation, LongTermMemory, MemoryBundle, TaskStore, WorkingState
from agentcore.persistence import DiskStore
from agentcore.registry import (
    ToolContext,
    ToolRegistry,
    validate_arguments,
)
from agentcore.sandbox import Sandbox, screen_command
from agentcore.security import ApprovalGate, PathGuard
from agentcore.skills import SkillLoader
from agentcore.tools import build_tools
from agentcore.tools.fs import ReadFileTool, WriteFileTool
from agentcore.tools.meta import SelfReportTool


@pytest.fixture
def ctx(tmp_settings):
    tmp_settings.workspace_root.mkdir(parents=True, exist_ok=True)
    tmp_settings.state_root.mkdir(parents=True, exist_ok=True)
    store = DiskStore(tmp_settings.state_root)
    memory = MemoryBundle(
        store=store,
        tasks=TaskStore(store),
        long_term=LongTermMemory(store),
        conversation=Conversation(),
        working=WorkingState(),
    )
    sandbox = Sandbox(tmp_settings.sandbox, tmp_settings.workspace_root)
    return ToolContext(
        workspace_root=tmp_settings.workspace_root,
        path_guard=PathGuard(tmp_settings.workspace_root),
        sandbox=sandbox,
        state=store,
        memory=memory,
        skills=SkillLoader(),
        gate=ApprovalGate(allow_destructive=False),
        settings=tmp_settings,
        task_id="task_test",
        principal="cli:test",
        extra={"registry": None, "router": None},
    )


# --------------------------------------------------------------------------
# schema validation
# --------------------------------------------------------------------------
def test_schema_validation_rejects_missing_required():
    schema = {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
    with pytest.raises(ToolValidationError) as exc:
        validate_arguments(schema, {})
    assert "missing required" in str(exc.value)


def test_schema_validation_rejects_unknown_arguments():
    schema = {"type": "object", "properties": {"path": {"type": "string"}}, "required": [], "additionalProperties": False}
    with pytest.raises(ToolValidationError) as exc:
        validate_arguments(schema, {"path": "a", "bogus": 1})
    assert "unknown argument" in str(exc.value)


def test_schema_validation_coerces_numeric_strings():
    """Models often pass numbers as strings; that must not fail the call."""
    schema = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}
    assert validate_arguments(schema, {"n": "42"})["n"] == 42


def test_schema_validation_applies_defaults_and_bounds():
    schema = {
        "type": "object",
        "properties": {"n": {"type": "integer", "default": 5, "maximum": 10}},
        "required": [],
    }
    assert validate_arguments(schema, {})["n"] == 5
    with pytest.raises(ToolValidationError):
        validate_arguments(schema, {"n": 99})


def test_schema_validation_enforces_enum():
    schema = {"type": "object", "properties": {"a": {"type": "string", "enum": ["x", "y"]}}, "required": ["a"]}
    with pytest.raises(ToolValidationError):
        validate_arguments(schema, {"a": "z"})


# --------------------------------------------------------------------------
# path confinement
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_read_outside_workspace_is_refused(ctx):
    tool = ReadFileTool()
    with pytest.raises(PermissionDenied):
        await tool.run({"path": "../../etc/passwd"}, ctx)


def test_path_guard_blocks_traversal(tmp_settings):
    guard = PathGuard(tmp_settings.workspace_root)
    for bad in ("../secrets.txt", "/etc/passwd", "a/../../b"):
        with pytest.raises(PermissionDenied):
            guard.resolve(bad)


@pytest.mark.asyncio
async def test_write_then_read_verifies_content(ctx):
    writer, reader = WriteFileTool(), ReadFileTool()
    out = await writer.run({"path": "sub/dir/file.txt", "content": "line1\nline2"}, ctx)
    assert out["verified"] is True
    got = await reader.run({"path": "sub/dir/file.txt"}, ctx)
    assert "line1" in got["content"] and got["total_lines"] == 2


# --------------------------------------------------------------------------
# registry behaviour
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_registry_returns_structured_error_for_unknown_tool(ctx):
    registry = ToolRegistry()
    registry.register_all(build_tools())
    result = await registry.execute("no_such_tool", {}, ctx)
    assert result.ok is False
    assert result.code == "tool_not_found"
    assert "unknown tool" in (result.error or "")


@pytest.mark.asyncio
async def test_registry_returns_structured_error_for_bad_arguments(ctx):
    registry = ToolRegistry()
    registry.register_all(build_tools())
    result = await registry.execute("read_file", {"wrong": 1}, ctx)
    assert result.ok is False
    assert result.code == "tool_validation_error"


@pytest.mark.asyncio
async def test_registry_specs_are_valid_function_schemas(ctx):
    registry = ToolRegistry()
    registry.register_all(build_tools())
    specs = registry.specs()
    assert len(specs) >= 12
    for spec in specs:
        assert spec["type"] == "function"
        assert spec["function"]["name"]
        assert spec["function"]["description"]
        assert spec["function"]["parameters"]["type"] == "object"


def test_registry_rejects_duplicate_tool_names():
    registry = ToolRegistry()
    registry.register(WriteFileTool())
    with pytest.raises(ValueError):
        registry.register(WriteFileTool())


@pytest.mark.asyncio
async def test_ordinary_command_runs_without_a_per_call_approval(ctx):
    """`echo hi` is not destructive, so it must just run.

    Regression guard: the first cut marked RunCommandTool `privileged`, which made
    the agent stop and ask a human before every `ls`. Authorisation comes from the
    principal allowlist, not from a per-command token.
    """
    from agentcore.tools.exec import RunCommandTool

    registry = ToolRegistry()
    registry.register(RunCommandTool())
    result = await registry.execute("run_command", {"command": "echo hi"}, ctx)
    assert result.ok is True
    assert "hi" in result.content["stdout"]


@pytest.mark.asyncio
async def test_destructive_command_escalates_to_approval_through_registry(ctx):
    """A destructive command must surface as `approval_required`, not run."""
    from agentcore.tools.exec import RunCommandTool

    registry = ToolRegistry()
    registry.register(RunCommandTool())
    result = await registry.execute("run_command", {"command": "rm -rf /"}, ctx)
    assert result.ok is False
    assert result.code == "approval_required"


# --------------------------------------------------------------------------
# sandbox
# --------------------------------------------------------------------------
def test_destructive_commands_are_screened():
    """Two tiers: destructive asks for approval, catastrophic is a hard refusal."""
    # Tier 1 — destructive: escalate to a human rather than run silently.
    with pytest.raises(ApprovalRequired):
        screen_command("rm -rf /", allow_destructive=False)
    with pytest.raises(ApprovalRequired):
        screen_command("shutdown -h now", allow_destructive=False)
    # Explicit opt-in grants a standing approval for the destructive tier.
    screen_command("rm -rf /tmp/x", allow_destructive=True)
    # Tier 2 — catastrophic: no approval can unlock it.
    with pytest.raises(PermissionDenied):
        screen_command(":(){:|:&};:", allow_destructive=True)
    # ...and an ordinary command is never screened out.
    screen_command("ls -la && python3 -m pytest -q", allow_destructive=False)


@pytest.mark.asyncio
async def test_sandbox_executes_and_captures_output(tmp_settings):
    tmp_settings.workspace_root.mkdir(parents=True, exist_ok=True)
    sandbox = Sandbox(tmp_settings.sandbox, tmp_settings.workspace_root)
    result = await sandbox.run("echo hello && echo err >&2")
    assert result.exit_code == 0
    assert "hello" in result.stdout
    assert "err" in result.stderr


@pytest.mark.asyncio
async def test_sandbox_enforces_timeout(tmp_settings):
    settings = SandboxConfig(enabled=True, timeout_seconds=2, max_output_bytes=10_000)
    tmp_settings.workspace_root.mkdir(parents=True, exist_ok=True)
    sandbox = Sandbox(settings, tmp_settings.workspace_root)
    result = await sandbox.run("sleep 30", timeout_seconds=2)
    assert result.timed_out is True
    assert result.ok is False


@pytest.mark.asyncio
async def test_sandbox_is_disabled_when_configured(tmp_settings):
    from agentcore.errors import ToolError

    tmp_settings.workspace_root.mkdir(parents=True, exist_ok=True)
    sandbox = Sandbox(SandboxConfig(enabled=False), tmp_settings.workspace_root)
    with pytest.raises(ToolError):
        await sandbox.run("echo hi")


# --------------------------------------------------------------------------
# self-report honesty
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_self_report_reports_real_limits(ctx):
    tool = SelfReportTool()
    report = await tool.run({"section": "all"}, ctx)
    assert report["limits"]["command_timeout_seconds"] == ctx.settings.sandbox.timeout_seconds
    assert report["permissions"]["allow_destructive"] is False
    assert report["environment"]["workspace_writable"] is True
    assert "cannot_do" in report


@pytest.mark.asyncio
async def test_skill_loading(ctx):
    from agentcore.tools.meta import LoadSkillTool

    tool = LoadSkillTool()
    listing = await tool.run({"action": "list"}, ctx)
    assert listing["skills"]
    names = [s["name"] for s in listing["skills"]]
    assert "verification" in names

    loaded = await tool.run({"action": "load", "name": "verification"}, ctx)
    assert loaded["found"] is True
    assert "finish condition" in loaded["content"]

    missing = await tool.run({"action": "load", "name": "does-not-exist"}, ctx)
    assert missing["found"] is False

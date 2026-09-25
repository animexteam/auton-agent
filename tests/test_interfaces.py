"""Interfaces: Telegram update parsing, result formatting, HTTP surface."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from agentcore.interfaces.telegram import (
    TelegramInterface,
    parse_update,
    strip_bot_mention,
)
from agentcore.loop import RunResult, StepRecord
from agentcore.memory import (
    Conversation,
    LongTermMemory,
    MemoryBundle,
    TaskStore,
    WorkingState,
)
from agentcore.persistence import DiskStore
from agentcore.llm.mock import MockProvider
from agentcore.llm.router import ModelRouter
from agentcore.registry import ToolRegistry
from agentcore.sandbox import Sandbox
from agentcore.security import ApprovalGate, Authorizer, PathGuard, RateLimiter
from agentcore.service import create_app
from agentcore.skills import SkillLoader
from agentcore.loop import Agent
from agentcore.tools import build_tools


# --------------------------------------------------------------------------
# update parsing
# --------------------------------------------------------------------------
def test_parse_update_extracts_the_text_message():
    update = {
        "update_id": 7,
        "message": {
            "text": "do the thing",
            "chat": {"id": -100123, "type": "group"},
            "from": {"id": 42, "username": "tester"},
        },
    }
    message = parse_update(update)
    assert message is not None
    assert message.text == "do the thing"
    assert message.user_id == "42"
    assert message.is_group is True


def test_parse_update_ignores_non_text_updates():
    assert parse_update({"update_id": 1, "message": {"photo": []}}) is None
    assert parse_update({"update_id": 2}) is None


def test_strip_bot_mention_removes_the_prefix():
    assert strip_bot_mention("@mybot hello there", "mybot") == "hello there"
    assert strip_bot_mention("hello there", "mybot") == "hello there"


# --------------------------------------------------------------------------
# result formatting
# --------------------------------------------------------------------------
def test_format_result_includes_status_answer_and_artifacts():
    result = RunResult(
        task_id="t1",
        objective="x",
        status="completed",
        answer="It worked.",
        artifacts=["artifacts/report.md"],
        duration_ms=1500,
    )
    result.steps.append(StepRecord(index=1, started_at=0.0))
    text = TelegramInterface.format_result(result)
    assert "completed" in text
    assert "It worked." in text
    assert "artifacts/report.md" in text
    assert "t1" in text


def test_format_result_for_a_blocked_task():
    result = RunResult(task_id="t2", objective="x", status="blocked", answer="Need approval.")
    assert "blocked" in TelegramInterface.format_result(result)


# --------------------------------------------------------------------------
# HTTP surface
# --------------------------------------------------------------------------
def _build(tmp_settings, provider):
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
    registry = ToolRegistry()
    registry.register_all(build_tools())
    agent = Agent(
        router=ModelRouter(provider, [tmp_settings.model.primary]),
        settings=tmp_settings,
        registry=registry,
        skills=SkillLoader(),
        authorizer=Authorizer(tmp_settings.telegram.allowed_users),
        rate_limiter=RateLimiter(100),
        gate=ApprovalGate(),
        memory=memory,
        sandbox=Sandbox(tmp_settings.sandbox, tmp_settings.workspace_root),
    )

    class _Runtime:
        pass

    runtime = _Runtime()
    runtime.settings = tmp_settings
    runtime.store = store
    runtime.memory = memory
    runtime.router = agent.router
    runtime.agent = agent
    runtime.events = agent.events
    runtime.skills = agent.skills
    runtime.sandbox = agent.sandbox

    async def _aclose():
        return None

    runtime.aclose = _aclose
    return runtime


def test_health_is_public_and_leaks_nothing(tmp_settings):
    app = create_app(_build(tmp_settings, MockProvider([])))
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        # No paths, hostnames or secrets.
        text = resp.text
        assert "workspace" not in text
        assert "api-token" not in text


def test_private_endpoints_require_the_api_token(tmp_settings):
    app = create_app(_build(tmp_settings, MockProvider([])))
    with TestClient(app) as client:
        assert client.get("/self").status_code == 401
        assert client.get("/tasks").status_code == 401
        assert client.get("/events").status_code == 401
        ok = client.get("/self", headers={"Authorization": f"Bearer {tmp_settings.api_auth_token}"})
        assert ok.status_code == 200
        assert "model_health" in ok.json()


def test_private_endpoints_are_disabled_without_a_configured_token(tmp_settings):
    settings = type(tmp_settings)(**{**tmp_settings.__dict__, "api_auth_token": None})
    app = create_app(_build(settings, MockProvider([])))
    with TestClient(app) as client:
        assert client.get("/self", headers={"Authorization": "Bearer anything"}).status_code == 401


def test_create_task_validates_the_objective(tmp_settings):
    app = create_app(_build(tmp_settings, MockProvider([{"content": "done"}])))
    headers = {"Authorization": f"Bearer {tmp_settings.api_auth_token}"}
    with TestClient(app) as client:
        assert client.post("/tasks", json={}, headers=headers).status_code == 422
        assert client.post("/tasks", json={"objective": "   "}, headers=headers).status_code == 422


def test_unknown_task_returns_404(tmp_settings):
    app = create_app(_build(tmp_settings, MockProvider([])))
    headers = {"Authorization": f"Bearer {tmp_settings.api_auth_token}"}
    with TestClient(app) as client:
        assert client.get("/tasks/nope", headers=headers).status_code == 404


def test_telegram_webhook_rejects_a_bad_secret(tmp_settings):
    tmp_settings = type(tmp_settings)(
        **{
            **tmp_settings.__dict__,
            "telegram": tmp_settings.telegram.__class__(webhook_secret="real-secret"),
        }
    )
    runtime = _build(tmp_settings, MockProvider([]))
    tg = TelegramInterface(agent=runtime.agent, settings=tmp_settings, client=None)
    app = create_app(runtime, tg)
    with TestClient(app) as client:
        resp = client.post(
            "/telegram/webhook",
            json={"update_id": 1},
            headers={"X-Telegram-Bot-Api-Secret-Token": "wrong"},
        )
        assert resp.status_code == 403


def test_telegram_webhook_without_an_interface_returns_503(tmp_settings):
    app = create_app(_build(tmp_settings, MockProvider([])), None)
    with TestClient(app) as client:
        assert client.post("/telegram/webhook", json={"update_id": 1}).status_code == 503

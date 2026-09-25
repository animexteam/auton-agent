"""Shared test fixtures.

Every test runs against a temporary workspace and an in-memory model, so the
suite is deterministic, fast and offline.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agentcore.config import (  # noqa: E402
    AgentConfig,
    ModelConfig,
    PersistenceConfig,
    SandboxConfig,
    Settings,
    TelegramConfig,
)
from agentcore.llm.mock import MockProvider  # noqa: E402
from agentcore.persistence import DiskStore  # noqa: E402
from agentcore.redaction import configure as configure_redaction  # noqa: E402


@pytest.fixture(autouse=True)
def clean_secret_env(monkeypatch):
    """Keep the redactor deterministic and never touch real credentials."""
    for name in (
        "OLLAMA_API_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_WEBHOOK_SECRET",
        "API_AUTH_TOKEN", "GIST_API_KEY", "GITHUB_API_KEY",
        "RENDER_API_KEY_1", "RENDER_API_KEY_2",
    ):
        monkeypatch.delenv(name, raising=False)
    configure_redaction({})
    yield


@pytest.fixture
def tmp_settings(tmp_path: Path) -> Settings:
    return Settings(
        model=ModelConfig(api_key="test-key", primary="mock-primary", fallbacks=("mock-fallback",)),
        telegram=TelegramConfig(bot_token=None, allowed_users=("42",)),
        persistence=PersistenceConfig(backend="disk"),
        sandbox=SandboxConfig(enabled=True, timeout_seconds=10, max_output_bytes=20_000, max_memory_mb=512),
        agent=AgentConfig(max_steps=8, max_seconds=60, session_turns=6),
        workspace_root=tmp_path / "workspace",
        state_root=tmp_path / "state",
        api_auth_token="test-api-token",
        github_api_key=None,
        github_username="tester",
    )


@pytest.fixture
def disk_store(tmp_settings: Settings) -> DiskStore:
    return DiskStore(tmp_settings.state_root)


@pytest.fixture
def mock_provider() -> MockProvider:
    return MockProvider()

"""Environment-driven configuration.

Everything the agent needs is read from the process environment, so the same
container image runs unchanged on Render, a VPS, or a laptop. No value is
hard-coded and no secret is ever written to disk by the application.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

_TRUE = {"1", "true", "yes", "on", "y"}
_FALSE = {"0", "false", "no", "off", "n"}

#: The primary reasoning model. Chosen on measured evidence, not on name
#: recognition: the Ollama Cloud catalogue was probed model by model, and this is
#: the largest model the account is actually entitled to that also supports tool
#: calling. Verified live: it answers, and it emits correct tool calls.
DEFAULT_PRIMARY_MODEL = "gpt-oss:120b"

#: Models tried, in order, when the primary is unavailable or not entitled.
#: Every entry below was confirmed reachable on the account; nothing here is
#: aspirational. The chain degrades in capability, never in correctness.
DEFAULT_FALLBACK_MODELS = (
    "nemotron-3-ultra",
    "gpt-oss:20b",
    "nemotron-3-nano:30b",
    "nemotron-3-super",
    "gemma4:31b",
)

#: Environment variable names whose *values* must never appear in a log line,
#: an event record, a model prompt, or a tool result.
SECRET_ENV_NAMES: tuple[str, ...] = (
    "OLLAMA_API_KEY",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_WEBHOOK_SECRET",
    "API_AUTH_TOKEN",
    "GIST_API_KEY",
    "GITHUB_API_KEY",
    "RENDER_API_KEY_1",
    "RENDER_API_KEY_2",
    "GITHUB_TOKEN",
)


def env_str(name: str, default: str | None = None) -> str | None:
    raw = os.environ.get(name)
    if raw is None:
        return default
    raw = raw.strip()
    return raw if raw else default


def env_bool(name: str, default: bool = False) -> bool:
    raw = env_str(name)
    if raw is None:
        return default
    low = raw.lower()
    if low in _TRUE:
        return True
    if low in _FALSE:
        return False
    return default


def env_int(name: str, default: int) -> int:
    raw = env_str(name)
    if raw is None:
        return default
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return default


def env_float(name: str, default: float) -> float:
    raw = env_str(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def env_list(name: str, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    raw = env_str(name)
    if raw is None:
        return tuple(default)
    parts = raw.replace(";", ",").split(",")
    return tuple(p.strip() for p in parts if p.strip())


@dataclass(frozen=True)
class ModelConfig:
    provider: str = "ollama_cloud"
    base_url: str = "https://ollama.com"
    api_key: str | None = None
    primary: str = "gpt-oss:120b"
    fallbacks: tuple[str, ...] = DEFAULT_FALLBACK_MODELS
    timeout_seconds: int = 240
    max_retries: int = 3
    temperature: float = 0.2

    @property
    def chain(self) -> tuple[str, ...]:
        """Primary model first, then the de-duplicated fallback chain."""
        seen: list[str] = []
        for name in (self.primary, *self.fallbacks):
            if name and name not in seen:
                seen.append(name)
        return tuple(seen)

    @property
    def configured(self) -> bool:
        return bool(self.api_key)


@dataclass(frozen=True)
class TelegramConfig:
    bot_token: str | None = None
    mode: str = "webhook"
    webhook_secret: str | None = None
    allowed_users: tuple[str, ...] = ()
    max_requests_per_minute: int = 6

    @property
    def enabled(self) -> bool:
        return bool(self.bot_token)


@dataclass(frozen=True)
class PersistenceConfig:
    backend: str = "chained"
    gist_id: str | None = None
    gist_api_key: str | None = None
    filename_prefix: str = "auton-agent"

    @property
    def gist_ready(self) -> bool:
        # Only the API key is required: the gist itself is discovered or created at
        # runtime, so the operator never has to hardcode an id.
        return bool(self.gist_api_key)


@dataclass(frozen=True)
class SandboxConfig:
    enabled: bool = True
    timeout_seconds: int = 60
    max_output_bytes: int = 120_000
    max_memory_mb: int = 768
    max_cpu_seconds: int = 60
    allow_destructive: bool = False


@dataclass(frozen=True)
class AgentConfig:
    max_steps: int = 40
    max_seconds: int = 900
    session_turns: int = 12


@dataclass(frozen=True)
class Settings:
    model: ModelConfig
    telegram: TelegramConfig
    persistence: PersistenceConfig
    sandbox: SandboxConfig
    agent: AgentConfig
    workspace_root: Path
    state_root: Path
    api_auth_token: str | None = None
    github_api_key: str | None = None
    github_username: str | None = None
    log_level: str = "INFO"
    port: int = 10_000
    extra: dict[str, str] = field(default_factory=dict)

    def secret_values(self) -> dict[str, str]:
        """Map of secret-env-name -> live value, for redaction.

        Only values that are actually present are returned, so the redactor
        never accidentally scrubs an empty string (which would garble output).
        """
        found: dict[str, str] = {}
        for name in SECRET_ENV_NAMES:
            value = os.environ.get(name)
            if value and len(value) >= 8:
                found[name] = value
        return found


def _resolve(base: Path, raw: str) -> Path:
    p = Path(raw)
    if not p.is_absolute():
        p = (base / p).resolve()
    return p


def load_settings(base_dir: Path | None = None) -> Settings:
    """Build :class:`Settings` from the current environment."""
    root = base_dir or Path.cwd()

    model = ModelConfig(
        provider=env_str("MODEL_PROVIDER", "ollama_cloud") or "ollama_cloud",
        base_url=(env_str("OLLAMA_BASE_URL", "https://ollama.com") or "https://ollama.com").rstrip("/"),
        api_key=env_str("OLLAMA_API_KEY"),
        primary=env_str("MODEL_PRIMARY", DEFAULT_PRIMARY_MODEL) or DEFAULT_PRIMARY_MODEL,
        fallbacks=env_list("MODEL_FALLBACKS", DEFAULT_FALLBACK_MODELS),
        timeout_seconds=env_int("MODEL_TIMEOUT_SECONDS", 240),
        max_retries=env_int("MODEL_MAX_RETRIES", 3),
        temperature=env_float("MODEL_TEMPERATURE", 0.2),
    )

    telegram = TelegramConfig(
        bot_token=env_str("TELEGRAM_BOT_TOKEN"),
        mode=(env_str("TELEGRAM_MODE", "webhook") or "webhook").lower(),
        webhook_secret=env_str("TELEGRAM_WEBHOOK_SECRET"),
        allowed_users=env_list("TELEGRAM_ALLOWED_USERS"),
        max_requests_per_minute=env_int("TELEGRAM_MAX_REQUESTS_PER_MINUTE", 6),
    )

    persistence = PersistenceConfig(
        backend=(env_str("PERSISTENCE_BACKEND", "chained") or "chained").lower(),
        gist_id=env_str("GIST_ID"),
        gist_api_key=env_str("GIST_API_KEY") or env_str("GITHUB_API_KEY"),
        filename_prefix=env_str("GIST_FILENAME_PREFIX", "auton-agent") or "auton-agent",
    )

    sandbox = SandboxConfig(
        enabled=env_bool("SANDBOX_ENABLED", True),
        timeout_seconds=env_int("SANDBOX_TIMEOUT_SECONDS", 60),
        max_output_bytes=env_int("SANDBOX_MAX_OUTPUT_BYTES", 120_000),
        max_memory_mb=env_int("SANDBOX_MAX_MEMORY_MB", 768),
        max_cpu_seconds=env_int("SANDBOX_MAX_CPU_SECONDS", 60),
        allow_destructive=env_bool("ALLOW_DESTRUCTIVE", False),
    )

    agent = AgentConfig(
        max_steps=env_int("AGENT_MAX_STEPS", 40),
        max_seconds=env_int("AGENT_MAX_SECONDS", 900),
        session_turns=env_int("AGENT_SESSION_TURNS", 12),
    )

    return Settings(
        model=model,
        telegram=telegram,
        persistence=persistence,
        sandbox=sandbox,
        agent=agent,
        workspace_root=_resolve(root, env_str("WORKSPACE_ROOT", "./workspace") or "./workspace"),
        state_root=_resolve(root, env_str("STATE_ROOT", "./.agentstate") or "./.agentstate"),
        api_auth_token=env_str("API_AUTH_TOKEN"),
        github_api_key=env_str("GITHUB_API_KEY"),
        github_username=env_str("GITHUB_USERNAME"),
        log_level=(env_str("LOG_LEVEL", "INFO") or "INFO").upper(),
        port=env_int("PORT", 10_000),
    )

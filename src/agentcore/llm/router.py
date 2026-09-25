"""Model router — an ordered fallback chain over one provider.

Why a chain rather than one model: the free model tier available to this
project does not include every model the account can *see*. Rather than fail
the task, the router walks the configured chain and remembers what worked, so
subsequent calls in the same run go straight to a working model.

The router also degrades gracefully: if the primary model is permanently
rejected it is blacklisted for the process lifetime with a cooldown, and the
next model takes over. Transient failures are retried by the provider itself.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..config import ModelConfig
from ..errors import ModelUnavailableError
from .base import LLMProvider, LLMResponse, Message
from .ollama_cloud import OllamaCloudProvider

log = logging.getLogger(__name__)

COOLDOWN_SECONDS = 1800


@dataclass
class ModelHealth:
    """Per-model reputation, so the router can avoid known-bad models."""

    name: str
    failures: int = 0
    successes: int = 0
    disabled_until: float = 0.0
    last_error: str | None = None

    @property
    def available(self) -> bool:
        return time.time() >= self.disabled_until


@dataclass
class RouterStats:
    total_calls: int = 0
    failures: int = 0
    fallbacks: int = 0
    last_model: str | None = None
    per_model: dict[str, ModelHealth] = field(default_factory=dict)


class ModelRouter:
    def __init__(self, provider: LLMProvider, models: Sequence[str]) -> None:
        self._provider = provider
        self._models: list[str] = [m for m in models if m]
        if not self._models:
            raise ValueError("ModelRouter needs at least one model name")
        self.stats = RouterStats(
            per_model={m: ModelHealth(name=m) for m in self._models}
        )
        self._preferred: str | None = None

    @property
    def provider(self) -> LLMProvider:
        return self._provider

    @property
    def models(self) -> tuple[str, ...]:
        return tuple(self._models)

    def health(self) -> dict[str, Any]:
        return {
            "provider": self._provider.name,
            "provider_usable": self._provider.available()[1],
            "preferred": self._preferred,
            "calls": self.stats.total_calls,
            "failures": self.stats.failures,
            "fallbacks": self.stats.fallbacks,
            "models": [
                {
                    "name": m.name,
                    "successes": m.successes,
                    "failures": m.failures,
                    "available": m.available,
                    "last_error": m.last_error,
                }
                for m in self.stats.per_model.values()
            ],
        }

    def _order(self) -> list[str]:
        if self._preferred and self._preferred in self.stats.per_model:
            head = self._preferred
            rest = [m for m in self._models if m != head]
            ordered = [head, *rest]
        else:
            ordered = list(self._models)
        usable = [m for m in ordered if self.stats.per_model[m].available]
        # If every model is cooling down, try them all rather than refuse.
        return usable or ordered

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        temperature: float | None = None,
        model_hint: str | None = None,
    ) -> LLMResponse:
        candidates = self._order()
        if model_hint and model_hint in self.stats.per_model:
            candidates = [model_hint, *[m for m in candidates if m != model_hint]]

        errors: list[str] = []
        for index, name in enumerate(candidates):
            health = self.stats.per_model[name]
            self.stats.total_calls += 1
            if index > 0:
                self.stats.fallbacks += 1
                log.warning("falling back to model", extra={"model": name})
            try:
                response = await self._provider.chat(
                    messages, tools=tools, model=name, temperature=temperature
                )
            except ModelUnavailableError as exc:
                health.failures += 1
                health.last_error = exc.message
                health.disabled_until = time.time() + COOLDOWN_SECONDS
                self.stats.failures += 1
                errors.append(f"{name}: {exc.message}")
                if model_hint == name:
                    # The caller asked specifically; do not silently substitute.
                    raise
                continue
            except Exception as exc:  # noqa: BLE001 - try the next model, then report
                health.failures += 1
                health.last_error = str(exc)[:200]
                self.stats.failures += 1
                errors.append(f"{name}: {exc}")
                if model_hint == name:
                    raise
                continue

            health.successes += 1
            self._preferred = name
            self.stats.last_model = name
            if response.model:
                response.model = name
            return response

        raise ModelUnavailableError(
            "every model in the chain failed",
            detail="; ".join(errors)[:800] or "no candidates",
        )

    async def aclose(self) -> None:
        await self._provider.aclose()


def build_provider(config: ModelConfig, client: Any | None = None) -> LLMProvider:
    """Instantiate the provider named by configuration.

    Adding a vendor = one branch here plus one :class:`LLMProvider` subclass.
    """
    provider = (config.provider or "ollama_cloud").lower()
    if provider in ("ollama_cloud", "ollama", "ollama-cloud"):
        return OllamaCloudProvider(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout_seconds=config.timeout_seconds,
            max_retries=config.max_retries,
            temperature=config.temperature,
            client=client,
        )
    raise ValueError(f"unknown MODEL_PROVIDER: {config.provider!r}")

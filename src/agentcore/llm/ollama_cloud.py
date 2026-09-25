"""Ollama Cloud provider.

Talks to the Ollama-native API (``/api/chat``) because it returns tool calls
with parsed-argument objects, which avoids the JSON-string round-trip that the
OpenAI-compatible layer requires. Auth is a bearer key.

Error classification is deliberate: an error the vendor will never accept
(model not entitled on this plan, unknown model) must NOT be retried — it
should immediately fall through the model chain instead.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Mapping, Sequence

import httpx

from ..errors import ModelError, ModelUnavailableError
from .base import LLMProvider, LLMResponse, Message, ToolCall, normalize_messages

log = logging.getLogger(__name__)

#: Substrings that mean "retrying this exact model is pointless".
_PERMANENT_MARKERS = (
    "not included in your free usage",
    "upgrade for included usage",
    "add usage credits",
    "not found",
    "does not exist",
    "unknown model",
    "unauthorized",
    "invalid api key",
)


class OllamaCloudProvider(LLMProvider):
    name = "ollama_cloud"

    def __init__(
        self,
        api_key: str | None,
        base_url: str = "https://ollama.com",
        timeout_seconds: int = 240,
        max_retries: int = 3,
        temperature: float = 0.2,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds
        self._max_retries = max(1, max_retries)
        self._temperature = temperature
        self._client = client
        self._owns_client = client is None

    # -- plumbing -------------------------------------------------------

    def available(self) -> tuple[bool, str]:
        if not self._api_key:
            return False, "OLLAMA_API_KEY is not set"
        return True, "ok"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    async def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
            self._owns_client = True
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    # -- api ------------------------------------------------------------

    async def list_models(self) -> list[str]:
        if not self._api_key:
            return []
        client = await self._http()
        try:
            resp = await client.get(f"{self._base_url}/api/tags", headers=self._headers())
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:  # noqa: BLE001 - surfaced as "unknown" upstream
            log.warning("model listing failed", extra={"error": str(exc)})
            return []
        return [m.get("name", "") for m in payload.get("models", []) if m.get("name")]

    def _classify(self, status: int, body: str) -> Exception:
        low = body.lower()
        if any(marker in low for marker in _PERMANENT_MARKERS):
            return ModelUnavailableError(
                f"model rejected permanently (http {status})", detail=body[:400]
            )
        if status in (401, 403):
            return ModelUnavailableError(
                "model provider rejected the credential", detail=f"http {status}"
            )
        return ModelError(f"model call failed (http {status})", detail=body[:400])

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        model: str | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        ok, reason = self.available()
        if not ok:
            raise ModelUnavailableError(reason)
        target = model or ""
        if not target:
            raise ModelError("no model name supplied to provider")

        body: dict[str, Any] = {
            "model": target,
            "messages": normalize_messages(messages),
            "stream": False,
            "options": {
                "temperature": self._temperature if temperature is None else temperature,
                "num_predict": 4096,
            },
        }
        if tools:
            body["tools"] = list(tools)

        client = await self._http()
        last_error: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            try:
                resp = await client.post(
                    f"{self._base_url}/api/chat", headers=self._headers(), json=body
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = ModelError(f"transport failure: {exc}")
                if attempt < self._max_retries:
                    await asyncio.sleep(min(2 ** attempt, 8))
                    continue
                raise last_error from exc

            if resp.status_code >= 400:
                error = self._classify(resp.status_code, resp.text)
                if isinstance(error, ModelUnavailableError):
                    raise error
                last_error = error
                if attempt < self._max_retries:
                    await asyncio.sleep(min(2 ** attempt, 8))
                    continue
                raise error

            try:
                payload = resp.json()
            except json.JSONDecodeError as exc:
                last_error = ModelError("provider returned non-JSON body")
                if attempt < self._max_retries:
                    await asyncio.sleep(min(2 ** attempt, 8))
                    continue
                raise last_error from exc

            # Ollama can return 200 with an {"error": ...} body.
            if isinstance(payload, dict) and payload.get("error"):
                error = self._classify(200, str(payload["error"]))
                if isinstance(error, ModelUnavailableError):
                    raise error
                last_error = error
                if attempt < self._max_retries:
                    await asyncio.sleep(min(2 ** attempt, 8))
                    continue
                raise error

            message = (payload or {}).get("message") or {}
            raw_calls = message.get("tool_calls") or []
            calls = [ToolCall.from_raw(c, i) for i, c in enumerate(raw_calls)]
            return LLMResponse(
                content=message.get("content") or "",
                tool_calls=[c for c in calls if c.name],
                model=target,
                usage={
                    "prompt_tokens": payload.get("prompt_eval_count", 0),
                    "completion_tokens": payload.get("eval_count", 0),
                },
                finish_reason=payload.get("done_reason"),
            )

        raise last_error or ModelError("model call failed with no captured error")

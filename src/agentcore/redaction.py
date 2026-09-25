"""Secret redaction.

Secrets must never reach a log file, an event record, a Telegram message or a
model prompt. We do two things:

1. Substitute the *literal live values* of known secret environment variables.
2. Substitute anything matching a set of high-signal credential shapes, so a
   secret pasted into a task by a user is still scrubbed.
"""

from __future__ import annotations

import re
from typing import Iterable, Mapping, Pattern

PLACEHOLDER = "[REDACTED]"

#: Credential shapes worth scrubbing even when we were never told the value.
_PATTERNS: tuple[Pattern[str], ...] = (
    re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"),          # GitHub PAT (classic)
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),  # GitHub PAT (fine-grained)
    re.compile(r"\bghs_[A-Za-z0-9]{20,}\b"),          # GitHub app token
    re.compile(r"\brnd_[A-Za-z0-9]{16,}\b"),          # Render API key
    re.compile(r"\bsk-[A-Za-z0-9\-_]{16,}\b"),        # OpenAI-style key
    re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b"), # Slack token
    re.compile(r"\b\d{8,12}:[A-Za-z0-9_\-]{30,}\b"),  # Telegram bot token
    re.compile(r"\bAIza[0-9A-Za-z\-_]{30,}\b"),       # Google API key
    re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b"),  # JWT
)


class Redactor:
    """Idempotent scrubber for arbitrary strings and nested structures."""

    def __init__(self, secrets: Mapping[str, str] | Iterable[str] | None = None) -> None:
        if secrets is None:
            values: list[str] = []
        elif isinstance(secrets, Mapping):
            values = [v for v in secrets.values() if v]
        else:
            values = [v for v in secrets if v]
        # Longest first so a secret that contains another secret is fully eaten.
        self._values: tuple[str, ...] = tuple(
            sorted({v for v in values if len(v) >= 8}, key=len, reverse=True)
        )

    def scrub(self, text: str) -> str:
        if not text:
            return text
        out = text
        for value in self._values:
            if value in out:
                out = out.replace(value, PLACEHOLDER)
        for pattern in _PATTERNS:
            out = pattern.sub(PLACEHOLDER, out)
        return out

    def scrub_deep(self, value: object) -> object:
        """Recursively scrub strings inside dicts / lists / tuples."""
        if isinstance(value, str):
            return self.scrub(value)
        if isinstance(value, Mapping):
            return {k: self.scrub_deep(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            scrubbed = [self.scrub_deep(v) for v in value]
            return type(value)(scrubbed) if isinstance(value, tuple) else scrubbed
        return value


_GLOBAL: Redactor = Redactor()


def configure(secrets: Mapping[str, str] | Iterable[str] | None) -> Redactor:
    """Install the process-wide redactor built from the live secrets."""
    global _GLOBAL
    _GLOBAL = Redactor(secrets)
    return _GLOBAL


def redactor() -> Redactor:
    return _GLOBAL


def scrub(text: str) -> str:
    """Module-level convenience using the process-wide redactor."""
    return _GLOBAL.scrub(text)

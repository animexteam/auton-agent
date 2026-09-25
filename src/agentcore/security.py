"""Security boundaries.

Three distinct concerns live here, all fail-closed:

* **Authorisation** — who may drive the agent. An empty allowlist denies
  everyone; there is no "open by default" mode.
* **Rate limiting** — a per-principal token bucket so one chat cannot exhaust
  the free model quota.
* **Approval gates** — destructive or infrastructure-mutating actions require
  an explicit, human-granted approval token instead of being silently allowed.
"""

from __future__ import annotations

import hmac
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from .errors import ApprovalRequired, PermissionDenied, RateLimited

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Authorisation
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Principal:
    """The identity driving a task."""

    kind: str  # "telegram" | "api" | "cli" | "system"
    id: str

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.kind}:{self.id}"


class Authorizer:
    """Allowlist authorisation. Unknown principals are denied, not trusted."""

    def __init__(self, allowed_telegram_users: tuple[str, ...] = ()) -> None:
        self._allowed = {u.strip() for u in allowed_telegram_users if u.strip()}

    @property
    def allowed_count(self) -> int:
        return len(self._allowed)

    def is_open(self) -> bool:
        """True only when nobody at all may drive the agent via Telegram."""
        return not self._allowed

    def check(self, principal: Principal) -> None:
        if principal.kind == "telegram":
            if not self._allowed:
                raise PermissionDenied(
                    "telegram access is disabled: TELEGRAM_ALLOWED_USERS is empty "
                    "(set it to the numeric Telegram user IDs allowed to use this bot)"
                )
            if str(principal.id) not in self._allowed:
                raise PermissionDenied(f"telegram user {principal.id} is not on the allowlist")
            return
        if principal.kind in ("api", "cli", "system"):
            return
        raise PermissionDenied(f"unknown principal kind: {principal.kind}")


# --------------------------------------------------------------------------
# Rate limiting
# --------------------------------------------------------------------------
class RateLimiter:
    """Fixed-window per-principal limiter.

    Deliberately simple: a monotonically-refilled counter per principal, no
    background sweeper thread, bounded memory via periodic pruning.
    """

    def __init__(self, per_minute: int = 6, window_seconds: int = 60, max_principals: int = 512) -> None:
        self._limit = max(1, per_minute)
        self._window = window_seconds
        self._max_principals = max_principals
        self._state: dict[str, tuple[float, int]] = {}
        self._lock = threading.Lock()

    @property
    def limit(self) -> int:
        return self._limit

    def check(self, principal: Principal) -> None:
        key = str(principal)
        now = time.time()
        with self._lock:
            if len(self._state) > self._max_principals:
                cutoff = now - self._window
                self._state = {k: v for k, v in self._state.items() if v[0] > cutoff}
            window_start, count = self._state.get(key, (now, 0))
            if now - window_start >= self._window:
                window_start, count = now, 0
            if count >= self._limit:
                retry_after = int(self._window - (now - window_start)) + 1
                raise RateLimited(
                    f"rate limit reached for {key}: max {self._limit} requests per "
                    f"{self._window}s. Try again in ~{retry_after}s.",
                    detail=str(retry_after),
                )
            self._state[key] = (window_start, count + 1)

    def remaining(self, principal: Principal) -> int:
        with self._lock:
            window_start, count = self._state.get(str(principal), (time.time(), 0))
        if time.time() - window_start >= self._window:
            return self._limit
        return max(0, self._limit - count)


# --------------------------------------------------------------------------
# Approval gates
# --------------------------------------------------------------------------
@dataclass
class ApprovalGate:
    """Tracks which privileged actions a human has authorised.

    Nothing is authorised by default. `ALLOW_DESTRUCTIVE=true` grants a
    standing approval for destructive sandbox commands; every other privileged
    action (delete_file, infrastructure mutation) needs a one-shot token that
    the human hands over out of band.
    """

    allow_destructive: bool = False
    _tokens: set[str] = field(default_factory=set)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def grant(self, token: str) -> None:
        token = (token or "").strip()
        if not token:
            raise ValueError("approval token must not be empty")
        with self._lock:
            self._tokens.add(token)

    def revoke(self, token: str) -> None:
        with self._lock:
            self._tokens.discard(token)

    def is_granted(self, token: str | None) -> bool:
        if not token:
            return False
        with self._lock:
            return token in self._tokens

    def require(self, *, action: str, approval_token: str | None, destructive: bool = False) -> None:
        if destructive and self.allow_destructive:
            return
        if self.is_granted(approval_token):
            return
        raise ApprovalRequired(
            f"'{action}' requires explicit approval. Re-run with a valid approval token "
            f"(POST /approvals to obtain one) or set ALLOW_DESTRUCTIVE=true to permit "
            f"destructive sandbox commands."
        )


# --------------------------------------------------------------------------
# Path confinement
# --------------------------------------------------------------------------
class PathGuard:
    """Confines every filesystem operation to the agent workspace."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve(self, raw: str, *, must_exist: bool = False) -> Path:
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        try:
            resolved = candidate.resolve(strict=False)
        except OSError as exc:
            raise PermissionDenied(f"unresolvable path {raw!r}: {exc}") from exc

        if resolved != self.root and self.root not in resolved.parents:
            raise PermissionDenied(
                f"path escapes the agent workspace: {raw!r} "
                f"(everything must live under {self.root})"
            )
        if must_exist and not resolved.exists():
            raise PermissionDenied(f"path does not exist: {raw!r}")
        return resolved

    def relative(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.root))
        except ValueError:
            return str(path)


# --------------------------------------------------------------------------
# Webhook verification
# --------------------------------------------------------------------------
def verify_webhook_secret(expected: str | None, provided: str | None) -> None:
    """Constant-time comparison of the Telegram webhook secret."""
    if not expected:
        return  # no secret configured -> nothing to verify (documented, not silent)
    if not provided or not hmac.compare_digest(expected, provided):
        raise PermissionDenied("telegram webhook secret mismatch")


def require_api_token(expected: str | None, provided: str | None) -> None:
    """Bearer-token check for the private HTTP surface."""
    if not expected:
        raise PermissionDenied("the HTTP API surface is disabled (API_AUTH_TOKEN is not set)")
    if not provided or not hmac.compare_digest(expected, provided):
        raise PermissionDenied("invalid API token")

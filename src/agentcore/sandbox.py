"""Controlled execution sandbox.

The agent is allowed to run programs, but never *unbounded* ones. Every
command goes through this module, which applies, in order:

1. **Admission control** — a destructive-pattern deny-list that refuses
   obviously catastrophic commands unless the operator has explicitly opted in.
2. **Resource limits** — wall-clock timeout, address-space cap, CPU-time cap
   and file-size cap via ``setrlimit`` in the child, so a runaway process
   cannot take the container (and therefore the whole agent) down.
3. **Output bounding** — stdout/stderr are capped, so a tool cannot blow up
   the model context with a gigabyte of logs.
4. **Redaction** — output is scrubbed before it ever reaches the model.

On Render Free (0.1 CPU / 512 MB) the limits are what make shell execution
safe at all: without them one bad command is a full outage.
"""

from __future__ import annotations

import asyncio
import logging
import os
import resource
import shlex
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .config import SandboxConfig
from .errors import ApprovalRequired, PermissionDenied, ToolError, ToolTimeout
from .redaction import redactor

log = logging.getLogger(__name__)

#: Patterns that are refused unless ALLOW_DESTRUCTIVE=true. These are the
#: commands that can destroy the host or the agent's own ability to function.
_DESTRUCTIVE_PATTERNS: tuple[str, ...] = (
    "rm -rf /",
    "rm -rf /*",
    "rm -rf ~",
    "mkfs",
    "dd if=/dev/zero",
    "dd if=/dev/random",
    ":(){:|:&};:",
    "shutdown",
    "reboot",
    "poweroff",
    "halt",
    "> /dev/sda",
    "chmod -R 777 /",
    "chown -R",
    "iptables -F",
    "userdel",
    "passwd root",
    "kill -9 1",
    "kill -9 -1",
)

#: Commands that are usually wanted but must be declared, because they can
#: still be abused. Kept explicit rather than clever.
_SENSITIVE_PATTERNS: tuple[str, ...] = (
    "curl | sh",
    "curl | bash",
    "wget | sh",
    "wget | bash",
)

_BLOCKED_ALWAYS = (":(){:|:&};:",)


@dataclass
class ExecResult:
    command: str
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
    duration_ms: int
    truncated: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def to_payload(self) -> dict[str, Any]:
        return {
            "command": redactor().scrub(self.command),
            "exit_code": self.exit_code,
            "ok": self.ok,
            "timed_out": self.timed_out,
            "duration_ms": self.duration_ms,
            "truncated": self.truncated,
            "stdout": redactor().scrub(self.stdout),
            "stderr": redactor().scrub(self.stderr),
        }

    def summary(self, max_chars: int = 4000) -> str:
        head = f"$ {self.command}\nexit={self.exit_code} ({self.duration_ms}ms)"
        if self.timed_out:
            head += " [TIMED OUT]"
        if self.truncated:
            head += " [OUTPUT TRUNCATED]"
        parts = [head]
        if self.stdout:
            parts.append(f"--- stdout ---\n{self.stdout}")
        if self.stderr:
            parts.append(f"--- stderr ---\n{self.stderr}")
        text = redactor().scrub("\n".join(parts))
        return text[:max_chars]


def screen_command(command: str, allow_destructive: bool) -> None:
    """Screen a command before it runs.

    Two tiers, because they are genuinely different risks:

    * **Never allowed** (``_BLOCKED_ALWAYS``) — a fork bomb or equivalent. No
      approval can unlock these; they are a straight refusal.
    * **Needs approval** (``_DESTRUCTIVE_PATTERNS``) — wiping a filesystem,
      rebooting, killing init. Legitimate in rare cases, so they escalate to
      ``ApprovalRequired`` (which the loop reports to the human) rather than
      being silently permitted. Setting ``ALLOW_DESTRUCTIVE=true`` grants a
      standing approval for these.

    Ordinary commands (``ls``, ``git``, ``pytest``, ``curl``) pass through: the
    authorisation for running *any* command comes from the principal being on
    the allowlist, and the blast radius is contained by the resource limits in
    :meth:`Sandbox.run`.
    """
    import re  # noqa: F401  (kept for callers that extend the screeners)

    normalised = " ".join(command.split())
    for pattern in _BLOCKED_ALWAYS:
        if pattern in normalised:
            raise PermissionDenied(f"refused: command matches a blocked pattern ({pattern})")
    if allow_destructive:
        return
    pattern = destructive_pattern(command)
    if pattern:
        raise ApprovalRequired(
            f"'{pattern}' is a destructive command. It needs explicit approval: "
            f"grant an approval token, or set ALLOW_DESTRUCTIVE=true to allow such "
            f"commands for this deployment."
        )


def destructive_pattern(command: str) -> str | None:
    """Return the destructive pattern a command matches, or ``None``.

    Exposed separately so a caller that *has* an approval can distinguish "this
    needs approval" from "this is fine" without duplicating the pattern list.
    """
    lowered = " ".join(command.split()).lower()
    for pattern in _DESTRUCTIVE_PATTERNS:
        if pattern in lowered:
            return pattern
    return None


def assert_not_blocked(command: str) -> None:
    """Raise for commands that no approval can ever unlock."""
    normalised = " ".join(command.split())
    for pattern in _BLOCKED_ALWAYS:
        if pattern in normalised:
            raise PermissionDenied(f"refused: command matches a blocked pattern ({pattern})")


class Sandbox:
    """Async command runner with hard resource limits."""

    def __init__(self, config: SandboxConfig, workspace_root: Path) -> None:
        self.config = config
        self.workspace_root = Path(workspace_root)
        self.workspace_root.mkdir(parents=True, exist_ok=True)

    def _limits(self) -> "list[tuple[int, int]]":  # pragma: no cover - set in child
        limits: list[tuple[int, int]] = []
        if self.config.max_memory_mb > 0:
            nbytes = self.config.max_memory_mb * 1024 * 1024
            limits.append((resource.RLIMIT_AS, nbytes))
            limits.append((resource.RLIMIT_DATA, nbytes))
        if self.config.max_cpu_seconds > 0:
            limits.append((resource.RLIMIT_CPU, self.config.max_cpu_seconds))
        try:
            limits.append((resource.RLIMIT_FSIZE, 64 * 1024 * 1024))
        except (AttributeError, ValueError):  # pragma: no cover
            pass
        try:
            limits.append((resource.RLIMIT_NPROC, 256))
        except (AttributeError, ValueError):  # pragma: no cover
            pass
        return limits

    async def run(
        self,
        command: str | Sequence[str],
        *,
        cwd: str | Path | None = None,
        env: dict[str, str] | None = None,
        timeout_seconds: float | None = None,
        shell: bool = True,
        approval_token: str | None = None,
    ) -> ExecResult:
        """Run a command.

        ``approval_token`` is the escape hatch for the destructive tier: a caller
        that has already obtained human approval passes the token, and only then is
        :func:`screen_command`'s destructive check relaxed. The always-blocked tier
        (fork bombs and the like) is enforced regardless — no token unlocks it.
        """
        if not self.config.enabled:
            raise ToolError("command execution is disabled (SANDBOX_ENABLED=false)")

        command_str = command if isinstance(command, str) else shlex.join(list(command))
        if approval_token:
            assert_not_blocked(command_str)
        else:
            screen_command(command_str, self.config.allow_destructive)

        timeout = float(timeout_seconds or self.config.timeout_seconds)
        workdir = Path(cwd) if cwd is not None else self.workspace_root
        workdir.mkdir(parents=True, exist_ok=True)

        # A deliberately sparse child environment: no parent secrets leak in.
        child_env = {
            "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
            "HOME": str(workdir),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "TMPDIR": str(workdir / ".tmp"),
        }
        Path(child_env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
        if env:
            child_env.update({k: str(v) for k, v in env.items()})

        limits = self._limits()

        def _preexec() -> None:  # pragma: no cover - runs in the forked child
            os.setsid()
            for resource_id, value in limits:
                try:
                    resource.setrlimit(resource_id, (value, value))
                except (ValueError, OSError):
                    continue

        started = time.perf_counter()
        timed_out = False

        try:
            proc = await asyncio.create_subprocess_shell(
                command_str,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(workdir),
                env=child_env,
                preexec_fn=_preexec,
            )
        except OSError as exc:
            raise ToolError(f"could not start command: {exc}") from exc

        try:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            exit_code = proc.returncode if proc.returncode is not None else -1
        except asyncio.TimeoutError:
            timed_out = True
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
            try:
                stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=10)
            except (asyncio.TimeoutError, Exception):  # noqa: BLE001
                stdout_b, stderr_b = b"", b""
            exit_code = -9

        duration_ms = int((time.perf_counter() - started) * 1000)
        cap = max(1024, self.config.max_output_bytes)
        raw_out = stdout_b or b""
        raw_err = stderr_b or b""
        truncated = len(raw_out) > cap or len(raw_err) > cap

        return ExecResult(
            command=command_str,
            exit_code=exit_code,
            stdout=raw_out[:cap].decode("utf-8", errors="replace"),
            stderr=raw_err[:cap].decode("utf-8", errors="replace"),
            timed_out=timed_out,
            duration_ms=duration_ms,
            truncated=truncated,
        )

    async def run_checked(self, command: str, **kwargs: Any) -> ExecResult:
        result = await self.run(command, **kwargs)
        if result.timed_out:
            raise ToolTimeout(
                f"command exceeded {kwargs.get('timeout_seconds', self.config.timeout_seconds)}s: {command[:120]}"
            )
        return result

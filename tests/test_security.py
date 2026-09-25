"""Security: authorisation, rate limiting, approval gates, webhook verification."""

from __future__ import annotations

import pytest

from agentcore.errors import ApprovalRequired, PermissionDenied, RateLimited
from agentcore.security import (
    ApprovalGate,
    Authorizer,
    PathGuard,
    Principal,
    RateLimiter,
    require_api_token,
    verify_webhook_secret,
)


# --------------------------------------------------------------------------
# authorisation
# --------------------------------------------------------------------------
def test_allowlisted_telegram_user_is_permitted():
    Authorizer(("42", "43")).check(Principal("telegram", "42"))


def test_unknown_telegram_user_is_denied():
    with pytest.raises(PermissionDenied):
        Authorizer(("42",)).check(Principal("telegram", "99"))


def test_empty_allowlist_denies_everyone():
    auth = Authorizer(())
    assert auth.is_open()
    with pytest.raises(PermissionDenied):
        auth.check(Principal("telegram", "42"))


def test_non_telegram_principals_pass_the_allowlist():
    auth = Authorizer(())
    auth.check(Principal("cli", "local"))
    auth.check(Principal("api", "token"))


# --------------------------------------------------------------------------
# rate limiting
# --------------------------------------------------------------------------
def test_rate_limiter_blocks_after_the_limit():
    limiter = RateLimiter(per_minute=3)
    principal = Principal("telegram", "42")
    for _ in range(3):
        limiter.check(principal)
    with pytest.raises(RateLimited) as exc:
        limiter.check(principal)
    assert "rate limit" in str(exc.value).lower()


def test_rate_limiter_is_per_principal():
    limiter = RateLimiter(per_minute=1)
    limiter.check(Principal("telegram", "1"))
    limiter.check(Principal("telegram", "2"))  # a different user is unaffected
    with pytest.raises(RateLimited):
        limiter.check(Principal("telegram", "1"))


def test_rate_limiter_reports_remaining():
    limiter = RateLimiter(per_minute=2)
    principal = Principal("cli", "x")
    assert limiter.remaining(principal) == 2
    limiter.check(principal)
    assert limiter.remaining(principal) == 1


# --------------------------------------------------------------------------
# approval gate
# --------------------------------------------------------------------------
def test_privileged_action_requires_a_token_by_default():
    gate = ApprovalGate(allow_destructive=False)
    with pytest.raises(ApprovalRequired):
        gate.require(action="run_command", approval_token=None)


def test_privileged_action_accepted_with_a_granted_token():
    gate = ApprovalGate()
    gate.grant("abc")
    gate.require(action="run_command", approval_token="abc")


def test_ungranted_token_is_rejected():
    gate = ApprovalGate()
    gate.grant("abc")
    with pytest.raises(ApprovalRequired):
        gate.require(action="run_command", approval_token="wrong")


def test_destructive_allow_flag_grants_standing_approval():
    gate = ApprovalGate(allow_destructive=True)
    gate.require(action="run_command", approval_token=None, destructive=True)


def test_destructive_flag_does_not_unlock_non_destructive_privileged_actions():
    gate = ApprovalGate(allow_destructive=True)
    with pytest.raises(ApprovalRequired):
        gate.require(action="delete_file", approval_token=None, destructive=False)


def test_revoked_token_stops_working():
    gate = ApprovalGate()
    gate.grant("abc")
    gate.revoke("abc")
    with pytest.raises(ApprovalRequired):
        gate.require(action="run_command", approval_token="abc")


def test_empty_approval_token_is_rejected_at_grant_time():
    with pytest.raises(ValueError):
        ApprovalGate().grant("   ")


# --------------------------------------------------------------------------
# path confinement
# --------------------------------------------------------------------------
def test_path_guard_confines_to_the_root(tmp_path):
    root = tmp_path / "workspace"
    guard = PathGuard(root)
    assert guard.resolve("a/b.txt") == (root / "a/b.txt").resolve()
    with pytest.raises(PermissionDenied):
        guard.resolve("../../etc/passwd")
    with pytest.raises(PermissionDenied):
        guard.resolve("/etc/passwd")


def test_path_guard_relative_reports_workspace_paths(tmp_path):
    guard = PathGuard(tmp_path / "workspace")
    assert guard.relative(guard.resolve("a/b.txt")) == "a/b.txt"


# --------------------------------------------------------------------------
# transport verification
# --------------------------------------------------------------------------
def test_webhook_secret_mismatch_is_rejected():
    with pytest.raises(PermissionDenied):
        verify_webhook_secret("expected-secret", "wrong-secret")
    with pytest.raises(PermissionDenied):
        verify_webhook_secret("expected-secret", None)


def test_webhook_secret_match_is_accepted():
    verify_webhook_secret("expected-secret", "expected-secret")


def test_webhook_without_configured_secret_is_a_noop():
    verify_webhook_secret(None, None)


def test_api_token_is_required():
    with pytest.raises(PermissionDenied):
        require_api_token(None, "anything")
    with pytest.raises(PermissionDenied):
        require_api_token("real", "wrong")
    require_api_token("real", "real")

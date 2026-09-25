"""Error taxonomy.

Every failure the agent can hit is classified, because the recovery strategy
depends on the class: transient failures are worth retrying, permanent ones
are not, and *identical* repeated failures mean the agent must change approach
rather than loop.
"""

from __future__ import annotations


class AgentError(Exception):
    """Base class for all agent failures."""

    #: Whether retrying the exact same call could plausibly succeed.
    retryable: bool = False
    #: Short machine-readable code surfaced to the model and the event log.
    code: str = "agent_error"

    def __init__(self, message: str, *, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "error": True,
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
        }
        if self.detail:
            payload["detail"] = self.detail
        return payload


class ConfigError(AgentError):
    code = "config_error"


class ModelError(AgentError):
    code = "model_error"
    retryable = True


class ModelUnavailableError(ModelError):
    """Model returned a permanent refusal (e.g. not entitled on this plan)."""

    code = "model_unavailable"
    retryable = False


class ToolError(AgentError):
    code = "tool_error"


class ToolNotFound(ToolError):
    code = "tool_not_found"


class ToolValidationError(ToolError):
    code = "tool_validation_error"


class ToolTimeout(ToolError):
    code = "tool_timeout"
    retryable = True


class PermissionDenied(ToolError):
    code = "permission_denied"


class ApprovalRequired(ToolError):
    """The action is legitimate but needs an explicit human authorisation."""

    code = "approval_required"


class BudgetExceeded(AgentError):
    code = "budget_exceeded"


class PersistenceError(AgentError):
    code = "persistence_error"
    retryable = True


class Unauthorized(AgentError):
    code = "unauthorized"


class RateLimited(AgentError):
    code = "rate_limited"
    retryable = True

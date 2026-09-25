"""Webhook registration helpers.

These cover the pure logic in ``scripts/set_webhook.py``. The point is that the
operational script used to point a live bot at a live service is itself tested,
rather than being trusted because it "looked right" when it ran.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "set_webhook.py"


def _load():
    spec = importlib.util.spec_from_file_location("set_webhook", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["set_webhook"] = module
    spec.loader.exec_module(module)
    return module


set_webhook = _load()


# --------------------------------------------------------------------------
# URL building
# --------------------------------------------------------------------------
def test_build_webhook_url_appends_the_route():
    assert (
        set_webhook.build_webhook_url("https://auton-agent.onrender.com")
        == "https://auton-agent.onrender.com/telegram/webhook"
    )


def test_build_webhook_url_tolerates_a_trailing_slash():
    assert (
        set_webhook.build_webhook_url("https://auton-agent.onrender.com/")
        == "https://auton-agent.onrender.com/telegram/webhook"
    )


def test_build_webhook_url_tolerates_a_missing_scheme():
    assert (
        set_webhook.build_webhook_url("auton-agent.onrender.com")
        == "https://auton-agent.onrender.com/telegram/webhook"
    )


def test_build_webhook_url_does_not_double_the_path():
    already = "https://auton-agent.onrender.com/telegram/webhook"
    assert set_webhook.build_webhook_url(already) == already


def test_build_webhook_url_rejects_an_empty_base():
    with pytest.raises(ValueError):
        set_webhook.build_webhook_url("   ")


def test_webhook_path_matches_the_route_the_service_serves():
    """The registered path and the FastAPI route must not drift apart."""
    service = (
        Path(__file__).resolve().parents[1] / "src" / "agentcore" / "service.py"
    ).read_text(encoding="utf-8")
    assert f'@app.post("{set_webhook.WEBHOOK_PATH}")' in service


# --------------------------------------------------------------------------
# secret handling
# --------------------------------------------------------------------------
def test_fingerprint_never_reveals_the_secret():
    # A synthetic 64-char value shaped like the real one. Never paste a live
    # secret into a test: this file is committed, and the assertion below would
    # then be checking that the fingerprint of a real credential stays secret.
    secret = "0" * 32 + "f" * 32
    shown = set_webhook.fingerprint(secret)
    assert secret not in shown
    assert "len=64" in shown
    assert shown.startswith("set (")


def test_fingerprint_handles_an_absent_secret():
    assert set_webhook.fingerprint(None) == "(unset)"
    assert set_webhook.fingerprint("") == "(unset)"


def test_allowed_updates_are_what_the_interface_parses():
    """The interface reads message / edited_message; nothing else is delivered."""
    assert set_webhook.ALLOWED_UPDATES == ["message", "edited_message"]


# --------------------------------------------------------------------------
# webhook info summary
# --------------------------------------------------------------------------
def test_summarise_reports_a_healthy_webhook():
    text = set_webhook.summarise_webhook_info(
        {
            "url": "https://x/telegram/webhook",
            "pending_update_count": 0,
            "allowed_updates": ["message"],
        }
    )
    assert "https://x/telegram/webhook" in text
    assert "last_error       : none" in text


def test_summarise_surfaces_a_delivery_error():
    text = set_webhook.summarise_webhook_info(
        {
            "url": "https://x/telegram/webhook",
            "pending_update_count": 4,
            "last_error_message": "Wrong response from the webhook: 503 Service Unavailable",
            "last_error_code": 503,
        }
    )
    assert "503" in text
    assert "pending_updates  : 4" in text


def test_summarise_handles_no_webhook():
    assert "(none registered)" in set_webhook.summarise_webhook_info({})

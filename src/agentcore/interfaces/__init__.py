"""Interfaces — how the outside world talks to the agent.

The agent never imports an interface; interfaces import the agent. That is what
keeps Telegram (and the HTTP API, and the CLI) replaceable.
"""

from .telegram import TelegramClient, TelegramInterface, parse_update, strip_bot_mention

__all__ = ["TelegramClient", "TelegramInterface", "parse_update", "strip_bot_mention"]

"""Structured JSON logging with mandatory secret redaction.

Log lines are newline-delimited JSON on stdout so Render's log stream stays
machine-readable. A logging filter scrubs every message and every argument
*before* it is formatted, so a secret interpolated into a log call can never
leak. Nothing here ever logs a full request body or an environment dump.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

from .redaction import redactor

_RESERVED = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "taskName", "message", "asctime",
}


class RedactionFilter(logging.Filter):
    """Scrub secrets out of the message and args of every record."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        red = redactor()
        if isinstance(record.msg, str):
            record.msg = red.scrub(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {k: red.scrub_deep(v) for k, v in record.args.items()}
            elif isinstance(record.args, tuple):
                record.args = tuple(red.scrub_deep(a) for a in record.args)
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc)
            .isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in _RESERVED or key.startswith("_"):
                continue
            payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)
        text = json.dumps(payload, default=str, ensure_ascii=False)
        return redactor().scrub(text)


def configure_logging(level: str = "INFO") -> None:
    """Install a single stdout handler; safe to call more than once."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    handler.addFilter(RedactionFilter())
    root.addHandler(handler)

    # Third-party noise control.
    for noisy in ("httpx", "httpcore", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

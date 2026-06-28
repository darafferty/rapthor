"""Logging helpers for Rapthor's Prefect execution path."""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from typing import Iterator, Optional

from rapthor.execution.prefect_context import in_prefect_run_context

PREFECT_API_LOG_HANDLER_NAME = "rapthor-prefect-api"

_handler_lock = threading.RLock()
_active_contexts = 0
_api_handler: Optional[logging.Handler] = None


class RapthorLogFilter(logging.Filter):
    """Allow only Rapthor loggers through the Prefect API bridge."""

    def filter(self, record: logging.LogRecord) -> bool:
        return (
            record.name == "rapthor"
            or record.name.startswith("rapthor:")
            or record.name.startswith("rapthor.")
        )


def _load_api_log_handler():
    from prefect.logging.handlers import APILogHandler

    return APILogHandler


def _logging_level(level: object = None) -> int:
    if level is None:
        console_level = _console_handler_level()
        return logging.INFO if console_level is None else console_level
    if isinstance(level, int):
        return level
    normalized = str(level).strip().upper()
    return logging._nameToLevel.get(normalized, logging.INFO)


def _console_handler_level() -> Optional[int]:
    for handler in logging.root.handlers:
        if handler.name == "console":
            return handler.level
    return None


def _install_api_handler(level: int) -> logging.Handler:
    handler_cls = _load_api_log_handler()
    handler = handler_cls(level=level)
    handler.set_name(PREFECT_API_LOG_HANDLER_NAME)
    handler.addFilter(RapthorLogFilter())
    handler.setFormatter(logging.Formatter("%(message)s"))

    logging.root.handlers.insert(0, handler)
    return handler


def _remove_api_handler(handler: logging.Handler) -> None:
    try:
        handler.flush()
    except Exception:
        pass
    if handler in logging.root.handlers:
        logging.root.removeHandler(handler)
    try:
        handler.close()
    except Exception:
        pass


@contextmanager
def publish_python_logs_to_prefect(level: object = None) -> Iterator[None]:
    """Forward Rapthor Python logging records to the active Prefect run.

    The handler is inserted before Rapthor's console handler so API logs keep the
    original uncoloured message even though the console handler mutates records
    while adding ANSI colours.
    """
    global _active_contexts, _api_handler

    if not in_prefect_run_context():
        yield
        return

    with _handler_lock:
        if _api_handler is None or _api_handler not in logging.root.handlers:
            _api_handler = _install_api_handler(_logging_level(level))
        else:
            _api_handler.setLevel(min(_api_handler.level, _logging_level(level)))
        _active_contexts += 1

    try:
        yield
    finally:
        with _handler_lock:
            _active_contexts -= 1
            if _active_contexts == 0 and _api_handler is not None:
                handler = _api_handler
                _api_handler = None
                _remove_api_handler(handler)

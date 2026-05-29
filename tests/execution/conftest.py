"""Pytest setup for execution-layer tests."""

import os


def pytest_sessionstart(session):  # pragma: no cover
    _ = session
    os.environ.setdefault("PREFECT_API_URL", "")
    os.environ.setdefault("PREFECT_LOGGING_RICH_CONSOLE", "false")
    os.environ.setdefault("PREFECT_LOGGING_LEVEL", "INFO")
    os.environ.setdefault("PREFECT_SERVER_EPHEMERAL_STARTUP_TIMEOUT_SECONDS", "180")

"""Pytest setup for execution-layer tests."""

import os
from contextlib import nullcontext
from unittest.mock import patch

from prefect.testing.utilities import prefect_test_harness

from rapthor.execution.config import ExecutionConfig


def pytest_sessionstart(session):  # pragma: no cover
    _ = session
    os.environ.setdefault("PREFECT_API_URL", "")
    os.environ.setdefault("PREFECT_LOGGING_RICH_CONSOLE", "false")
    os.environ.setdefault("PREFECT_LOGGING_LEVEL", "INFO")
    os.environ.setdefault("PREFECT_SERVER_EPHEMERAL_STARTUP_TIMEOUT_SECONDS", "180")


def run_flow_for_test(
    flow_fn,
    payload,
    execution_config=None,
    shell_operation_cls=None,
):
    """Run a production Prefect flow with optional fake shell loading for tests."""
    shell_loader = (
        patch("rapthor.execution.shell._load_shell_operation_cls", return_value=shell_operation_cls)
        if shell_operation_cls is not None
        else nullcontext()
    )
    config = execution_config or ExecutionConfig(task_runner="sync")
    with shell_loader, prefect_test_harness(server_startup_timeout=None):
        return flow_fn(payload, execution_config=config)

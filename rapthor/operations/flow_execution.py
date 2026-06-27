"""Execution bridge used by operation adapters."""

from typing import Any, Callable, Mapping

from rapthor.execution.config import ExecutionConfig


def run_prefect_flow(
    flow: Callable[..., Any],
    payload: object,
    parset: Mapping[str, Any],
) -> Any:
    """Run a Prefect flow with execution settings derived from an operation parset."""
    return flow(payload, execution_config=ExecutionConfig.from_parset(parset))

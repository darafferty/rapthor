"""Shared runtime helpers for operation-level Prefect flows."""

from typing import Optional

from rapthor.execution.config import ExecutionConfig
from rapthor.execution.task_runner import build_task_runner


def run_flow_with_task_runner(
    prefect_flow,
    *flow_args,
    execution_config: Optional[ExecutionConfig] = None,
    **flow_kwargs,
):
    """Run a Prefect flow with the task runner requested by execution config."""
    config = execution_config or ExecutionConfig()
    task_runner = build_task_runner(config)
    configured_flow = prefect_flow.with_options(task_runner=task_runner)
    return configured_flow(*flow_args, execution_config=config, **flow_kwargs)

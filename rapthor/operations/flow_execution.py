"""Execution bridge used by operation adapters."""

from typing import Any, Callable, Mapping

from rapthor.execution.config import ExecutionConfig
from rapthor.execution.workspace import recover_scratch_workspace
from rapthor.lib.operation import Operation


class FlowOperation(Operation):
    """Operation adapter that repairs interrupted scratch staging before setup."""

    def _prepare_working_directory(self):
        recover_scratch_workspace(self.pipeline_working_dir)
        super()._prepare_working_directory()


def run_prefect_flow(
    flow: Callable[..., Any],
    payload: object,
    parset: Mapping[str, Any],
) -> Any:
    """Run a Prefect flow with execution settings derived from an operation parset."""
    return flow(payload, execution_config=ExecutionConfig.from_parset(parset))

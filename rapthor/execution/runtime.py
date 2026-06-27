"""Runtime environment helpers for command execution."""

from dataclasses import dataclass, field
from typing import Mapping, Optional

from rapthor.execution.config import ExecutionConfig
from rapthor.execution.resources import (
    ResourceRequest,
    thread_environment,
    validate_resource_request,
)
from rapthor.execution.task_runner import build_task_runner


class UnsupportedRuntimeError(RuntimeError):
    """Raised when requested runtime behaviour is not implemented yet."""


@dataclass(frozen=True)
class RuntimeSpec:
    """Resolved runtime settings for one external command."""

    environment: Mapping[str, str] = field(default_factory=dict)
    working_directory: Optional[str] = None


def build_command_environment(
    execution_config: ExecutionConfig,
    resource_request: Optional[ResourceRequest] = None,
    extra_environment: Optional[Mapping[str, object]] = None,
) -> dict[str, str]:
    """Build a command environment from execution and resource settings."""
    environment = {}
    if resource_request is not None:
        validate_resource_request(resource_request, execution_config)
        environment.update(thread_environment(resource_request))
    for key, value in (extra_environment or {}).items():
        environment[str(key)] = str(value)
    return environment


def ensure_supported_runtime(execution_config: ExecutionConfig) -> None:
    """Fail early for runtime modes that do not have an implementation yet."""
    if execution_config.use_container:
        raise UnsupportedRuntimeError(
            "container execution is not supported by the Prefect/Dask path yet"
        )


def build_runtime_spec(
    execution_config: ExecutionConfig,
    resource_request: Optional[ResourceRequest] = None,
    extra_environment: Optional[Mapping[str, object]] = None,
    working_directory: Optional[object] = None,
) -> RuntimeSpec:
    """Resolve runtime settings for a command task."""
    ensure_supported_runtime(execution_config)
    return RuntimeSpec(
        environment=build_command_environment(
            execution_config,
            resource_request=resource_request,
            extra_environment=extra_environment,
        ),
        working_directory=str(working_directory) if working_directory is not None else None,
    )


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

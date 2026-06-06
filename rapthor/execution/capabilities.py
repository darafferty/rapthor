"""Preflight checks for the Prefect/Dask execution path."""

import shutil
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Set

from rapthor.execution.config import ExecutionConfig
from rapthor.execution.resources import ResourceRequest, collect_resource_request_issues
from rapthor.execution.slurm import collect_slurm_config_issues


@dataclass(frozen=True)
class PreflightIssue:
    """A single execution preflight issue."""

    code: str
    message: str
    option: Optional[str] = None


class PreflightError(RuntimeError):
    """Raised when execution preflight checks fail."""

    def __init__(self, issues: Sequence[PreflightIssue]):
        self.issues = list(issues)
        joined = "; ".join(issue.message for issue in self.issues)
        super().__init__(f"Execution preflight failed: {joined}")


ToolResolver = Callable[[str], Optional[str]]
SchedulerChecker = Callable[[str], object]


def _as_set(values: Optional[Iterable[str]]) -> Set[str]:
    return set(values or ())


def collect_preflight_issues(
    execution_config: ExecutionConfig,
    requested_features: Optional[Iterable[str]] = None,
    supported_features: Optional[Iterable[str]] = None,
    required_tools: Optional[Iterable[str]] = None,
    resource_requests: Optional[Iterable[ResourceRequest]] = None,
    tool_resolver: ToolResolver = shutil.which,
    scheduler_checker: Optional[SchedulerChecker] = None,
) -> List[PreflightIssue]:
    """Collect preflight issues without raising.

    This skeleton intentionally checks only runtime concerns that are useful
    before the operation flows exist. Feature names are caller-defined strings
    so later PRs can add strategy-derived feature checks without changing this
    API.
    """
    issues = []

    scheduler = execution_config.resolved_dask_scheduler()
    if execution_config.task_runner == "external_dask":
        if not scheduler:
            issues.append(
                PreflightIssue(
                    code="missing_dask_scheduler",
                    option="dask_scheduler",
                    message="external_dask requires a dask_scheduler value or DASK_SCHEDULER",
                )
            )
        elif scheduler_checker is not None:
            try:
                scheduler_checker(scheduler)
            except Exception as err:
                issues.append(
                    PreflightIssue(
                        code="dask_scheduler_unreachable",
                        option="dask_scheduler",
                        message=str(err) or f"could not connect to Dask scheduler {scheduler!r}",
                    )
                )

    if execution_config.use_container:
        issues.append(
            PreflightIssue(
                code="unsupported_container",
                option="use_container",
                message="container execution is not supported by the Prefect/Dask path yet",
            )
        )

    if supported_features is not None:
        unsupported_features = _as_set(requested_features) - _as_set(supported_features)
        for feature in sorted(unsupported_features):
            issues.append(
                PreflightIssue(
                    code="unsupported_feature",
                    message=f"feature {feature!r} is not supported by the Prefect/Dask path",
                )
            )

    for tool in required_tools or ():
        if tool_resolver(tool) is None:
            issues.append(
                PreflightIssue(
                    code="missing_tool",
                    message=f"required external tool {tool!r} was not found",
                )
            )

    for code, message in collect_resource_request_issues(resource_requests or (), execution_config):
        issues.append(PreflightIssue(code=code, message=message, option="resources"))

    for code, message in collect_slurm_config_issues(execution_config):
        issues.append(PreflightIssue(code=code, message=message, option="slurm"))

    return issues


def preflight_execution(
    execution_config: ExecutionConfig,
    requested_features: Optional[Iterable[str]] = None,
    supported_features: Optional[Iterable[str]] = None,
    required_tools: Optional[Iterable[str]] = None,
    resource_requests: Optional[Iterable[ResourceRequest]] = None,
    tool_resolver: ToolResolver = shutil.which,
    scheduler_checker: Optional[SchedulerChecker] = None,
) -> None:
    """Run execution preflight checks and raise if any fail."""
    issues = collect_preflight_issues(
        execution_config=execution_config,
        requested_features=requested_features,
        supported_features=supported_features,
        required_tools=required_tools,
        resource_requests=resource_requests,
        tool_resolver=tool_resolver,
        scheduler_checker=scheduler_checker,
    )
    if issues:
        raise PreflightError(issues)

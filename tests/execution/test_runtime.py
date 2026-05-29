import pytest

from rapthor.execution.config import ExecutionConfig
from rapthor.execution.resources import ResourceRequest
from rapthor.execution.runtime import (
    UnsupportedRuntimeError,
    build_command_environment,
    build_runtime_spec,
)


def test_build_command_environment_combines_resources_and_extra_env():
    env = build_command_environment(
        ExecutionConfig(),
        resource_request=ResourceRequest(threads=2),
        extra_environment={"NUMEXPR_MAX_THREADS": 1},
    )

    assert env == {
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "NUMEXPR_MAX_THREADS": "1",
    }


def test_build_runtime_spec_records_working_directory():
    runtime = build_runtime_spec(
        ExecutionConfig(),
        resource_request=ResourceRequest(threads=2),
        working_directory="/tmp/task",
    )

    assert runtime.working_directory == "/tmp/task"
    assert runtime.environment["OPENBLAS_NUM_THREADS"] == "2"


def test_container_runtime_is_explicitly_unsupported_for_now():
    with pytest.raises(UnsupportedRuntimeError, match="container execution"):
        build_runtime_spec(ExecutionConfig(use_container=True))

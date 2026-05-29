import pytest

from rapthor.execution.config import ExecutionConfig
from rapthor.execution.resources import ResourceRequest, thread_environment


def test_resource_request_from_execution_config_uses_threads_and_memory():
    request = ResourceRequest.from_execution_config(
        ExecutionConfig(cpus_per_task=8, mem_per_node_gb=64),
        name="wsclean",
    )

    assert request.name == "wsclean"
    assert request.threads == 8
    assert request.memory_gb == 64
    assert request.exclusive is False


def test_resource_request_mpi_is_exclusive():
    request = ResourceRequest.from_execution_config(ExecutionConfig(), use_mpi=True)

    assert request.use_mpi is True
    assert request.exclusive is True


def test_resource_request_rejects_invalid_threads():
    with pytest.raises(ValueError, match="threads"):
        ResourceRequest(threads=0)


def test_thread_environment_sets_common_thread_variables():
    assert thread_environment(ResourceRequest(threads=3)) == {
        "OMP_NUM_THREADS": "3",
        "OPENBLAS_NUM_THREADS": "3",
    }

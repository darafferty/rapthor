import pytest

from rapthor.execution.config import ExecutionConfig
from rapthor.execution.resources import (
    ResourceRequest,
    collect_resource_request_issues,
    thread_environment,
    validate_resource_request,
)


def test_resource_request_rejects_invalid_threads():
    with pytest.raises(ValueError, match="threads"):
        ResourceRequest(threads=0)


def test_validate_resource_request_rejects_thread_oversubscription():
    request = ResourceRequest(name="wsclean", threads=5)

    with pytest.raises(ValueError, match="requests 5 threads"):
        validate_resource_request(request, ExecutionConfig(cpus_per_task=4))


def test_validate_resource_request_rejects_memory_oversubscription():
    request = ResourceRequest(name="wsclean", memory_gb=65)

    with pytest.raises(ValueError, match="requests 65 GB"):
        validate_resource_request(request, ExecutionConfig(mem_per_node_gb=64))


def test_validate_resource_request_rejects_local_dask_process_oversubscription():
    request = ResourceRequest(name="dp3", processes=3)

    with pytest.raises(ValueError, match="requests 3 concurrent processes"):
        validate_resource_request(
            request,
            ExecutionConfig(task_runner="local_dask", max_nodes=2),
        )


def test_validate_resource_request_allows_external_dask_processes():
    request = ResourceRequest(name="dp3", processes=3)

    assert (
        validate_resource_request(
            request,
            ExecutionConfig(task_runner="external_dask", max_nodes=1),
        )
        == request
    )


def test_validate_resource_request_rejects_slurm_process_oversubscription():
    request = ResourceRequest(name="dp3", processes=3)

    with pytest.raises(ValueError, match="Slurm allocation is limited to 2 nodes"):
        validate_resource_request(
            request,
            ExecutionConfig(
                task_runner="external_dask",
                batch_system="slurm",
                max_nodes=2,
            ),
        )


def test_validate_resource_request_allows_unlimited_slurm_nodes():
    request = ResourceRequest(name="dp3", processes=3)

    assert (
        validate_resource_request(
            request,
            ExecutionConfig(
                task_runner="external_dask",
                batch_system="slurm",
                max_nodes=0,
            ),
        )
        == request
    )


def test_validate_resource_request_rejects_nonexclusive_mpi_request():
    request = ResourceRequest(name="wsclean-mpi", use_mpi=True, exclusive=False)

    with pytest.raises(ValueError, match="must be marked exclusive"):
        validate_resource_request(request, ExecutionConfig(max_nodes=2))


def test_validate_resource_request_rejects_mpi_process_oversubscription():
    request = ResourceRequest(
        name="wsclean-mpi",
        processes=3,
        use_mpi=True,
        exclusive=True,
    )

    with pytest.raises(ValueError, match="requests 3 MPI processes"):
        validate_resource_request(request, ExecutionConfig(max_nodes=2))


def test_validate_resource_request_allows_mpi_when_node_count_is_unlimited():
    request = ResourceRequest(
        name="wsclean-mpi",
        processes=3,
        use_mpi=True,
        exclusive=True,
    )

    assert validate_resource_request(request, ExecutionConfig(max_nodes=0)) == request


def test_collect_resource_request_issues_preserves_issue_codes():
    issues = collect_resource_request_issues(
        [
            ResourceRequest(name="dp3", threads=8),
            ResourceRequest(
                name="wsclean-mpi",
                processes=2,
                use_mpi=True,
                exclusive=False,
            ),
        ],
        ExecutionConfig(cpus_per_task=4, max_nodes=1),
    )

    assert [code for code, _ in issues] == [
        "resource_threads_oversubscribed",
        "mpi_not_exclusive",
        "mpi_processes_oversubscribed",
    ]


def test_thread_environment_sets_common_thread_variables():
    assert thread_environment(ResourceRequest(threads=3)) == {
        "OMP_NUM_THREADS": "3",
        "OPENBLAS_NUM_THREADS": "3",
    }

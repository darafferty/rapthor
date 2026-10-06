import pytest

from rapthor.execution.config import ExecutionConfig
from rapthor.execution.resources import (
    ResourceRequest,
    collect_resource_request_issues,
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
            ExecutionConfig(task_runner="local_dask", max_nodes=1, local_dask_workers=2),
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


@pytest.mark.parametrize(
    "command",
    [
        ["DP3", "numthreads=9", "steps=[]"],
        ["wsclean", "-j", "9", "data.ms"],
        ["mpirun", "-np", "2", "wsclean-mp", "-j", "9", "data.ms"],
        ["python3", "-m", "rapthor.execution.image.skymodel_filter_cli", "--ncores=9"],
    ],
)
def test_emitted_commands_cannot_exceed_cpu_budget(command, monkeypatch):
    from rapthor.execution.resources import validate_command_threads

    monkeypatch.setattr("rapthor.execution.resources.available_cpu_count", lambda: 32)
    with pytest.raises(ValueError, match="requests 9 threads.*budget is 8"):
        validate_command_threads(command, ExecutionConfig(cpus_per_task=8))


def test_worker_affinity_is_checked_again_before_command(monkeypatch):
    from rapthor.execution.resources import validate_command_threads

    monkeypatch.setattr("rapthor.execution.resources.available_cpu_count", lambda: 8)
    with pytest.raises(ValueError, match="budget is 4"):
        validate_command_threads(
            ["DP3", "numthreads=8"], ExecutionConfig(cpus_per_task=8, workers_per_node=2)
        )


def test_command_arguments_are_not_treated_as_executables():
    from rapthor.execution.resources import validate_command_threads

    assert validate_command_threads(["cp", "wsclean", "backup"], ExecutionConfig()) == 1


@pytest.mark.parametrize("command", [["DP3", "steps=[]"], ["wsclean", "data.ms"]])
def test_tools_must_declare_their_thread_count(command):
    from rapthor.execution.resources import validate_command_threads

    with pytest.raises(ValueError, match="must declare their thread count"):
        validate_command_threads(command, ExecutionConfig())

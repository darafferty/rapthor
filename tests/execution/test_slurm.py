import subprocess
from pathlib import Path

from rapthor.execution.config import ExecutionConfig
from rapthor.execution.slurm import collect_slurm_config_issues, slurm_cluster_spec

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_slurm_cluster_spec_uses_slurm_allocation_environment():
    config = ExecutionConfig(
        task_runner="external_dask",
        batch_system="slurm",
        max_nodes=1,
        cpus_per_task=0,
        mem_per_node_gb=256,
    )
    environ = {
        "SLURM_NNODES": "4",
        "SLURM_NTASKS": "4",
        "SLURM_CPUS_PER_TASK": "32",
    }

    spec = slurm_cluster_spec(config, environ=environ)

    assert spec.node_count == 4
    assert spec.task_count == 4
    assert spec.cpus_per_task == 32
    assert spec.worker_count == 4
    assert spec.threads_per_worker == 1
    assert spec.command_threads_per_task == 32
    assert spec.memory_per_node_gb == 256


def test_slurm_cluster_spec_uses_execution_config_without_slurm_environment():
    spec = slurm_cluster_spec(
        ExecutionConfig(
            task_runner="external_dask",
            batch_system="slurm",
            max_nodes=2,
            cpus_per_task=8,
        )
    )

    assert spec.node_count == 2
    assert spec.task_count == 2
    assert spec.worker_count == 2
    assert spec.threads_per_worker == 1
    assert spec.command_threads_per_task == 8


def test_slurm_cluster_spec_keeps_prefect_task_execution_single_threaded():
    spec = slurm_cluster_spec(
        ExecutionConfig(
            task_runner="external_dask",
            batch_system="slurm",
            max_nodes=2,
            cpus_per_task=8,
        )
    )

    assert spec.threads_per_worker == 1
    assert spec.command_threads_per_task == 8


def test_collect_slurm_config_issues_reports_too_few_tasks():
    issues = collect_slurm_config_issues(
        ExecutionConfig(
            task_runner="external_dask",
            dask_scheduler="tcp://scheduler:8786",
            batch_system="slurm",
            max_nodes=0,
        ),
        environ={"SLURM_NNODES": "4", "SLURM_NTASKS": "2"},
    )

    assert issues == [
        (
            "slurm_tasks_less_than_nodes",
            "Slurm allocation exposes 2 tasks for 4 nodes; Rapthor expects at "
            "least one task per node",
        )
    ]


def test_collect_slurm_config_issues_rejects_invalid_environment():
    issues = collect_slurm_config_issues(
        ExecutionConfig(task_runner="external_dask", batch_system="slurm"),
        environ={"SLURM_NNODES": "many"},
    )

    assert issues == [("invalid_slurm_allocation", "SLURM_NNODES must be an integer")]


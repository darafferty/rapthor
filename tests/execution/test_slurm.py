import subprocess
from pathlib import Path

from rapthor.execution.config import ExecutionConfig
from rapthor.execution.slurm import collect_slurm_config_issues, slurm_cluster_spec

REPO_ROOT = Path(__file__).resolve().parents[2]
SLURM_SCRIPTS = (
    REPO_ROOT / "scripts/prod/run-rapthor-slurm.sbatch",
    REPO_ROOT / "scripts/dev/run-rapthor-slurm-dev.sbatch",
)


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
    assert spec.threads_per_worker == 31
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
    assert spec.threads_per_worker == 7


def test_slurm_cluster_spec_can_use_all_worker_threads():
    spec = slurm_cluster_spec(
        ExecutionConfig(
            task_runner="external_dask",
            batch_system="slurm",
            max_nodes=2,
            cpus_per_task=8,
        ),
        reserve_scheduler_cpu=False,
    )

    assert spec.threads_per_worker == 8


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


def test_slurm_scripts_are_bash_syntax_valid():
    for script in SLURM_SCRIPTS:
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_slurm_scripts_export_dask_scheduler_and_start_one_worker_per_node():
    for script in SLURM_SCRIPTS:
        content = script.read_text()

        assert "export DASK_SCHEDULER=" in content
        assert "dask scheduler" in content
        assert "dask worker" in content
        assert '--nodes="$NODE_COUNT" --ntasks="$NODE_COUNT" --ntasks-per-node=1' in content

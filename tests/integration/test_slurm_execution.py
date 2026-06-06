import os
import shutil

import pytest

from rapthor.execution.config import DASK_SCHEDULER_ENV, ExecutionConfig
from rapthor.execution.slurm import slurm_cluster_spec
from rapthor.execution.task_runner import check_dask_scheduler

RUN_SLURM_INTEGRATION_ENV = "RAPTHOR_RUN_SLURM_INTEGRATION"


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get(RUN_SLURM_INTEGRATION_ENV) != "1",
    reason=f"set {RUN_SLURM_INTEGRATION_ENV}=1 inside a Slurm allocation to run",
)
def test_slurm_external_dask_environment_is_usable():
    """Validate the selected Slurm mode in a real target allocation."""
    assert shutil.which("srun") is not None
    assert os.environ.get(DASK_SCHEDULER_ENV)

    config = ExecutionConfig(
        task_runner="external_dask",
        batch_system="slurm",
        max_nodes=0,
        cpus_per_task=0,
    )
    spec = slurm_cluster_spec(config, environ=os.environ)

    assert spec.node_count >= 1
    assert check_dask_scheduler(config.resolved_dask_scheduler(), timeout="30s") >= spec.worker_count

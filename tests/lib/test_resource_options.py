import pytest

from rapthor.lib.resource_options import resolve_cluster_resources


def resolve(**options):
    return resolve_cluster_resources(options, available_cpus=32, environ={})


def test_automatic_threads_share_local_cpus():
    settings = resolve(local_dask_workers=4)
    assert settings["workers_per_node"] == 4
    assert settings["cpus_per_task"] == 8
    assert (
        settings["max_threads"]
        == settings["dp3_max_threads"]
        == settings["wsclean_max_threads"]
        == 8
    )
    assert settings["filter_skymodel_ncores"] == 8


@pytest.mark.parametrize(
    "option", ["max_threads", "dp3_max_threads", "wsclean_max_threads", "filter_skymodel_ncores"]
)
def test_explicit_thread_limit_cannot_exceed_worker_budget(option):
    with pytest.raises(ValueError, match=f"{option}=9 exceeds cpus_per_task=8"):
        resolve(local_dask_workers=4, **{option: 9})


def test_independent_tool_limits_fit_allocation():
    settings = resolve(cpus_per_task=32, max_threads=8, dp3_max_threads=4, wsclean_max_threads=24)
    assert settings["dp3_max_threads"] == 4
    assert settings["wsclean_max_threads"] == 24
    assert settings["deconvolution_threads"] == 9
    assert settings["parallel_gridding_tasks"] == 3


def test_explicit_worker_allocations_cannot_oversell_host():
    with pytest.raises(ValueError, match="available CPUs"):
        resolve(local_dask_workers=4, cpus_per_task=16)


def test_zero_tool_limits_inherit_task_default_for_mpi_too():
    settings = resolve(cpus_per_task=32, max_threads=8, wsclean_max_threads=0)
    assert settings["wsclean_max_threads"] == 8


def test_auto_source_filtering_retains_conservative_ceiling():
    assert resolve()["filter_skymodel_ncores"] == 15


def test_slurm_allocation_overrides_controller_cpu_count():
    options = {"batch_system": "slurm", "prefect_task_runner": "external_dask"}
    settings = resolve_cluster_resources(
        options, available_cpus=96, environ={"SLURM_CPUS_PER_TASK": "16"}
    )
    assert settings["cpus_per_task"] == settings["max_threads"] == 16
    with pytest.raises(ValueError, match="exceeds SLURM_CPUS_PER_TASK"):
        resolve_cluster_resources(
            {**options, "cpus_per_task": 32},
            available_cpus=96,
            environ={"SLURM_CPUS_PER_TASK": "16"},
        )


def test_local_bootstrap_preserves_resolved_shares():
    original = resolve(local_dask_workers=4)
    connected = resolve_cluster_resources(
        {**original, "prefect_task_runner": "external_dask"}, available_cpus=32, environ={}
    )
    assert connected == {**original, "prefect_task_runner": "external_dask"}


def test_slurm_node_limit_uses_and_respects_allocation():
    settings = resolve_cluster_resources(
        {"batch_system": "slurm"}, available_cpus=32, environ={"SLURM_NNODES": "2"}
    )
    assert settings["max_nodes"] == 2
    with pytest.raises(ValueError, match="max_nodes exceeds"):
        resolve_cluster_resources(
            {"batch_system": "slurm", "max_nodes": 4},
            available_cpus=32,
            environ={"SLURM_NNODES": "2"},
        )

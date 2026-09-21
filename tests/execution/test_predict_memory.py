"""Prediction post-processing must respect the memory allocation of its worker."""

from types import SimpleNamespace

import distributed
import distributed.system
import psutil
import pytest

from rapthor.execution.config import ExecutionConfig
from rapthor.execution.predict import flow as predict_flow
from rapthor.lib.records import directory_record


@pytest.fixture
def memory_environment(monkeypatch):
    monkeypatch.setattr(distributed.system, "memory_limit", lambda: 8_000_000_000)
    monkeypatch.setattr(
        psutil,
        "Process",
        lambda: SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=100_000_000)),
    )

    def no_worker():
        raise ValueError("No worker found")

    monkeypatch.setattr(distributed, "get_worker", no_worker)


@pytest.mark.parametrize("configured_gb, expected", [(0, 7_900_000_000), (2, 1_900_000_000)])
def test_postprocess_budget_respects_system_and_configured_limits(
    memory_environment, configured_gb, expected
):
    config = ExecutionConfig(task_runner="sync", mem_per_node_gb=configured_gb)
    assert predict_flow._postprocess_memory_budget(config) == expected


@pytest.mark.parametrize(
    "worker_limit, threads, expected",
    [
        (1_000_000_000, 1, 900_000_000),
        (1_000_000_000, 2, 450_000_000),
        (0, 1, 1_900_000_000),
    ],
)
def test_postprocess_budget_respects_worker_limit_and_concurrent_threads(
    monkeypatch, memory_environment, worker_limit, threads, expected
):
    worker = SimpleNamespace(
        memory_manager=SimpleNamespace(memory_limit=worker_limit),
        state=SimpleNamespace(nthreads=threads),
    )
    monkeypatch.setattr(distributed, "get_worker", lambda: worker)
    config = ExecutionConfig(mem_per_node_gb=2)
    assert predict_flow._postprocess_memory_budget(config) == expected


def test_postprocess_budget_rejects_exhausted_allocation(monkeypatch, memory_environment):
    monkeypatch.setattr(distributed.system, "memory_limit", lambda: 50_000_000)
    with pytest.raises(MemoryError, match="predict post-processing"):
        predict_flow._postprocess_memory_budget(ExecutionConfig())


@pytest.mark.parametrize("mode", ["di", "dd"])
def test_postprocess_task_passes_budget_to_data_helpers(
    tmp_path, monkeypatch, memory_environment, mode
):
    budgets = []

    def process_models(*args, **kwargs):
        budgets.append(kwargs["memory_budget_bytes"])
        suffix = "_di.ms" if mode == "di" else ""
        (tmp_path / f"input.ms.sector_1{suffix}").mkdir()

    monkeypatch.setattr(predict_flow, "add_sector_models", process_models)
    monkeypatch.setattr(predict_flow, "subtract_sector_models", process_models)
    task = {
        "msobs": "input.ms",
        "data_colname": "DATA",
        "obs_starttime": "start",
        "infix": "",
        "solint_sec": 60,
        "solint_hz": 0,
        "min_uv_lambda": 80,
        "max_uv_lambda": 1e6,
        "nr_outliers": 1,
        "peel_outliers": True,
        "nr_bright": 0,
        "peel_bright": False,
        "reweight": False,
    }
    predict_flow.predict_postprocess_task.fn(
        mode,
        task,
        [directory_record("model.ms")],
        str(tmp_path),
        execution_config=ExecutionConfig(task_runner="sync", mem_per_node_gb=2),
    )
    assert budgets == [1_900_000_000]

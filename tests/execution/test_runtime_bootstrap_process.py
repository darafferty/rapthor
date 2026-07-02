import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SMOKE_SCRIPT = r"""
import json
import logging
import os
import sys
from contextlib import ExitStack
from pathlib import Path

from dask.distributed import Client, LocalCluster
from prefect import flow, task
from prefect.settings import PREFECT_API_URL
from prefect.testing.utilities import prefect_test_harness

from rapthor.execution.config import ExecutionConfig
from rapthor.execution.runtime_bootstrap import bootstrapped_runtime
from rapthor.execution.task_runner import run_flow_with_task_runner

for logger_name in ("prefect", "httpcore", "httpx", "websockets"):
    logging.getLogger(logger_name).setLevel(logging.WARNING)


@task
def add_one(value):
    return value + 1


@flow(name="runtime-bootstrap-smoke")
def runtime_smoke_flow(execution_config=None):
    return {
        "computed": add_one(1),
        "prefect_api_url": os.environ.get("PREFECT_API_URL"),
        "prefect_home": os.environ.get("PREFECT_HOME"),
        "analytics": os.environ.get("PREFECT_SERVER_ANALYTICS_ENABLED"),
    }


def run_smoke(scenario, prefect_home):
    use_prefect = scenario.startswith("prefect")
    use_external_dask = "external-dask" in scenario

    with ExitStack() as stack:
        if use_prefect:
            stack.enter_context(prefect_test_harness(server_startup_timeout=None))
            prefect_api_url = PREFECT_API_URL.value()
        else:
            prefect_api_url = None

        if use_external_dask:
            cluster = LocalCluster(
                n_workers=1,
                threads_per_worker=1,
                processes=False,
                dashboard_address=None,
            )
            client = Client(cluster)
            client.wait_for_workers(1, timeout="60s")
            stack.callback(client.close)
            stack.callback(cluster.close)
            task_runner = "external_dask"
            dask_scheduler = cluster.scheduler_address
        else:
            task_runner = "local_dask"
            dask_scheduler = None

        config = ExecutionConfig(
            task_runner=task_runner,
            prefect_api_mode="auto",
            prefect_api_url=prefect_api_url,
            dask_scheduler=dask_scheduler,
            dask_dashboard_address=None,
            local_dask_workers=1,
            cpus_per_task=1,
        )

        with bootstrapped_runtime(config) as plan:
            flow_result = run_flow_with_task_runner(
                runtime_smoke_flow,
                execution_config=plan.execution_config,
            )
            return {
                "scenario": scenario,
                "input_prefect_api_url": prefect_api_url,
                "input_dask_scheduler": dask_scheduler,
                "plan_prefect_api_url": plan.prefect_api_url,
                "plan_dask_scheduler": plan.dask_scheduler,
                "effective_task_runner": plan.execution_config.task_runner,
                "effective_dask_scheduler": plan.execution_config.dask_scheduler,
                "dask_worker_count": plan.dask_worker_count,
                "flow_result": flow_result,
                "outer_prefect_home": prefect_home,
            }


if __name__ == "__main__":
    result = run_smoke(sys.argv[1], Path(sys.argv[2]).as_posix())
    print("RESULT " + json.dumps(result, sort_keys=True))
"""


@pytest.mark.prefect
@pytest.mark.parametrize(
    "scenario",
    [
        "no-prefect-no-dask",
        "prefect-no-dask",
        "no-prefect-external-dask",
        "prefect-external-dask",
    ],
)
def test_runtime_bootstrap_process_matrix(tmp_path, scenario):
    pytest.importorskip("dask.distributed")
    pytest.importorskip("prefect")
    pytest.importorskip("prefect_dask")

    script = tmp_path / "runtime_bootstrap_smoke.py"
    script.write_text(SMOKE_SCRIPT)
    prefect_home = tmp_path / "prefect-home"
    prefect_home.mkdir()
    env = _isolated_runtime_environment(prefect_home)

    completed = subprocess.run(
        [sys.executable, script.as_posix(), scenario, prefect_home.as_posix()],
        cwd=Path(__file__).parents[2],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=180,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = _extract_smoke_result(completed.stdout)

    assert result["flow_result"]["computed"] == 2
    assert result["flow_result"]["analytics"] == "false"
    assert result["effective_task_runner"] == "external_dask"
    assert result["dask_worker_count"] == 1

    if scenario.startswith("prefect"):
        assert result["plan_prefect_api_url"] == result["input_prefect_api_url"]
        assert result["flow_result"]["prefect_api_url"] == result["input_prefect_api_url"]
    else:
        assert result["plan_prefect_api_url"] is None
        assert result["flow_result"]["prefect_api_url"] == ""
        assert result["flow_result"]["prefect_home"] != result["outer_prefect_home"]

    if "external-dask" in scenario:
        assert result["plan_dask_scheduler"] == result["input_dask_scheduler"]
        assert result["effective_dask_scheduler"] == result["input_dask_scheduler"]
    else:
        assert result["input_dask_scheduler"] is None
        assert result["plan_dask_scheduler"].startswith("tcp://")
        assert result["effective_dask_scheduler"] == result["plan_dask_scheduler"]


def _isolated_runtime_environment(prefect_home):
    env = os.environ.copy()
    env.pop("PREFECT_API_URL", None)
    env.pop("DASK_SCHEDULER", None)
    env["PREFECT_HOME"] = prefect_home.as_posix()
    env["PREFECT_LOGGING_LEVEL"] = "WARNING"
    env["PREFECT_LOGGING_TO_API_ENABLED"] = "false"
    env["PREFECT_SERVER_ANALYTICS_ENABLED"] = "false"
    return env


def _extract_smoke_result(stdout):
    for line in reversed(stdout.splitlines()):
        if line.startswith("RESULT "):
            return json.loads(line.removeprefix("RESULT "))
    raise AssertionError(f"smoke process did not print a RESULT line:\n{stdout}")

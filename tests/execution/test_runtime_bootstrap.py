import logging
from pathlib import Path

import pytest

from rapthor.execution.config import PREFECT_API_URL_ENV, ExecutionConfig
from rapthor.execution.runtime_bootstrap import (
    PREFECT_HOME_ENV,
    PREFECT_SERVER_ANALYTICS_ENABLED_ENV,
    RuntimeBootstrapError,
    bootstrapped_runtime,
    prefect_api_health_url,
    prefect_dashboard_url,
    preflight_runtime,
    resolve_prefect_api,
)


class FakeLocalDaskCluster:
    def __init__(
        self,
        scheduler_address="tcp://127.0.0.1:8786",
        dashboard_url="http://127.0.0.1:8787/status",
        worker_count=2,
    ):
        self.scheduler_address = scheduler_address
        self.dashboard_url = dashboard_url
        self.worker_count = worker_count
        self.closed = False

    def close(self):
        self.closed = True


def test_prefect_api_health_url():
    assert (
        prefect_api_health_url("http://127.0.0.1:4200/api/") == "http://127.0.0.1:4200/api/health"
    )


def test_prefect_dashboard_url_from_local_api_url():
    assert prefect_dashboard_url("http://127.0.0.1:4200/api") == "http://127.0.0.1:4200"


def test_prefect_dashboard_url_preserves_server_prefix():
    assert prefect_dashboard_url("https://prefect.example/rapthor/api") == (
        "https://prefect.example/rapthor"
    )


def test_auto_prefect_api_without_url_allows_ephemeral():
    def fail_if_checked(api_url):
        raise AssertionError(f"unexpected Prefect health check for {api_url}")

    api_url = resolve_prefect_api(
        ExecutionConfig(prefect_api_mode="auto"),
        checker=fail_if_checked,
    )

    assert api_url is None


def test_external_prefect_api_requires_url():
    with pytest.raises(RuntimeBootstrapError, match="requires cluster.prefect_api_url"):
        resolve_prefect_api(ExecutionConfig(prefect_api_mode="external"))


def test_auto_prefect_api_with_url_checks_and_returns_it():
    calls = []

    api_url = resolve_prefect_api(
        ExecutionConfig(
            prefect_api_mode="auto",
            prefect_api_url="http://prefect.example:4200/api",
        ),
        checker=calls.append,
    )

    assert api_url == "http://prefect.example:4200/api"
    assert calls == ["http://prefect.example:4200/api"]


def test_ephemeral_prefect_api_ignores_configured_url():
    def fail_if_checked(api_url):
        raise AssertionError(f"unexpected Prefect health check for {api_url}")

    api_url = resolve_prefect_api(
        ExecutionConfig(
            prefect_api_mode="ephemeral",
            prefect_api_url="http://prefect.example:4200/api",
        ),
        checker=fail_if_checked,
    )

    assert api_url is None


def test_preflight_external_dask_checks_worker_count(caplog):
    caplog.set_level(logging.INFO, logger="rapthor:runtime")
    calls = []

    plan = preflight_runtime(
        ExecutionConfig(
            prefect_api_url="http://prefect.example:4200/api",
            task_runner="external_dask",
            dask_scheduler="tcp://scheduler:8786",
        ),
        prefect_api_checker=lambda api_url: None,
        dask_scheduler_checker=lambda scheduler: calls.append(scheduler) or 3,
    )

    assert plan.prefect_dashboard_url == "http://prefect.example:4200"
    assert plan.dask_scheduler == "tcp://scheduler:8786"
    assert plan.dask_worker_count == 3
    assert calls == ["tcp://scheduler:8786"]
    assert "Prefect dashboard: http://prefect.example:4200" in caplog.text


def test_preflight_external_dask_requires_scheduler():
    with pytest.raises(RuntimeBootstrapError, match="requires cluster.dask_scheduler"):
        preflight_runtime(
            ExecutionConfig(task_runner="external_dask"),
            dask_scheduler_checker=lambda scheduler: 1,
        )


def test_preflight_local_dask_reports_local_settings_without_scheduler_check(caplog):
    caplog.set_level(logging.INFO, logger="rapthor:runtime")

    def fail_if_checked(scheduler):
        raise AssertionError(f"unexpected Dask scheduler check for {scheduler}")

    plan = preflight_runtime(
        ExecutionConfig(
            task_runner="local_dask", max_nodes=1, local_dask_workers=2, cpus_per_task=4
        ),
        dask_scheduler_checker=fail_if_checked,
    )

    assert plan.dask_scheduler is None
    assert plan.dask_worker_count is None
    assert (
        "Using local Dask with 2 single-threaded worker(s); "
        "external commands may use up to 4 thread(s) per task."
    ) in caplog.text


def test_bootstrapped_runtime_sets_and_restores_external_prefect_api():
    environ = {PREFECT_API_URL_ENV: "http://original:4200/api"}
    config = ExecutionConfig(
        task_runner="sync",
        prefect_api_mode="external",
        prefect_api_url="http://configured:4200/api",
    )

    with bootstrapped_runtime(
        config,
        environ=environ,
        prefect_api_checker=lambda api_url: None,
    ) as plan:
        assert plan.prefect_api_url == "http://configured:4200/api"
        assert plan.prefect_dashboard_url == "http://configured:4200"
        assert environ[PREFECT_API_URL_ENV] == "http://configured:4200/api"
        assert environ[PREFECT_SERVER_ANALYTICS_ENABLED_ENV] == "false"

    assert environ[PREFECT_API_URL_ENV] == "http://original:4200/api"
    assert PREFECT_SERVER_ANALYTICS_ENABLED_ENV not in environ


def test_bootstrapped_runtime_uses_prefect_ephemeral_server_when_no_api_is_configured(caplog):
    caplog.set_level(logging.INFO, logger="rapthor:runtime")
    environ = {}

    with bootstrapped_runtime(
        ExecutionConfig(prefect_api_mode="auto", task_runner="sync"),
        environ=environ,
        prefect_api_checker=lambda api_url: None,
    ) as plan:
        assert plan.prefect_api_url is None
        assert plan.prefect_dashboard_url is None
        assert environ[PREFECT_API_URL_ENV] == ""
        assert environ[PREFECT_SERVER_ANALYTICS_ENABLED_ENV] == "false"
        temporary_prefect_home = Path(environ[PREFECT_HOME_ENV])
        assert temporary_prefect_home.exists()

    assert PREFECT_API_URL_ENV not in environ
    assert PREFECT_HOME_ENV not in environ
    assert not temporary_prefect_home.exists()
    assert "Ignoring any Prefect profile API URL for this run." in caplog.text
    assert "Using isolated temporary Prefect home for this run." in caplog.text


def test_bootstrapped_runtime_ignores_external_prefect_api_for_ephemeral_mode(caplog):
    caplog.set_level(logging.INFO, logger="rapthor:runtime")
    environ = {
        PREFECT_API_URL_ENV: "http://original:4200/api",
        PREFECT_HOME_ENV: "/original/prefect-home",
        PREFECT_SERVER_ANALYTICS_ENABLED_ENV: "true",
    }

    with bootstrapped_runtime(
        ExecutionConfig(prefect_api_mode="ephemeral", task_runner="sync"),
        environ=environ,
    ) as plan:
        assert plan.prefect_api_url is None
        assert plan.prefect_dashboard_url is None
        assert environ[PREFECT_API_URL_ENV] == ""
        assert environ[PREFECT_SERVER_ANALYTICS_ENABLED_ENV] == "false"
        temporary_prefect_home = Path(environ[PREFECT_HOME_ENV])
        assert temporary_prefect_home.exists()

    assert environ[PREFECT_API_URL_ENV] == "http://original:4200/api"
    assert environ[PREFECT_HOME_ENV] == "/original/prefect-home"
    assert environ[PREFECT_SERVER_ANALYTICS_ENABLED_ENV] == "true"
    assert not temporary_prefect_home.exists()
    assert "Ignoring any Prefect profile API URL for this run." in caplog.text
    assert "Using isolated temporary Prefect home for this run." in caplog.text


def test_bootstrapped_runtime_reuses_one_local_dask_scheduler_for_run(caplog):
    caplog.set_level(logging.INFO, logger="rapthor:runtime")
    environ = {}
    clusters = []

    def start_cluster(execution_config):
        clusters.append(FakeLocalDaskCluster(worker_count=execution_config.local_dask_worker_count))
        return clusters[-1]

    with bootstrapped_runtime(
        ExecutionConfig(
            prefect_api_mode="auto",
            task_runner="local_dask",
            local_dask_workers=2,
            cpus_per_task=4,
        ),
        environ=environ,
        prefect_api_checker=lambda api_url: None,
        local_dask_cluster_starter=start_cluster,
    ) as plan:
        assert plan.dask_scheduler == "tcp://127.0.0.1:8786"
        assert plan.dask_worker_count == 2
        assert plan.dask_dashboard_url == "http://127.0.0.1:8787/status"
        assert plan.execution_config.task_runner == "external_dask"
        assert plan.execution_config.dask_scheduler == "tcp://127.0.0.1:8786"
        assert clusters[0].closed is False

    assert clusters[0].closed is True
    assert "Started local Dask scheduler tcp://127.0.0.1:8786 with 2 worker(s)." in caplog.text
    assert "Dask dashboard: http://127.0.0.1:8787/status" in caplog.text


@pytest.mark.parametrize(
    (
        "prefect_api_url",
        "task_runner",
        "dask_scheduler",
        "expected_prefect_api",
        "expected_dask_scheduler",
        "expected_effective_task_runner",
        "expected_local_cluster_count",
        "expected_dask_checks",
    ),
    [
        pytest.param(
            None,
            "local_dask",
            None,
            None,
            "tcp://local-dask:8786",
            "external_dask",
            1,
            [],
            id="no-prefect-no-dask",
        ),
        pytest.param(
            "http://prefect.example:4200/api",
            "local_dask",
            None,
            "http://prefect.example:4200/api",
            "tcp://local-dask:8786",
            "external_dask",
            1,
            [],
            id="existing-prefect-no-dask",
        ),
        pytest.param(
            None,
            "external_dask",
            "tcp://external-dask:8786",
            None,
            "tcp://external-dask:8786",
            "external_dask",
            0,
            ["tcp://external-dask:8786"],
            id="no-prefect-existing-dask",
        ),
        pytest.param(
            "http://prefect.example:4200/api",
            "external_dask",
            "tcp://external-dask:8786",
            "http://prefect.example:4200/api",
            "tcp://external-dask:8786",
            "external_dask",
            0,
            ["tcp://external-dask:8786"],
            id="existing-prefect-existing-dask",
        ),
    ],
)
def test_bootstrapped_runtime_launch_matrix(
    prefect_api_url,
    task_runner,
    dask_scheduler,
    expected_prefect_api,
    expected_dask_scheduler,
    expected_effective_task_runner,
    expected_local_cluster_count,
    expected_dask_checks,
):
    environ = {}
    prefect_checks = []
    dask_checks = []
    clusters = []

    def check_prefect(api_url):
        prefect_checks.append(api_url)

    def check_dask(scheduler):
        dask_checks.append(scheduler)
        return 4

    def start_cluster(execution_config):
        cluster = FakeLocalDaskCluster(
            scheduler_address="tcp://local-dask:8786",
            worker_count=execution_config.local_dask_worker_count,
        )
        clusters.append(cluster)
        return cluster

    with bootstrapped_runtime(
        ExecutionConfig(
            prefect_api_mode="auto",
            prefect_api_url=prefect_api_url,
            task_runner=task_runner,
            dask_scheduler=dask_scheduler,
            local_dask_workers=2,
            cpus_per_task=4,
        ),
        environ=environ,
        prefect_api_checker=check_prefect,
        dask_scheduler_checker=check_dask,
        local_dask_cluster_starter=start_cluster,
    ) as plan:
        assert plan.prefect_api_url == expected_prefect_api
        assert plan.dask_scheduler == expected_dask_scheduler
        assert plan.execution_config.task_runner == expected_effective_task_runner
        assert plan.execution_config.dask_scheduler == expected_dask_scheduler
        assert environ[PREFECT_SERVER_ANALYTICS_ENABLED_ENV] == "false"
        if expected_prefect_api is None:
            assert environ[PREFECT_API_URL_ENV] == ""
            assert Path(environ[PREFECT_HOME_ENV]).exists()
        else:
            assert environ[PREFECT_API_URL_ENV] == expected_prefect_api
            assert PREFECT_HOME_ENV not in environ

    assert prefect_checks == ([expected_prefect_api] if expected_prefect_api else [])
    assert dask_checks == expected_dask_checks
    assert len(clusters) == expected_local_cluster_count
    assert all(cluster.closed for cluster in clusters)
    assert PREFECT_API_URL_ENV not in environ
    assert PREFECT_HOME_ENV not in environ
    assert PREFECT_SERVER_ANALYTICS_ENABLED_ENV not in environ

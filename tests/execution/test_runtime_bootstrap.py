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
        ExecutionConfig(task_runner="local_dask", max_nodes=2, cpus_per_task=4),
        dask_scheduler_checker=fail_if_checked,
    )

    assert plan.dask_scheduler is None
    assert plan.dask_worker_count is None
    assert "Using local Dask with 2 worker(s) and 4 thread(s) per worker." in caplog.text


def test_bootstrapped_runtime_sets_and_restores_external_prefect_api():
    environ = {PREFECT_API_URL_ENV: "http://original:4200/api"}
    config = ExecutionConfig(
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
        ExecutionConfig(prefect_api_mode="auto"),
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
        ExecutionConfig(prefect_api_mode="ephemeral"),
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

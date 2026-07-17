import configparser
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from rapthor.execution.config import ExecutionConfig

SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "dev" / "run-rapthor-prefect-demo.py"
DEMO_STRATEGY_PATH = Path(__file__).parents[2] / "examples" / "prefect_demo_strategy.py"


def load_demo_script():
    spec = importlib.util.spec_from_file_location("run_rapthor_prefect_demo", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_demo_strategy():
    spec = importlib.util.spec_from_file_location("prefect_demo_strategy", DEMO_STRATEGY_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_quick_demo_strategy_avoids_slow_gain_on_fixture_data():
    strategy = load_demo_strategy()

    assert strategy.strategy_steps
    assert all(step["target_flux"] is None for step in strategy.strategy_steps)
    assert all(step["max_directions"] >= 1 for step in strategy.strategy_steps)
    assert all(
        "slow_gains" not in step["calibration_strategy"].get("dd", [])
        for step in strategy.strategy_steps
    )


def test_dask_performance_report_path_defaults_to_run_dir(tmp_path):
    module = load_demo_script()

    assert module._dask_performance_report_path(False, None, tmp_path) is None
    assert module._dask_performance_report_path(True, None, tmp_path) == (
        tmp_path / "dask-performance-report.html"
    )
    assert module._dask_performance_report_path(True, "profile.html", tmp_path) == (
        Path.cwd() / "profile.html"
    )


def test_default_run_dir_is_unique():
    module = load_demo_script()

    assert module._default_run_dir() != module._default_run_dir()


def test_loading_demo_script_does_not_import_prefect(tmp_path):
    script = tmp_path / "import_demo.py"
    script.write_text(
        "\n".join(
            [
                "import importlib.util",
                "import json",
                "import sys",
                f"path = {str(SCRIPT_PATH)!r}",
                "spec = importlib.util.spec_from_file_location('demo_import_check', path)",
                "module = importlib.util.module_from_spec(spec)",
                "sys.modules[spec.name] = module",
                "spec.loader.exec_module(module)",
                "print(json.dumps({'prefect_imported': 'prefect' in sys.modules}))",
            ]
        ),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env.pop("PREFECT_API_URL", None)

    completed = subprocess.run(
        [sys.executable, script],
        cwd=Path(__file__).parents[2],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(completed.stdout)["prefect_imported"] is False


def test_working_dir_override_defaults_to_run_dir(tmp_path):
    module = load_demo_script()
    args = module._parse_args(["examples/prefect_demo.parset"])

    assert module._working_dir_override_from_args(args, tmp_path) == tmp_path / "rapthor-work"


def test_working_dir_override_can_be_disabled(tmp_path):
    module = load_demo_script()
    args = module._parse_args(["--no-unique-working-dir", "examples/prefect_demo.parset"])

    assert module._working_dir_override_from_args(args, tmp_path) is None


def test_materialize_parset_paths_overrides_working_dir(tmp_path):
    module = load_demo_script()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    parset_path = tmp_path / "demo.parset"
    parset_path.write_text(
        "\n".join(
            [
                "[global]",
                "dir_working = old-work",
                "input_ms = tests/resources/test.ms",
                "strategy = examples/prefect_demo_strategy.py",
                "",
                "[cluster]",
                "prefect_task_runner = sync",
            ]
        ),
        encoding="utf-8",
    )
    working_dir = run_dir / "rapthor-work"

    materialized = module._materialize_parset_paths(
        parset_path,
        run_dir,
        working_dir_override=working_dir,
    )

    parser = configparser.ConfigParser(interpolation=None)
    parser.read(materialized)
    assert parser["global"]["dir_working"] == str(working_dir)
    assert parser["global"]["input_ms"] == str((Path.cwd() / "tests/resources/test.ms").resolve())


def test_parse_args_keeps_parset_after_dask_performance_report_flag(monkeypatch):
    module = load_demo_script()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run-rapthor-prefect-demo.py",
            "--dask-performance-report",
            "examples/prefect_demo.parset",
        ],
    )

    args = module._parse_args()

    assert args.dask_performance_report is True
    assert args.dask_performance_report_path is None
    assert args.parset == Path("examples/prefect_demo.parset")


def test_parse_args_accepts_explicit_dask_performance_report_path(monkeypatch):
    module = load_demo_script()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run-rapthor-prefect-demo.py",
            "--dask-performance-report",
            "--dask-performance-report-path",
            "profile.html",
            "examples/prefect_demo.parset",
        ],
    )

    args = module._parse_args()

    assert args.dask_performance_report is True
    assert args.dask_performance_report_path == "profile.html"
    assert args.parset == Path("examples/prefect_demo.parset")


def test_execution_config_from_args_overrides_local_dask_resources(tmp_path):
    module = load_demo_script()
    parset_path = tmp_path / "demo.parset"
    parset_path.write_text(
        "\n".join(
            [
                "[global]",
                "dir_working = work",
                "input_ms = tests/resources/test.ms",
                "",
                "[cluster]",
                "prefect_task_runner = local_dask",
                "max_nodes = 1",
                "local_dask_workers = 1",
                "cpus_per_task = 2",
                "max_threads = 2",
                "prefect_command_profile = auto",
            ]
        ),
        encoding="utf-8",
    )
    args = module._parse_args(
        [
            "--local-dask-workers",
            "3",
            "--cpus-per-task",
            "30",
            "--max-threads",
            "30",
            "--filter-skymodel-ncores",
            "15",
            "--command-profile",
            "time",
            str(parset_path),
        ]
    )

    config = module._execution_config_from_args(parset_path, args)

    assert config.max_nodes == 1
    assert config.local_dask_workers == 3
    assert config.local_dask_worker_count == 3
    assert config.cpus_per_task == 30
    assert config.command_profile == "time"
    assert module._cluster_parset_overrides_from_args(args) == {
        "filter_skymodel_ncores": 15,
        "max_threads": 30,
    }


def test_demo_main_passes_benchmark_resources_to_dask_and_runtime_parset(monkeypatch, tmp_path):
    module = load_demo_script()
    run_dir = tmp_path / "run"
    parset_path = tmp_path / "benchmark.parset"
    parset_path.write_text(
        "\n".join(
            [
                "[global]",
                "dir_working = work",
                "input_ms = tests/resources/test.ms",
                "",
                "[cluster]",
                "prefect_task_runner = local_dask",
                "max_nodes = 1",
                "local_dask_workers = 0",
                "cpus_per_task = 0",
                "max_threads = 0",
                "filter_skymodel_ncores = 0",
                "prefect_command_profile = auto",
            ]
        ),
        encoding="utf-8",
    )
    calls = {"closed": False}

    @dataclass
    class FakeDaskCluster:
        scheduler_address: str = "tcp://127.0.0.1:8786"
        dashboard_url: str = "http://127.0.0.1:8787/status"
        worker_count: int = 2
        client: object = object()

        def close(self):
            calls["closed"] = True

    def fake_start_local_dask_cluster(execution_config):
        calls["cluster_config"] = execution_config
        return FakeDaskCluster()

    def fake_pipeline_flow(parset_file, logging_level, execution_config):
        calls["pipeline_parset"] = parset_file
        calls["logging_level"] = logging_level
        calls["pipeline_config"] = execution_config

    monkeypatch.setattr(module, "_wait_for_prefect", lambda api_url, timeout_seconds: None)
    monkeypatch.setattr(module, "_start_local_dask_cluster", fake_start_local_dask_cluster)
    monkeypatch.setattr(module, "_run_pipeline_flow", fake_pipeline_flow)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run-rapthor-prefect-demo.py",
            "--no-start-server",
            "--no-unique-working-dir",
            "--no-materialize-paths",
            "--run-dir",
            str(run_dir),
            "--local-dask-workers",
            "2",
            "--cpus-per-task",
            "30",
            "--max-threads",
            "30",
            "--filter-skymodel-ncores",
            "15",
            "--command-profile",
            "time",
            "--no-keep-server",
            "--no-keep-server-on-failure",
            str(parset_path),
        ],
    )

    assert module.main() == 0

    cluster_config = calls["cluster_config"]
    assert cluster_config.task_runner == "local_dask"
    assert cluster_config.local_dask_workers == 2
    assert cluster_config.local_dask_worker_count == 2
    assert cluster_config.cpus_per_task == 30
    assert cluster_config.local_dask_threads_per_worker == 1
    assert cluster_config.command_threads_per_task == 30
    assert cluster_config.command_profile == "time"

    pipeline_config = calls["pipeline_config"]
    assert pipeline_config.task_runner == "external_dask"
    assert pipeline_config.dask_scheduler == "tcp://127.0.0.1:8786"
    assert pipeline_config.local_dask_workers == 2
    assert pipeline_config.cpus_per_task == 30
    assert pipeline_config.command_profile == "time"
    assert calls["closed"] is True

    runtime_parser = configparser.ConfigParser(interpolation=None)
    runtime_parser.read(calls["pipeline_parset"])
    runtime_cluster = runtime_parser["cluster"]
    assert runtime_cluster["prefect_task_runner"] == "external_dask"
    assert runtime_cluster["dask_scheduler"] == "tcp://127.0.0.1:8786"
    assert runtime_cluster["local_dask_workers"] == "2"
    assert runtime_cluster["cpus_per_task"] == "30"
    assert runtime_cluster["max_threads"] == "30"
    assert runtime_cluster["filter_skymodel_ncores"] == "15"
    assert runtime_cluster["prefect_command_profile"] == "time"


def test_dask_performance_report_opens_external_client(tmp_path):
    module = load_demo_script()
    events = []

    class FakeClient:
        def __init__(self, scheduler):
            events.append(("client", scheduler))

        def close(self):
            events.append(("close", None))

    class FakePerformanceReport:
        def __init__(self, filename):
            self.filename = filename

        def __enter__(self):
            events.append(("enter", self.filename))

        def __exit__(self, exc_type, exc, traceback):
            events.append(("exit", self.filename))

    report_path = tmp_path / "report.html"
    config = ExecutionConfig(task_runner="external_dask", dask_scheduler="tcp://scheduler:8786")

    with module._dask_performance_report(
        report_path,
        config,
        performance_report_factory=FakePerformanceReport,
        client_cls=FakeClient,
    ) as active_report:
        assert active_report == report_path

    assert events == [
        ("client", "tcp://scheduler:8786"),
        ("enter", str(report_path)),
        ("exit", str(report_path)),
        ("close", None),
    ]


def test_dask_performance_report_requires_scheduler(tmp_path):
    module = load_demo_script()
    config = ExecutionConfig(task_runner="local_dask")

    with pytest.raises(
        RuntimeError,
        match="requires a persistent local or external Dask scheduler",
    ):
        with module._dask_performance_report(
            tmp_path / "report.html",
            config,
            performance_report_factory=object,
            client_cls=object,
        ):
            raise AssertionError("Expected _dask_performance_report to fail before yielding")


def test_started_prefect_server_disables_server_analytics(monkeypatch, tmp_path):
    module = load_demo_script()
    popen_calls = []

    class FakeServer:
        def terminate(self):
            raise AssertionError("server should not be terminated")

    def fake_popen(command, **kwargs):
        popen_calls.append((command, kwargs))
        return FakeServer()

    monkeypatch.setattr(module, "_is_prefect_healthy", lambda api_url: False)
    monkeypatch.setattr(module, "_wait_for_prefect", lambda api_url, timeout_seconds: None)
    monkeypatch.setattr(module.subprocess, "Popen", fake_popen)
    prefect_home = tmp_path / "prefect-home"
    prefect_home.mkdir()
    monkeypatch.setattr(module, "mkdtemp", lambda **kwargs: str(prefect_home))

    server = module._start_prefect_server(
        "0.0.0.0",
        4200,
        tmp_path,
        "http://127.0.0.1:4200/api",
        timeout_seconds=5,
    )

    assert isinstance(server, module.StartedPrefectServer)
    assert server.process.__class__ is FakeServer
    assert server.prefect_home == prefect_home
    assert popen_calls[0][0] == [
        "prefect",
        "server",
        "start",
        "--host",
        "0.0.0.0",
        "--port",
        "4200",
    ]
    assert popen_calls[0][1]["env"][module.PREFECT_HOME_ENV] == str(prefect_home)
    assert popen_calls[0][1]["env"][module.PREFECT_SERVER_ANALYTICS_ENABLED_ENV] == "false"
    assert popen_calls[0][1]["env"][module.PREFECT_UI_API_URL_ENV] == "/api"


def test_keep_prefect_server_open_waits_for_ctrl_c_and_cleans_state(monkeypatch, tmp_path):
    module = load_demo_script()
    prefect_home = tmp_path / "prefect-home"
    prefect_home.mkdir()
    calls = []

    class FakeProcess:
        pid = 1234

        def wait(self, timeout=None):
            calls.append(("wait", timeout))
            if timeout is None:
                raise KeyboardInterrupt

        def poll(self):
            return None

        def terminate(self):
            calls.append(("terminate", None))

        def kill(self):
            calls.append(("kill", None))

    server = module.StartedPrefectServer(FakeProcess(), prefect_home)

    module._keep_prefect_server_open(server)

    assert calls == [("wait", None), ("terminate", None), ("wait", 10)]
    assert not prefect_home.exists()

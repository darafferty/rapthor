import importlib.util
import configparser
import sys
from pathlib import Path

from rapthor.execution.config import ExecutionConfig


SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "dev" / "run-rapthor-prefect-demo.py"


def load_demo_script():
    spec = importlib.util.spec_from_file_location("run_rapthor_prefect_demo", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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

    try:
        with module._dask_performance_report(
            tmp_path / "report.html",
            config,
            performance_report_factory=object,
            client_cls=object,
        ):
            pass
    except RuntimeError as err:
        assert "requires a persistent local or external Dask scheduler" in str(err)
    else:
        raise AssertionError("Expected RuntimeError")

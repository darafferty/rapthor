import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from rapthor.execution.benchmarking import (
    BenchmarkRunResult,
    BenchmarkScenario,
    CommandMetric,
    OperationTimingMetric,
    ParsetOverride,
    benchmark_run_result,
    benchmark_scenarios_by_id,
    failed_benchmark_runs,
    format_failed_benchmark_runs,
    parse_command_log,
    parse_dask_performance_report,
    parse_operation_log,
    render_markdown_report,
    run_scenario_once,
    summarize_benchmark_runs,
    write_summary_artifacts,
)

BENCHMARK_SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "dev" / "run_benchmark_baseline.py"
DASK_REPORT_HTML = """
<html>
  &lt;h2&gt; Duration: 12.50 s &lt;/h2&gt;
  &lt;li&gt; number of tasks: 3 &lt;/li&gt;
  &lt;li&gt; compute time: 7.25 s &lt;/li&gt;
  &lt;li&gt; Workers: 2 &lt;/li&gt;
  &lt;li&gt; Threads: 8 &lt;/li&gt;
  &lt;li&gt; Memory: 12.00 GiB &lt;/li&gt;
  "name":"task_stream",
  ["duration",[1000.0,2500.0,3750.0]],
  ["name",["image_sector_task","calibrate_chunk_task","image_sector_task"]]
</html>
"""
RAPTHOR_LOG = """
2026-07-04 09:47:29,801 - DEBUG - rapthor:predict_di_1 - \x1b[35mTime for operation: 0:00:01.250000\x1b[0m
2026-07-04 09:47:29,801 - DEBUG - rapthor:predict_di_1 - \x1b[35m\x1b[35mTime for operation: 0:00:01.250000\x1b[0m\x1b[0m
2026-07-04 09:48:48,220 - DEBUG - rapthor:image_1 - \x1b[35mTime for operation: 0:01:12.500000\x1b[0m
"""


def load_benchmark_script():
    spec = importlib.util.spec_from_file_location("run_benchmark_baseline", BENCHMARK_SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_default_benchmark_scenarios_build_demo_commands():
    scenarios = benchmark_scenarios_by_id()
    scenario = scenarios["ci-benchmark"]
    calibration_scenario = scenarios["ci-benchmark-calibration-postprocess"]
    chunked_predict_scenario = scenarios["ci-benchmark-predict-chunks"]
    many_sector_scenario = scenarios["ci-benchmark-many-sector-mosaic"]
    sparse_fallback_scenario = scenarios["ci-benchmark-many-sector-mosaic-sparse-fallback"]

    command = scenario.command(Path("/repo"), Path("/runs/benchmark"))
    many_sector_command = many_sector_scenario.command(Path("/repo"), Path("/runs/benchmark"))

    assert set(scenarios) == {
        "ci-benchmark",
        "ci-benchmark-calibration-postprocess",
        "ci-benchmark-image-products",
        "ci-benchmark-many-sector-mosaic",
        "ci-benchmark-many-sector-mosaic-sparse-fallback",
        "ci-benchmark-predict-chunks",
        "ci-benchmark-wsclean-predict",
    }
    assert command[1] == "/repo/scripts/dev/run-rapthor-prefect-demo.py"
    assert "/repo/examples/generated/prefect_demo_rich/prefect_demo_benchmark.parset" in command
    assert "--dask-performance-report" in command
    assert command[command.index("--command-profile") + 1] == "time"
    assert command[command.index("--local-dask-workers") + 1] == "2"
    assert command[command.index("--cpus-per-task") + 1] == "30"
    assert command[command.index("--max-threads") + 1] == "30"
    assert "--no-keep-server" in command
    assert calibration_scenario.parset_overrides == (
        ParsetOverride(
            "global",
            "strategy",
            "examples/generated/prefect_demo_rich/"
            "prefect_demo_benchmark_calibration_postprocess_strategy.py",
        ),
    )
    assert chunked_predict_scenario.parset_overrides == (
        ParsetOverride("cluster", "max_nodes", "2"),
    )
    assert (
        "/repo/examples/generated/prefect_demo_rich/prefect_demo_multisector_benchmark.parset"
        in many_sector_command
    )
    assert many_sector_scenario.parset_overrides == ()
    assert sparse_fallback_scenario.parset_overrides == (
        ParsetOverride("imaging", "model_mosaic_method", "sparse_fits"),
    )


def test_hidden_path_benchmark_scenarios_materialize_parset_overrides(tmp_path):
    scenario = benchmark_scenarios_by_id()["ci-benchmark-image-products"]
    calibration_scenario = benchmark_scenarios_by_id()["ci-benchmark-calibration-postprocess"]
    chunked_predict_scenario = benchmark_scenarios_by_id()["ci-benchmark-predict-chunks"]
    sparse_fallback_scenario = benchmark_scenarios_by_id()[
        "ci-benchmark-many-sector-mosaic-sparse-fallback"
    ]
    repo_root = tmp_path / "repo"
    parset_path = repo_root / "examples/generated/prefect_demo_rich/prefect_demo_benchmark.parset"
    multi_sector_parset_path = (
        repo_root / "examples/generated/prefect_demo_rich/prefect_demo_multisector_benchmark.parset"
    )
    parset_path.parent.mkdir(parents=True)
    base_parset = (
        """
[global]
strategy = examples/generated/prefect_demo_rich/prefect_demo_benchmark_strategy.py

[imaging]
model_mosaic_method = wsclean
save_image_cube = False
normalization_skymodels =

[cluster]
prefect_task_runner = local_dask
""".strip()
        + "\n"
    )
    parset_path.write_text(
        base_parset,
        encoding="utf-8",
    )
    multi_sector_parset_path.write_text(base_parset, encoding="utf-8")

    command = scenario.command(repo_root, tmp_path / "run")
    calibration_command = calibration_scenario.command(repo_root, tmp_path / "run")
    chunked_predict_command = chunked_predict_scenario.command(repo_root, tmp_path / "run")
    sparse_fallback_command = sparse_fallback_scenario.command(repo_root, tmp_path / "run")
    scenario_parset = Path(command[2])
    calibration_parset = Path(calibration_command[2])
    chunked_predict_parset = Path(chunked_predict_command[2])
    sparse_fallback_parset = Path(sparse_fallback_command[2])

    assert scenario_parset.name == ("prefect_demo_benchmark.ci-benchmark-image-products.parset")
    text = scenario_parset.read_text(encoding="utf-8")
    assert "prefect_demo_benchmark_normalize_strategy.py" in text
    assert "prefect_demo_rich_reference_120mhz.txt" in text
    assert "prefect_demo_rich_reference_160mhz.txt" in text
    assert "normalization_reference_frequencies = [120000000.0, 160000000.0]" in text
    assert "save_image_cube = True" in text
    assert "make_quv_images = True" in text
    assert "compress_final_images = True" in text
    assert calibration_parset.name == (
        "prefect_demo_benchmark.ci-benchmark-calibration-postprocess.parset"
    )
    calibration_text = calibration_parset.read_text(encoding="utf-8")
    assert "prefect_demo_benchmark_calibration_postprocess_strategy.py" in calibration_text
    assert chunked_predict_parset.name == (
        "prefect_demo_benchmark.ci-benchmark-predict-chunks.parset"
    )
    assert "max_nodes = 2" in chunked_predict_parset.read_text(encoding="utf-8")
    assert sparse_fallback_parset.name == (
        "prefect_demo_multisector_benchmark.ci-benchmark-many-sector-mosaic-sparse-fallback.parset"
    )
    assert "model_mosaic_method = sparse_fits" in sparse_fallback_parset.read_text(encoding="utf-8")


def test_benchmark_scenario_without_overrides_uses_base_parset():
    scenario = benchmark_scenarios_by_id()["ci-benchmark"]

    assert scenario.parset_path_for_run(Path("/repo"), Path("/run")) == Path(
        "/repo/examples/generated/prefect_demo_rich/prefect_demo_benchmark.parset"
    )


def test_benchmark_runner_applies_runtime_overrides():
    module = load_benchmark_script()
    args = module._parse_args(
        [
            "--scenario",
            "ci-benchmark",
            "--local-dask-workers",
            "1",
            "--cpus-per-task",
            "4",
            "--max-threads",
            "4",
        ]
    )
    scenario = module._scenario_with_runtime_overrides(
        benchmark_scenarios_by_id()["ci-benchmark"],
        args,
    )
    command = scenario.command(Path("/repo"), Path("/runs/benchmark"))

    assert command[command.index("--local-dask-workers") + 1] == "1"
    assert command[command.index("--cpus-per-task") + 1] == "4"
    assert command[command.index("--max-threads") + 1] == "4"


def test_benchmark_runner_expands_resource_profiles_after_runtime_overrides():
    module = load_benchmark_script()
    args = module._parse_args(
        [
            "--scenario",
            "ci-benchmark",
            "--local-dask-workers",
            "2",
            "--cpus-per-task",
            "30",
            "--max-threads",
            "30",
            "--resource-profile",
            "filter-only-15",
            "--resource-profile",
            "filter-workers-4x15",
        ]
    )

    scenarios = module._selected_scenarios(benchmark_scenarios_by_id(), args)

    assert [scenario.scenario_id for scenario in scenarios] == [
        "ci-benchmark-filter-only-15",
        "ci-benchmark-filter-workers-4x15",
    ]
    assert [
        (
            scenario.local_dask_workers,
            scenario.cpus_per_task,
            scenario.max_threads,
            scenario.filter_skymodel_ncores,
        )
        for scenario in scenarios
    ] == [(2, 30, 30, 15), (4, 15, 15, None)]

    command = scenarios[0].command(Path("/repo"), Path("/runs/benchmark"))
    assert command[command.index("--max-threads") + 1] == "30"
    assert command[command.index("--filter-skymodel-ncores") + 1] == "15"


def test_benchmark_resource_profile_scenarios_still_use_generated_inputs():
    module = load_benchmark_script()
    args = module._parse_args(
        ["--scenario", "ci-benchmark", "--resource-profile", "filter-only-15"]
    )
    scenario = module._selected_scenarios(benchmark_scenarios_by_id(), args)[0]

    assert module._scenario_uses_generated_demo_inputs(scenario) is True


def test_benchmark_runner_recognizes_many_sector_generated_inputs():
    module = load_benchmark_script()
    scenario = benchmark_scenarios_by_id()["ci-benchmark-many-sector-mosaic"]
    sparse_fallback_scenario = benchmark_scenarios_by_id()[
        "ci-benchmark-many-sector-mosaic-sparse-fallback"
    ]

    assert module._scenario_uses_generated_demo_inputs(scenario) is True
    assert module._scenario_uses_multi_sector_demo_inputs(scenario) is True
    assert module._scenario_uses_generated_demo_inputs(sparse_fallback_scenario) is True
    assert module._scenario_uses_multi_sector_demo_inputs(sparse_fallback_scenario) is True


def test_benchmark_prepare_inputs_generates_many_sector_dataset(monkeypatch, tmp_path):
    module = load_benchmark_script()
    scenario = benchmark_scenarios_by_id()["ci-benchmark-many-sector-mosaic"]
    calls = []

    monkeypatch.setattr(module, "_ensure_test_ms", lambda repo_root: Path("/repo/test.ms"))
    monkeypatch.setattr(
        module, "_missing_scenario_inputs", lambda repo_root, scenarios: ["missing"]
    )

    def fake_run(command, cwd, check):
        calls.append({"command": command, "cwd": cwd, "check": check})

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    module._prepare_benchmark_inputs(tmp_path, [scenario])

    assert calls == [
        {
            "command": [
                sys.executable,
                str(tmp_path / "scripts" / "dev" / "generate-prefect-demo-data.py"),
                "--template-ms",
                "/repo/test.ms",
                "--force",
                "--include-multi-sector",
            ],
            "cwd": tmp_path,
            "check": True,
        }
    ]


def test_benchmark_input_validation_includes_scenario_override_paths(tmp_path):
    module = load_benchmark_script()
    parset_path = tmp_path / "examples/generated/prefect_demo_rich/prefect_demo_benchmark.parset"
    parset_path.parent.mkdir(parents=True)
    parset_path.write_text(
        """
[global]
strategy = examples/generated/prefect_demo_rich/prefect_demo_benchmark_strategy.py

[imaging]
normalization_skymodels =
normalization_reference_frequencies =
""".strip()
        + "\n",
        encoding="utf-8",
    )
    scenario = BenchmarkScenario(
        scenario_id="override-paths",
        description="override paths",
        parset="examples/generated/prefect_demo_rich/prefect_demo_benchmark.parset",
        local_dask_workers=1,
        cpus_per_task=1,
        max_threads=1,
        parset_overrides=(
            ParsetOverride(
                "global",
                "strategy",
                "examples/generated/prefect_demo_rich/missing_strategy.py",
            ),
            ParsetOverride(
                "imaging",
                "normalization_skymodels",
                "[examples/generated/prefect_demo_rich/missing_a.txt, "
                "examples/generated/prefect_demo_rich/missing_b.txt]",
            ),
        ),
    )

    missing = module._missing_scenario_inputs(tmp_path, [scenario])

    assert any("strategy does not exist" in item for item in missing)
    assert any("normalization_skymodels does not exist" in item for item in missing)


def test_parse_command_log_extracts_timing_records(tmp_path):
    command_log = tmp_path / "commands.jsonl"
    command_log.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "command": ["DP3", "msin=input.ms"],
                        "operation": "calibrate_1",
                        "status": "completed",
                        "duration_seconds": 1.25,
                    }
                ),
                json.dumps(
                    {
                        "name": "wsclean",
                        "operation": "image_1",
                        "status": "completed",
                        "duration_seconds": 2.5,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    metrics = parse_command_log(command_log)

    assert [metric.name for metric in metrics] == ["DP3", "wsclean"]
    assert [metric.operation for metric in metrics] == ["calibrate_1", "image_1"]
    assert sum(metric.duration_seconds for metric in metrics) == 3.75


def test_benchmark_run_result_reads_commands_and_dask_report(tmp_path):
    run_dir = tmp_path / "run"
    logs_dir = run_dir / "rapthor-work" / "logs"
    logs_dir.mkdir(parents=True)
    (logs_dir / "commands.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "command": ["DP3"],
                        "duration_seconds": 1.5,
                        "operation": "predict_di_1",
                    }
                ),
                json.dumps(
                    {
                        "command": ["wsclean"],
                        "duration_seconds": 10.0,
                        "operation": "image_1",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (logs_dir / "rapthor.log").write_text(RAPTHOR_LOG, encoding="utf-8")
    (run_dir / "dask-performance-report.html").write_text(DASK_REPORT_HTML, encoding="utf-8")

    result = benchmark_run_result(
        scenario_id="ci-benchmark",
        repetition=1,
        run_dir=run_dir,
        returncode=0,
        wall_seconds=3.0,
    )

    assert result.command_count == 2
    assert result.command_seconds == 11.5
    assert [(command.name, command.duration_seconds) for command in result.command_timings] == [
        ("DP3", 1.5),
        ("wsclean", 10.0),
    ]
    assert result.dask_performance_report == str(run_dir / "dask-performance-report.html")
    assert result.dask_duration_seconds == 12.5
    assert result.dask_compute_seconds == 7.25
    assert result.dask_duration_minus_compute_seconds == 5.25
    assert result.dask_task_count == 3
    assert result.dask_workers == 2
    assert result.dask_threads == 8
    assert result.dask_memory == "12.00 GiB"
    assert [group.name for group in result.dask_task_groups] == [
        "calibrate_chunk_task",
        "image_sector_task",
    ]
    assert result.dask_task_groups[1].total_seconds == 4.75
    assert [(timing.operation, timing.command_count) for timing in result.operation_timings] == [
        ("predict_di_1", 1),
        ("image_1", 1),
    ]
    assert result.operation_timings[0].operation_minus_command_seconds == -0.25
    assert result.operation_timings[1].operation_minus_command_seconds == 62.5


def test_parse_operation_log_deduplicates_colored_operation_timings(tmp_path):
    log_path = tmp_path / "rapthor.log"
    log_path.write_text(RAPTHOR_LOG, encoding="utf-8")

    timings = parse_operation_log(log_path)

    assert [(timing.operation, timing.elapsed_seconds) for timing in timings] == [
        ("predict_di_1", 1.25),
        ("image_1", 72.5),
    ]


def test_parse_dask_performance_report_extracts_gap_and_task_groups(tmp_path):
    report_path = tmp_path / "dask-performance-report.html"
    report_path.write_text(DASK_REPORT_HTML, encoding="utf-8")

    metrics = parse_dask_performance_report(report_path)

    assert metrics.duration_seconds == 12.5
    assert metrics.compute_seconds == 7.25
    assert metrics.duration_minus_compute_seconds == 5.25
    assert metrics.task_count == 3
    assert metrics.workers == 2
    assert metrics.threads == 8
    assert metrics.memory == "12.00 GiB"
    assert [(group.name, group.count) for group in metrics.task_groups] == [
        ("calibrate_chunk_task", 1),
        ("image_sector_task", 2),
    ]


def test_summarize_benchmark_runs_and_render_markdown():
    summary = summarize_benchmark_runs(
        [
            BenchmarkRunResult(
                "ci-benchmark",
                1,
                "/runs/1",
                0,
                4.0,
                2,
                3.0,
                dask_duration_seconds=12.0,
                dask_compute_seconds=8.0,
                dask_duration_minus_compute_seconds=4.0,
                dask_task_count=5,
                dask_workers=2,
                dask_threads=8,
                command_timings=(
                    CommandMetric("wsclean", "image_1", "completed", 40.0),
                    CommandMetric("filter_skymodel", "image_1", "completed", 20.0),
                ),
                operation_timings=(
                    OperationTimingMetric(
                        "image_1",
                        elapsed_seconds=72.0,
                        command_count=4,
                        command_seconds=60.0,
                        operation_minus_command_seconds=12.0,
                    ),
                ),
            ),
            BenchmarkRunResult(
                "ci-benchmark",
                2,
                "/runs/2",
                0,
                2.0,
                4,
                1.0,
                dask_duration_seconds=10.0,
                dask_compute_seconds=6.0,
                dask_duration_minus_compute_seconds=4.0,
                dask_task_count=3,
                dask_workers=2,
                dask_threads=8,
                command_timings=(
                    CommandMetric("wsclean", "image_1", "completed", 38.0),
                    CommandMetric("filter_skymodel", "image_1", "completed", 20.0),
                ),
                operation_timings=(
                    OperationTimingMetric(
                        "image_1",
                        elapsed_seconds=70.0,
                        command_count=4,
                        command_seconds=58.0,
                        operation_minus_command_seconds=12.0,
                    ),
                ),
            ),
            BenchmarkRunResult("comparison-demo", 1, "/runs/3", 1, 8.0, 5, 6.0),
        ]
    )

    assert summary["scenarios"]["ci-benchmark"]["wall_seconds"] == {
        "min": 2.0,
        "median": 3.0,
        "max": 4.0,
    }
    assert summary["scenarios"]["ci-benchmark"]["command_count"]["median"] == 3.0
    assert summary["scenarios"]["ci-benchmark"]["dask"]["duration_seconds"] == {
        "min": 10.0,
        "median": 11.0,
        "max": 12.0,
    }
    assert summary["scenarios"]["ci-benchmark"]["dask"]["duration_minus_compute_seconds"] == {
        "min": 4.0,
        "median": 4.0,
        "max": 4.0,
    }
    assert summary["scenarios"]["ci-benchmark"]["operations"]["by_operation"]["image_1"][
        "elapsed_seconds"
    ] == {
        "min": 70.0,
        "median": 71.0,
        "max": 72.0,
    }
    assert summary["scenarios"]["ci-benchmark"]["commands"]["wsclean"]["seconds"] == {
        "min": 38.0,
        "median": 39.0,
        "max": 40.0,
    }

    report = render_markdown_report(summary)

    assert "# Rapthor Benchmark Baseline" in report
    assert "| ci-benchmark | 2 | 0, 0 | 3.000 | 2.000-4.000 | 2.000 |" in report
    assert "| comparison-demo | 1 | 1 | 8.000 | 8.000-8.000 | 6.000 |" in report
    assert "## Dask Performance" in report
    assert "| ci-benchmark | 11.000 | 7.000 | 4.000 | 4.000 | 2 | 8 |" in report
    assert "## Operation Timing" in report
    assert "| ci-benchmark | image_1 | 71.000 | 59.000 | 12.000 | 4.000 |" in report
    assert "## Command Timing" in report
    assert "| ci-benchmark | wsclean | 39.000 | 1.000 |" in report
    assert "| ci-benchmark | filter_skymodel | 20.000 | 1.000 |" in report


def test_benchmark_report_includes_metadata():
    summary = summarize_benchmark_runs(
        [BenchmarkRunResult("ci-benchmark", 1, "/runs/1", 0, 4.0, 2, 3.0)]
    )
    summary["metadata"] = {
        "container_image": "registry.example/rapthor/full:abc123",
        "gitlab_job_url": "https://gitlab.example/jobs/123",
    }

    report = render_markdown_report(summary)

    assert "| Container image | registry.example/rapthor/full:abc123 |" in report
    assert "| GitLab job URL | https://gitlab.example/jobs/123 |" in report


def test_write_summary_artifacts(tmp_path):
    output_json = tmp_path / "summary.json"
    output_markdown = tmp_path / "report.md"

    summary = write_summary_artifacts(
        [BenchmarkRunResult("ci-benchmark", 1, "/runs/1", 0, 4.0, 2, 3.0)],
        output_json=output_json,
        output_markdown=output_markdown,
    )

    assert json.loads(output_json.read_text()) == summary
    assert output_markdown.read_text().startswith("# Rapthor Benchmark Baseline")


def test_write_summary_artifacts_can_include_metadata(tmp_path):
    output_json = tmp_path / "summary.json"
    output_markdown = tmp_path / "report.md"

    summary = write_summary_artifacts(
        [BenchmarkRunResult("ci-benchmark", 1, "/runs/1", 0, 4.0, 2, 3.0)],
        output_json=output_json,
        output_markdown=output_markdown,
        metadata={"gitlab_job_url": "https://gitlab.example/jobs/123"},
    )

    assert summary["metadata"]["gitlab_job_url"] == "https://gitlab.example/jobs/123"
    assert json.loads(output_json.read_text())["metadata"] == summary["metadata"]
    assert "https://gitlab.example/jobs/123" in output_markdown.read_text()


def test_benchmark_runner_reads_gitlab_metadata():
    module = load_benchmark_script()

    metadata = module._benchmark_metadata_from_env(
        {
            "CI_JOB_URL": "https://gitlab.example/jobs/123",
            "CI_PIPELINE_URL": "https://gitlab.example/pipelines/456",
            "FULL_IMAGE": "registry.example/rapthor/full:abc123",
        }
    )

    assert metadata == {
        "container_image": "registry.example/rapthor/full:abc123",
        "gitlab_job_url": "https://gitlab.example/jobs/123",
        "gitlab_pipeline_url": "https://gitlab.example/pipelines/456",
    }


def test_failed_benchmark_runs_are_reported():
    results = [
        BenchmarkRunResult("ci-benchmark", 1, "/runs/1", 0, 4.0, 2, 3.0),
        BenchmarkRunResult("comparison-demo", 2, "/runs/2", 2, 1.0, 0, 0.0),
    ]

    assert failed_benchmark_runs(results) == [results[1]]
    assert "comparison-demo repetition 2" in format_failed_benchmark_runs(results)


def test_run_scenario_once_writes_command_and_result_artifacts(tmp_path):
    scenario = benchmark_scenarios_by_id()["ci-benchmark"]
    times = iter([10.0, 14.5])

    def fake_runner(command, cwd, check):
        run_dir = Path(command[command.index("--run-dir") + 1])
        logs_dir = run_dir / "rapthor-work" / "logs"
        logs_dir.mkdir(parents=True)
        (logs_dir / "commands.jsonl").write_text(
            json.dumps({"command": ["DP3"], "duration_seconds": 2.0}) + "\n",
            encoding="utf-8",
        )
        assert cwd == tmp_path
        assert check is False
        return subprocess.CompletedProcess(command, 0)

    result = run_scenario_once(
        scenario,
        repetition=1,
        run_root=tmp_path / "runs",
        repo_root=tmp_path,
        command_runner=fake_runner,
        monotonic=lambda: next(times),
    )

    run_dir = tmp_path / "runs" / "ci-benchmark" / "rep-01"
    assert result.wall_seconds == 4.5
    assert result.command_seconds == 2.0
    assert (run_dir / "benchmark-command.json").is_file()
    assert json.loads((run_dir / "benchmark-result.json").read_text())["returncode"] == 0


def test_benchmark_runner_validates_missing_inputs(tmp_path):
    module = load_benchmark_script()
    repo_root = tmp_path
    scenario = benchmark_scenarios_by_id()["ci-benchmark"]
    parset = repo_root / scenario.parset
    parset.parent.mkdir(parents=True)
    parset.write_text(
        """
        [global]
        input_ms = tests/resources/test.ms
        input_skymodel = tests/resources/test_true_sky.txt
        apparent_skymodel = None
        strategy = examples/prefect_demo_strategy.py
        """,
        encoding="utf-8",
    )
    try:
        module._validate_scenario_inputs(repo_root, [scenario])
    except SystemExit as err:
        message = str(err)
    else:
        raise AssertionError("missing benchmark inputs should stop validation")

    assert "tests/resources/test.ms" in message
    assert "--prepare-inputs" in message


def test_benchmark_runner_returns_failure_for_failed_repetition(monkeypatch, tmp_path):
    module = load_benchmark_script()
    scenario = benchmark_scenarios_by_id()["ci-benchmark"]

    monkeypatch.setattr(module, "benchmark_scenarios_by_id", lambda: {"ci-benchmark": scenario})
    monkeypatch.setattr(module, "_validate_scenario_inputs", lambda repo_root, scenarios: None)
    monkeypatch.setattr(module, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        module,
        "run_scenario_once",
        lambda *args, **kwargs: BenchmarkRunResult(
            "ci-benchmark", 1, str(tmp_path / "run"), 1, 0.5, 0, 0.0
        ),
    )

    exit_code = module.main(
        [
            "--scenario",
            "ci-benchmark",
            "--repetitions",
            "1",
            "--run-root",
            str(tmp_path / "bench"),
        ]
    )

    assert exit_code == 1


def test_benchmark_runner_can_allow_failed_repetitions(monkeypatch, tmp_path):
    module = load_benchmark_script()
    scenario = benchmark_scenarios_by_id()["ci-benchmark"]

    monkeypatch.setattr(module, "benchmark_scenarios_by_id", lambda: {"ci-benchmark": scenario})
    monkeypatch.setattr(module, "_validate_scenario_inputs", lambda repo_root, scenarios: None)
    monkeypatch.setattr(module, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        module,
        "run_scenario_once",
        lambda *args, **kwargs: BenchmarkRunResult(
            "ci-benchmark", 1, str(tmp_path / "run"), 1, 0.5, 0, 0.0
        ),
    )

    exit_code = module.main(
        [
            "--scenario",
            "ci-benchmark",
            "--repetitions",
            "1",
            "--run-root",
            str(tmp_path / "bench"),
            "--allow-failures",
        ]
    )

    assert exit_code == 0

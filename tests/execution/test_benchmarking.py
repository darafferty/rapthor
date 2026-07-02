import json
import subprocess
from pathlib import Path

from rapthor.execution.benchmarking import (
    BenchmarkRunResult,
    benchmark_run_result,
    benchmark_scenarios_by_id,
    parse_command_log,
    render_markdown_report,
    run_scenario_once,
    summarize_benchmark_runs,
    write_summary_artifacts,
)


def test_default_benchmark_scenarios_build_demo_commands():
    scenarios = benchmark_scenarios_by_id()

    quick_command = scenarios["quick-demo"].command(Path("/repo"), Path("/runs/quick"))
    rich_command = scenarios["rich-demo"].command(Path("/repo"), Path("/runs/rich"))

    assert quick_command[1] == "/repo/scripts/dev/run-rapthor-prefect-demo.py"
    assert "/repo/examples/prefect_demo.parset" in quick_command
    assert "--dask-performance-report" in quick_command
    assert quick_command[quick_command.index("--command-profile") + 1] == "time"
    assert "--no-keep-server" in quick_command
    assert "/repo/examples/generated/prefect_demo_rich/prefect_demo_rich.parset" in rich_command
    assert "2" == rich_command[rich_command.index("--local-dask-workers") + 1]


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
        json.dumps({"command": ["DP3"], "duration_seconds": 1.5}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "dask-performance-report.html").write_text("<html></html>", encoding="utf-8")

    result = benchmark_run_result(
        scenario_id="quick-demo",
        repetition=1,
        run_dir=run_dir,
        returncode=0,
        wall_seconds=3.0,
    )

    assert result.command_count == 1
    assert result.command_seconds == 1.5
    assert result.dask_performance_report == str(run_dir / "dask-performance-report.html")


def test_summarize_benchmark_runs_and_render_markdown():
    summary = summarize_benchmark_runs(
        [
            BenchmarkRunResult("quick-demo", 1, "/runs/1", 0, 4.0, 2, 3.0),
            BenchmarkRunResult("quick-demo", 2, "/runs/2", 0, 2.0, 4, 1.0),
            BenchmarkRunResult("rich-demo", 1, "/runs/3", 1, 8.0, 5, 6.0),
        ]
    )

    assert summary["scenarios"]["quick-demo"]["wall_seconds"] == {
        "min": 2.0,
        "median": 3.0,
        "max": 4.0,
    }
    assert summary["scenarios"]["quick-demo"]["command_count"]["median"] == 3.0

    report = render_markdown_report(summary)

    assert "# Rapthor Benchmark Baseline" in report
    assert "| quick-demo | 2 | 0, 0 | 3.000 | 2.000-4.000 | 2.000 |" in report
    assert "| rich-demo | 1 | 1 | 8.000 | 8.000-8.000 | 6.000 |" in report


def test_write_summary_artifacts(tmp_path):
    output_json = tmp_path / "summary.json"
    output_markdown = tmp_path / "report.md"

    summary = write_summary_artifacts(
        [BenchmarkRunResult("quick-demo", 1, "/runs/1", 0, 4.0, 2, 3.0)],
        output_json=output_json,
        output_markdown=output_markdown,
    )

    assert json.loads(output_json.read_text()) == summary
    assert output_markdown.read_text().startswith("# Rapthor Benchmark Baseline")


def test_run_scenario_once_writes_command_and_result_artifacts(tmp_path):
    scenario = benchmark_scenarios_by_id()["quick-demo"]
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

    run_dir = tmp_path / "runs" / "quick-demo" / "rep-01"
    assert result.wall_seconds == 4.5
    assert result.command_seconds == 2.0
    assert (run_dir / "benchmark-command.json").is_file()
    assert json.loads((run_dir / "benchmark-result.json").read_text())["returncode"] == 0

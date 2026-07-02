"""Tests for the Rapthor command-line entry point."""

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pytest

from rapthor import cli


@dataclass
class FakeRuntimePlan:
    execution_config: object


def test_main_runs_pipeline_with_default_logging(monkeypatch):
    calls = []

    def fake_run_pipeline(parset_file, *, logging_level):
        calls.append((parset_file, logging_level))

    monkeypatch.setattr(cli, "_run_pipeline", fake_run_pipeline)

    assert cli.main(["input.parset"]) == 0
    assert calls == [("input.parset", "info")]


@pytest.mark.parametrize(
    ("option", "logging_level"),
    [
        ("-q", "warning"),
        ("-v", "debug"),
    ],
)
def test_main_applies_logging_options(monkeypatch, option, logging_level):
    calls = []

    def fake_run_pipeline(parset_file, *, logging_level):
        calls.append((parset_file, logging_level))

    monkeypatch.setattr(cli, "_run_pipeline", fake_run_pipeline)

    assert cli.main([option, "input.parset"]) == 0
    assert calls == [("input.parset", logging_level)]


def test_main_runs_state_reset(monkeypatch):
    calls = []

    monkeypatch.setattr(cli.modifystate, "run", calls.append)

    assert cli.main(["-r", "input.parset"]) == 0
    assert calls == ["input.parset"]


def test_run_pipeline_bootstraps_runtime(monkeypatch):
    calls = []

    class FakeParset:
        def __init__(self, parset_file):
            self.parset_file = parset_file

        def as_parset_dict(self):
            return {"cluster_specific": {"prefect_task_runner": "sync"}}

    @contextmanager
    def fake_bootstrapped_runtime(execution_config):
        calls.append(("bootstrap", execution_config))
        yield FakeRuntimePlan(execution_config)

    def fake_pipeline_flow(parset_file, *, logging_level, execution_config):
        calls.append(("pipeline", parset_file, logging_level, execution_config))

    def fake_materialize_parset_paths(source_parset, output_parset):
        calls.append(("materialize", source_parset, output_parset))
        return output_parset

    monkeypatch.setattr("rapthor.lib.parset.Parset", FakeParset)
    monkeypatch.setattr(
        "rapthor.lib.parset_paths.materialize_parset_paths",
        fake_materialize_parset_paths,
    )
    monkeypatch.setattr(
        "rapthor.execution.runtime_bootstrap.bootstrapped_runtime",
        fake_bootstrapped_runtime,
    )
    monkeypatch.setattr("rapthor.execution.pipeline.flow.pipeline_flow", fake_pipeline_flow)

    cli._run_pipeline("input.parset", logging_level="debug")

    assert calls[0][0] == "materialize"
    assert calls[0][1] == Path("input.parset")
    bootstrap_config = calls[1][1]
    assert bootstrap_config.task_runner == "sync"
    materialized_parset = calls[0][2]
    assert materialized_parset.name == "input.materialized.parset"
    assert calls[1:] == [
        ("bootstrap", bootstrap_config),
        ("pipeline", str(materialized_parset), "debug", bootstrap_config),
    ]


def test_main_returns_error_when_pipeline_fails(monkeypatch):
    def fail_pipeline(parset_file, *, logging_level):
        raise RuntimeError("boom")

    monkeypatch.setattr(cli, "_run_pipeline", fail_pipeline)

    assert cli.main(["input.parset"]) == 1


def test_main_prints_help_without_parset(capsys):
    assert cli.main([]) == 0

    output = capsys.readouterr().out
    assert "Usage:" in output
    assert "<parset>" in output

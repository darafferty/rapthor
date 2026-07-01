"""Tests for the Rapthor command-line entry point."""

import pytest

from rapthor import cli


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

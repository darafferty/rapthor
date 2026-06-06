"""Tests for integration test helper functions."""

import configparser
import json

import pytest

from .utils import (
    collect_command_records,
    find_command_records,
    first_command_arguments,
    parse_command_arguments,
    update_parset_path,
)


def test_update_parset_path(tmp_path):
    """Test the update_parset_path helper function."""
    parset_content = """[section1]
                        param1 = value1
                        param2 = value2
                        [section2]
                        param3 = value3
                        param4 = value4"""
    parset_path = tmp_path / "test.parset"
    parset_path.write_text(parset_content)
    updated_parset_path = update_parset_path(
        parset_path, {"param1": "new_value1", "param3": "new_value3"}
    )
    updated_parset = configparser.ConfigParser()
    updated_parset.read(updated_parset_path)
    assert updated_parset["section1"]["param1"] == "new_value1"
    assert updated_parset["section1"]["param2"] == "value2"
    assert updated_parset["section2"]["param3"] == "new_value3"
    assert updated_parset["section2"]["param4"] == "value4"


def test_update_parset_path_missing_param(tmp_path):
    """Test the update_parset_path helper function with a missing parameter."""
    parset_content = """[section1]
                        param1 = value1
                        param2 = value2
                        [section2]
                        param3 = value3
                        param4 = value4"""
    parset_path = tmp_path / "test.parset"
    parset_path.write_text(parset_content)
    with pytest.raises(ValueError, match="Parameters .* not found in parset."):
        update_parset_path(parset_path, {"param1": "new_value1", "param5": "new_value5"})


def test_collect_command_records_reads_prefect_jsonl(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "commands.jsonl").write_text(
        json.dumps(
            {
                "backend": "prefect",
                "operation": "calibrate_1",
                "name": "solve",
                "command": ["DP3", "msin=input.ms", "steps=[solve1]"],
                "command_string": "DP3 msin=input.ms 'steps=[solve1]'",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    records = collect_command_records(tmp_path)

    assert len(records) == 1
    assert records[0].backend == "prefect"
    assert records[0].operation == "calibrate_1"
    assert records[0].executable == "DP3"
    assert records[0].arguments["steps"] == "[solve1]"


def test_collect_command_records_extracts_legacy_shell_commands(tmp_path):
    log_dir = tmp_path / "logs" / "calibrate_1"
    log_dir.mkdir(parents=True)
    (log_dir / "pipeline.log").write_text(
        "$ DP3 \\\n"
        "  msin=input.ms \\\n"
        "  steps=[solve1] \\\n"
        "  solve1.mode=scalarphase\n"
        "completed success\n",
        encoding="utf-8",
    )

    records = find_command_records(tmp_path, operation="calibrate_1", executable="DP3")

    assert len(records) == 1
    assert records[0].backend == "legacy-log"
    assert records[0].arguments["solve1.mode"] == "scalarphase"


def test_first_command_arguments_filters_by_contains(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "commands.jsonl").write_text(
        json.dumps(
            {
                "operation": "image_1",
                "command": ["wsclean", "-restore-list", "bright.txt"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    arguments = first_command_arguments(
        tmp_path,
        operation="image_1",
        executable="wsclean",
        contains="-restore-list",
    )

    assert arguments == {}


def test_parse_command_arguments_accepts_shell_string():
    assert parse_command_arguments("DP3 msin=input.ms steps=[solve1]") == {
        "msin": "input.ms",
        "steps": "[solve1]",
    }

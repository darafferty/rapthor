import json
from pathlib import Path

from rapthor.execution.commands import command_matches_fixture, command_to_string, normalize_command

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def test_normalize_command_splits_shell_string():
    assert normalize_command("concat_ms.py input.ms --data_colname=DATA") == [
        "concat_ms.py",
        "input.ms",
        "--data_colname=DATA",
    ]


def test_command_to_string_quotes_tokens_when_needed():
    assert command_to_string(["echo", "hello world"]) == "echo 'hello world'"


def test_concatenate_command_matches_initial_reference_fixture():
    commands = json.loads((FIXTURE_DIR / "command_reference.json").read_text())
    expected = commands["concatenate"]["concat_ms_files"]

    assert command_matches_fixture(
        [
            "concat_ms.py",
            "epoch_0_input_0.ms",
            "epoch_0_input_1.ms",
            "--msout=epoch_0_concatenated.ms",
            "--concat_property=frequency",
            "--data_colname=DATA",
        ],
        expected,
    )

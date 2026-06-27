import json
from pathlib import Path

from rapthor.execution.commands import command_matches_fixture
from rapthor.execution.flows.concatenate import normalized_concatenate_command
from rapthor.lib.records import validate_output_record

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def test_command_reference_fixture_is_tokenized():
    commands = json.loads((FIXTURE_DIR / "command_reference.json").read_text())

    command = commands["concatenate"]["concat_ms_files"]

    assert command[0] == "concat_ms.py"
    assert "--concat_property=frequency" in command
    assert "--data_colname=DATA" in command
    assert command_matches_fixture(
        normalized_concatenate_command(
            ["epoch_0_input_0.ms", "epoch_0_input_1.ms"],
            "epoch_0_concatenated.ms",
            "DATA",
        ),
        command,
    )


def test_output_reference_fixture_matches_output_contract():
    outputs = json.loads((FIXTURE_DIR / "output_reference.json").read_text())

    validate_output_record(outputs["concatenate"]["concatenated_filenames"])

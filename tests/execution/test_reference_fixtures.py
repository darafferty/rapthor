import json
from pathlib import Path

from rapthor.execution.outputs import validate_output_record

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def test_initial_cwl_reference_command_fixture_is_tokenized():
    commands = json.loads((FIXTURE_DIR / "cwl_reference_commands.json").read_text())

    command = commands["concatenate"]["concat_ms_files"]

    assert command[0] == "concat_ms.py"
    assert "--concat_property=frequency" in command
    assert "--data_colname=DATA" in command


def test_initial_cwl_reference_output_fixture_matches_output_contract():
    outputs = json.loads((FIXTURE_DIR / "cwl_reference_outputs.json").read_text())

    validate_output_record(outputs["concatenate"]["concatenated_filenames"])

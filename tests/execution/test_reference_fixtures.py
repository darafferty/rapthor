import json
from pathlib import Path

from rapthor.lib.records import validate_output_record

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def test_command_reference_fixture_is_tokenized():
    commands = json.loads((FIXTURE_DIR / "command_reference.json").read_text())

    command = commands["concatenate"]["concat_ms_files"]

    assert command == [
        "DP3",
        "msin=[epoch_0_input_0.ms,epoch_0_input_1.ms]",
        "msin.datacolumn=DATA",
        "msout=epoch_0_concatenated.ms",
        "steps=[]",
        "msin.orderms=False",
        "msin.missingdata=True",
        "msout.writefullresflag=False",
        "msout.storagemanager=Dysco",
    ]


def test_output_reference_fixture_matches_output_contract():
    outputs = json.loads((FIXTURE_DIR / "output_reference.json").read_text())

    validate_output_record(outputs["concatenate"]["concatenated_filenames"])

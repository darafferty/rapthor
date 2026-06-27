import json
from pathlib import Path

from rapthor.execution.commands import (
    bool_token,
    bracketed_list_token,
    comma_join,
    command_matches_fixture,
    command_to_string,
    normalize_command,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def test_bool_token_uses_external_tool_spelling():
    assert bool_token(True) == "True"
    assert bool_token(False) == "False"


def test_comma_join_builds_one_command_token():
    assert comma_join(["model_a.ms", "model_b.ms", 3]) == "model_a.ms,model_b.ms,3"


def test_bracketed_list_token_builds_dp3_style_list():
    assert bracketed_list_token(["Dir00", "Dir01"]) == "[Dir00,Dir01]"


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

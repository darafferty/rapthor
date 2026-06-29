import json
from pathlib import Path

from rapthor.execution.commands import (
    append_flag,
    append_key_value,
    append_option_value,
    append_option_values,
    append_prefixed_value,
    bool_token,
    bracketed_list_token,
    comma_join,
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


def test_append_prefixed_value_skips_missing_optional_values():
    command = ["DP3"]

    append_prefixed_value(command, "applycal.parmdb=", None)
    append_prefixed_value(command, "applycal.steps=", "fastphase")

    assert command == ["DP3", "applycal.steps=fastphase"]


def test_append_option_value_adds_separate_tokens():
    command = ["wsclean"]

    append_option_value(command, "-name", "sector_0")

    assert command == ["wsclean", "-name", "sector_0"]


def test_append_flag_only_adds_enabled_flags():
    command = ["wsclean"]

    append_flag(command, "-multiscale", True)
    append_flag(command, "-save-source-list", False)

    assert command == ["wsclean", "-multiscale"]


def test_append_option_values_expands_list_values():
    command = ["wsclean"]

    append_option_values(
        command,
        [
            ("-name", "sector_0"),
            ("-size", [1024, 2048]),
            ("-parallel-gridding", None),
        ],
    )

    assert command == ["wsclean", "-name", "sector_0", "-size", "1024", "2048"]


def test_append_key_value_uses_dp3_bool_and_list_tokens():
    command = ["DP3"]

    append_key_value(command, "solve.onebeamperpatch", True)
    append_key_value(command, "solve.directions", ["Dir00", "Dir01"])
    append_key_value(command, "applycal.parmdb", None)

    assert command == [
        "DP3",
        "solve.onebeamperpatch=True",
        "solve.directions=[Dir00,Dir01]",
    ]


def test_append_key_value_skips_lists_containing_optional_values():
    command = ["DP3"]

    append_key_value(command, "solve.modeldatacolumns", ["MODEL_DATA", None])

    assert command == ["DP3"]


def test_normalize_command_splits_shell_string():
    assert normalize_command("DP3 msin=[input.ms] msout=output.ms") == [
        "DP3",
        "msin=[input.ms]",
        "msout=output.ms",
    ]


def test_command_to_string_quotes_tokens_when_needed():
    assert command_to_string(["echo", "hello world"]) == "echo 'hello world'"


def test_concatenate_command_matches_initial_reference_fixture():
    commands = json.loads((FIXTURE_DIR / "command_reference.json").read_text())
    expected = commands["concatenate"]["concat_ms_files"]

    assert normalize_command(
        [
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
    ) == normalize_command(expected)

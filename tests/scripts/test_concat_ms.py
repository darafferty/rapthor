import pytest

from rapthor.execution.concatenate.measurement_sets import (
    copy_measurement_set_command,
    concat_freq_command,
    concat_ms,
    concat_time_command,
    select_concatenation_command,
)
from rapthor.scripts.concat_ms import main


def test_concat_ms_copies_single_measurement_set(test_ms, tmp_path):
    output_file = tmp_path / "copied.ms"

    exit_code = concat_ms([test_ms], output_file.as_posix(), data_colname="DATA")

    assert exit_code == 0
    assert output_file.is_dir()
    assert (output_file / "table.info").read_text().startswith("Type = Measurement Set")


def test_concat_ms_rejects_existing_output_without_overwrite(test_ms, tmp_path):
    output_file = tmp_path / "existing.ms"
    output_file.mkdir()

    with pytest.raises(FileExistsError, match="overwrite=False"):
        concat_ms([test_ms], output_file.as_posix(), overwrite=False)


def test_concat_freq_command_uses_real_measurement_set_metadata(test_ms):
    command = concat_freq_command([test_ms], "DATA", "output.ms")

    assert command == [
        "DP3",
        f"msin=[{test_ms}]",
        "msin.datacolumn=DATA",
        "msout=output.ms",
        "steps=[]",
        "msin.orderms=False",
        "msin.missingdata=True",
        "msout.writefullresflag=False",
        "msout.storagemanager=Dysco",
    ]


def test_select_concatenation_command_copies_single_measurement_set(test_ms, tmp_path):
    output_file = tmp_path / "copied.ms"

    command = select_concatenation_command([test_ms], output_file.as_posix())

    assert command == copy_measurement_set_command(test_ms, output_file.as_posix())


def test_select_concatenation_command_can_choose_time_concatenation_without_metadata(tmp_path):
    output_file = tmp_path / "time.ms"

    command = select_concatenation_command(
        ["first.ms", "second.ms"],
        output_file.as_posix(),
        concat_property="time",
    )

    assert command == concat_time_command(["first.ms", "second.ms"], output_file.as_posix())


def test_concat_time_command_builds_taql_command():
    assert concat_time_command(["first.ms", "second.ms"], "output.ms") == [
        "taql",
        "select",
        "from",
        "[first.ms,second.ms]",
        "giving",
        "output.ms",
        "AS",
        "PLAIN",
    ]


def test_main_copies_single_measurement_set(test_ms, tmp_path, monkeypatch):
    direct_output = tmp_path / "direct.ms"
    output_file = tmp_path / "cli.ms"

    assert concat_ms([test_ms], direct_output.as_posix(), data_colname="DATA") == 0
    monkeypatch.setattr("sys.argv", ["concat_ms.py", test_ms, "--msout", output_file.as_posix()])

    assert main() == 0
    assert output_file.is_dir()
    assert (output_file / "table.info").read_text() == (direct_output / "table.info").read_text()

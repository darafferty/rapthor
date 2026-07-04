import pytest

from rapthor.execution.concatenate import linc_cli, measurement_sets
from rapthor.execution.concatenate.measurement_sets import (
    concat_freq_command,
    concat_linc_measurement_sets,
    concat_ms,
    concat_time_command,
    copy_measurement_set_command,
    linc_measurement_sets,
    select_concatenation_command,
)


class FakeTable:
    def __init__(self, times):
        self.times = times

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def getcol(self, column):
        assert column == "TIME"
        return self.times


def _patch_ms_times(monkeypatch, start_times):
    def fake_table(path, ack=False):
        assert ack is False
        return FakeTable([start_times[path]])

    monkeypatch.setattr(measurement_sets.pt, "table", fake_table)


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


def test_select_concatenation_command_can_choose_time_concatenation(monkeypatch, tmp_path):
    output_file = tmp_path / "time.ms"
    start_times = {"first.ms": 2.0, "second.ms": 1.0}
    _patch_ms_times(monkeypatch, start_times)

    command = select_concatenation_command(
        ["first.ms", "second.ms"],
        output_file.as_posix(),
        concat_property="time",
    )

    assert command == concat_time_command(["first.ms", "second.ms"], output_file.as_posix())


def test_concat_time_command_orders_measurement_sets_by_start_time(monkeypatch):
    start_times = {"/data/late.ms": 20.0, "/data/early.ms": 10.0}
    _patch_ms_times(monkeypatch, start_times)

    assert concat_time_command(["/data/late.ms", "/data/early.ms"], "/work/output.ms") == [
        "taql",
        "select",
        "from",
        '["/data/early.ms","/data/late.ms"]',
        "giving",
        '"/work/output.ms"',
        "AS",
        "PLAIN",
    ]


def test_linc_measurement_sets_finds_lower_and_upper_case_ms_dirs(tmp_path):
    lower = tmp_path / "lower.ms"
    upper = tmp_path / "upper.MS"
    ignored = tmp_path / "not-a-measurement-set.txt"
    lower.mkdir()
    upper.mkdir()
    ignored.write_text("ignore me")

    assert linc_measurement_sets(tmp_path) == sorted([lower.as_posix(), upper.as_posix()])


def test_concat_linc_measurement_sets_delegates_to_concat_ms(monkeypatch, tmp_path):
    lower = tmp_path / "lower.ms"
    upper = tmp_path / "upper.MS"
    lower.mkdir()
    upper.mkdir()
    calls = []

    def fake_concat_ms(msfiles, output_file, overwrite=False):
        calls.append((msfiles, output_file, overwrite))
        return 3

    monkeypatch.setattr(measurement_sets, "concat_ms", fake_concat_ms)

    exit_code = concat_linc_measurement_sets(tmp_path, tmp_path / "output.ms", overwrite=True)

    assert exit_code == 3
    assert calls == [
        (
            sorted([lower.as_posix(), upper.as_posix()]),
            (tmp_path / "output.ms").as_posix(),
            True,
        )
    ]


def test_concat_linc_cli_passes_arguments_to_execution_helper(monkeypatch):
    calls = []

    def fake_concat_linc_measurement_sets(input_path, output_file, overwrite=False):
        calls.append((input_path, output_file, overwrite))
        return 4

    monkeypatch.setattr(
        linc_cli,
        "concat_linc_measurement_sets",
        fake_concat_linc_measurement_sets,
    )

    exit_code = linc_cli.main(["/input", "/output.ms", "--overwrite"])

    assert exit_code == 4
    assert calls == [("/input", "/output.ms", True)]

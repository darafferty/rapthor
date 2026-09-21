from types import SimpleNamespace

import numpy as np
import psutil
import pytest

import rapthor.execution.predict.measurement_sets as ms_helpers


class FakeTable:
    def __init__(self, columns, nrows=None):
        self.columns = {name: np.asarray(value) for name, value in columns.items()}
        self._nrows = nrows if nrows is not None else len(next(iter(self.columns.values())))
        self.closed = False

    def getcol(self, name, startrow=0, nrow=None):
        values = self.columns[name]
        if nrow is None:
            return values[startrow:]
        return values[startrow : startrow + nrow]

    def nrows(self):
        return self._nrows

    def close(self):
        self.closed = True


@pytest.mark.parametrize(
    "compressed, fraction, expected", [(False, 1.0, 2), (True, 1.0, 8), (True, 0.5, 4)]
)
def test_predict_chunk_count_uses_available_ram(monkeypatch, compressed, fraction, expected):
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=1000 * 1024**2))
    monkeypatch.setattr(
        ms_helpers.subprocess,
        "check_output",
        lambda command: b"100 input.ms\n",
    )

    assert (
        ms_helpers.predict_chunk_count(
            "input.ms", 2, scale_factor=4.0, compressed=compressed, fraction=fraction
        )
        == expected
    )


@pytest.mark.parametrize("available_mb, budget_mb, expected", [(1000, 200, 8), (100, 200, 16)])
def test_predict_chunk_count_uses_smaller_of_available_ram_and_task_budget(
    monkeypatch, available_mb, budget_mb, expected
):
    monkeypatch.setattr(
        psutil, "virtual_memory", lambda: SimpleNamespace(available=available_mb * 1024**2)
    )
    monkeypatch.setattr(ms_helpers.subprocess, "check_output", lambda command: b"100 input.ms\n")
    assert (
        ms_helpers.predict_chunk_count("input.ms", 2, memory_budget_bytes=budget_mb * 1024**2)
        == expected
    )


@pytest.mark.parametrize("budget", [0, -1])
def test_predict_chunk_count_rejects_nonpositive_budget(budget):
    with pytest.raises(ValueError, match="memory_budget_bytes"):
        ms_helpers.predict_chunk_count("input.ms", 2, memory_budget_bytes=budget)


def test_predict_chunk_count_rejects_exhausted_ram(monkeypatch):
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=0))
    with pytest.raises(MemoryError, match="predict post-processing"):
        ms_helpers.predict_chunk_count("input.ms", 2)


def test_select_models_for_starttime_filters_models_and_closes_tables(monkeypatch):
    tables = {
        "model_a.ms": FakeTable({"TIME": [100.2, 100.2]}, nrows=10),
        "model_b.ms": FakeTable({"TIME": [102.5, 102.5]}, nrows=10),
        "model_c.ms": FakeTable({"TIME": [100.1, 100.1]}, nrows=10),
    }
    monkeypatch.setattr(ms_helpers.pt, "table", lambda path, **kwargs: tables[path])
    monkeypatch.setattr(ms_helpers.misc, "convert_mvt2mjd", lambda starttime: 100.0)
    monkeypatch.setattr(ms_helpers.misc, "convert_mjd2mvt", lambda value: f"mvt-{value}")

    selection = ms_helpers.select_models_for_starttime(
        ["model_a.ms", "model_b.ms", "model_c.ms"],
        "requested-start",
    )

    assert selection.paths == ["model_a.ms", "model_c.ms"]
    assert selection.nrows == [10, 10]
    assert selection.starttime_exact == "mvt-100.1"
    assert all(table.closed for table in tables.values())


def test_select_models_for_starttime_rejects_mismatched_model_rows(monkeypatch):
    tables = {
        "model_a.ms": FakeTable({"TIME": [100.0]}, nrows=10),
        "model_b.ms": FakeTable({"TIME": [100.0]}, nrows=12),
    }
    monkeypatch.setattr(ms_helpers.pt, "table", lambda path, **kwargs: tables[path])
    monkeypatch.setattr(ms_helpers.misc, "convert_mvt2mjd", lambda starttime: 100.0)
    monkeypatch.setattr(ms_helpers.misc, "convert_mjd2mvt", lambda value: f"mvt-{value}")

    with pytest.raises(RuntimeError, match="differing number of rows"):
        ms_helpers.select_models_for_starttime(["model_a.ms", "model_b.ms"], "requested-start")

    assert all(table.closed for table in tables.values())


def test_select_models_for_frequency_keeps_matching_spectral_windows(monkeypatch):
    tables = {
        "input.ms::SPECTRAL_WINDOW": FakeTable({"CHAN_FREQ": [[1.0, 2.0, 3.0]]}),
        "model_a.ms::SPECTRAL_WINDOW": FakeTable({"CHAN_FREQ": [[1.0, 2.0, 3.0]]}),
        "model_b.ms::SPECTRAL_WINDOW": FakeTable({"CHAN_FREQ": [[4.0, 5.0, 6.0]]}),
    }
    monkeypatch.setattr(ms_helpers.pt, "table", lambda path, **kwargs: tables[path])

    selected = ms_helpers.select_models_for_frequency(
        "input.ms",
        ["model_a.ms", "model_b.ms"],
        spectral_window_separator="::",
    )

    assert selected == ["model_a.ms"]
    assert all(table.closed for table in tables.values())


def test_input_rows_for_models_uses_full_input_without_starttime():
    table = FakeTable({"TIME": [10.0, 10.0, 20.0, 20.0, 30.0, 30.0]})

    rows = ms_helpers.input_rows_for_models(
        table,
        starttime=None,
        starttime_exact=None,
        model_nrows=[],
    )

    assert rows == ms_helpers.InputRows(startrow=0, nrows=6, baseline_rows=2)


def test_input_rows_for_models_finds_starttime_row_range(monkeypatch):
    table = FakeTable({"TIME": [100.0, 100.0, 200.0, 200.0, 300.0, 300.0]})
    monkeypatch.setattr(ms_helpers.misc, "convert_mvt2mjd", lambda starttime: 300.0)
    monkeypatch.setattr(ms_helpers.misc, "convert_mjd2mvt", lambda value: f"mvt-{int(value)}")

    rows = ms_helpers.input_rows_for_models(
        table,
        starttime="requested-start",
        starttime_exact="mvt-300",
        model_nrows=[2],
    )

    assert rows == ms_helpers.InputRows(startrow=4, nrows=2, baseline_rows=2)


def test_input_rows_for_models_requires_selected_model_rows():
    table = FakeTable({"TIME": [100.0, 100.0]})

    with pytest.raises(ValueError, match="No model data found"):
        ms_helpers.input_rows_for_models(
            table,
            starttime="requested-start",
            starttime_exact=None,
            model_nrows=[],
        )


def test_plan_row_chunks_preserves_input_and_model_offsets():
    chunks = ms_helpers.plan_row_chunks(nrows=10, nchunks=3, input_startrow=5)

    assert chunks == [
        ms_helpers.RowChunk(input_startrow=5, model_startrow=0, nrows=3),
        ms_helpers.RowChunk(input_startrow=8, model_startrow=3, nrows=3),
        ms_helpers.RowChunk(input_startrow=11, model_startrow=6, nrows=4),
    ]


def test_plan_row_chunks_aligns_to_complete_timeslots():
    chunks = ms_helpers.plan_row_chunks(
        nrows=10,
        nchunks=3,
        input_startrow=5,
        baseline_rows=2,
    )

    assert chunks == [
        ms_helpers.RowChunk(input_startrow=5, model_startrow=0, nrows=2),
        ms_helpers.RowChunk(input_startrow=7, model_startrow=2, nrows=2),
        ms_helpers.RowChunk(input_startrow=9, model_startrow=4, nrows=2),
        ms_helpers.RowChunk(input_startrow=11, model_startrow=6, nrows=2),
        ms_helpers.RowChunk(input_startrow=13, model_startrow=8, nrows=2),
    ]


@pytest.mark.parametrize("baseline_rows", [None, 1, 2, 4])
@pytest.mark.parametrize("nchunks", [1, 3, 8, 9, 100])
def test_plan_row_chunks_covers_each_row_once_without_empty_chunks(baseline_rows, nchunks):
    chunks = ms_helpers.plan_row_chunks(
        nrows=8, nchunks=nchunks, input_startrow=17, baseline_rows=baseline_rows
    )

    input_rows = []
    model_rows = []
    for chunk in chunks:
        assert chunk.nrows > 0
        input_rows.extend(range(chunk.input_startrow, chunk.input_startrow + chunk.nrows))
        model_rows.extend(range(chunk.model_startrow, chunk.model_startrow + chunk.nrows))
        if baseline_rows is not None:
            assert chunk.model_startrow % baseline_rows == 0
            assert chunk.nrows % baseline_rows == 0
    assert input_rows == list(range(17, 25))
    assert model_rows == list(range(8))


@pytest.mark.parametrize("baseline_rows", [None, 2])
def test_plan_row_chunks_returns_no_chunks_for_empty_selection(baseline_rows):
    assert ms_helpers.plan_row_chunks(nrows=0, nchunks=3, baseline_rows=baseline_rows) == []


def test_plan_row_chunks_does_not_collect_remainder_in_one_oversized_chunk():
    chunks = ms_helpers.plan_row_chunks(nrows=100, nchunks=51)

    assert len(chunks) == 51
    assert sum(chunk.nrows for chunk in chunks) == 100
    assert max(chunk.nrows for chunk in chunks) <= 2


@pytest.mark.parametrize(
    "invalid, message",
    [
        ({"nrows": -1}, "nrows"),
        ({"nchunks": 0}, "nchunks"),
        ({"nchunks": -1}, "nchunks"),
        ({"input_startrow": -1}, "input_startrow"),
        ({"baseline_rows": 0}, "baseline_rows"),
        ({"baseline_rows": -1}, "baseline_rows"),
    ],
)
def test_plan_row_chunks_rejects_invalid_ranges(invalid, message):
    arguments = {"nrows": 8, "nchunks": 3, **invalid}
    with pytest.raises(ValueError, match=message):
        ms_helpers.plan_row_chunks(**arguments)


def test_copy_measurement_set_replaces_existing_destination(monkeypatch):
    removed = []
    commands = []
    monkeypatch.setattr(ms_helpers.os.path, "exists", lambda path: path == "output.ms")
    monkeypatch.setattr(
        ms_helpers.shutil, "rmtree", lambda path, ignore_errors: removed.append(path)
    )
    monkeypatch.setattr(
        ms_helpers.subprocess, "check_call", lambda command: commands.append(command)
    )

    ms_helpers.copy_measurement_set("input.ms", "output.ms")

    assert removed == ["output.ms"]
    assert commands == [["cp", "-r", "-L", "--no-preserve=mode", "input.ms", "output.ms"]]


def test_modeldata_output_stem_removes_only_modeldata_suffix():
    assert ms_helpers.modeldata_output_stem("/work/input.ms.sector_1_modeldata") == (
        "input.ms.sector_1"
    )
    assert ms_helpers.modeldata_output_stem("/work/input.ms.sector_1") == "input.ms.sector_1"


def test_read_model_data_reads_requested_chunk_and_closes_tables(monkeypatch):
    tables = {
        "model_a.ms": FakeTable({"DATA": [0, 1, 2, 3]}),
        "model_b.ms": FakeTable({"DATA": [10, 11, 12, 13]}),
    }
    monkeypatch.setattr(ms_helpers.pt, "table", lambda path, **kwargs: tables[path])

    data = ms_helpers.read_model_data(["model_a.ms", "model_b.ms"], "DATA", startrow=1, nrows=2)

    assert [chunk.tolist() for chunk in data] == [[1, 2], [11, 12]]
    assert all(table.closed for table in tables.values())


def test_sum_model_data_returns_copy_of_sum():
    first = np.array([1, 2, 3])
    second = np.array([10, 20, 30])

    summed = ms_helpers.sum_model_data([first, second])

    assert summed.tolist() == [11, 22, 33]
    assert first.tolist() == [1, 2, 3]
    assert ms_helpers.sum_model_data([]) is None

import shutil
from unittest.mock import patch

import casacore.tables as pt
import numpy as np
import pytest

import rapthor.execution.predict.sector_model_addition as sector_model_addition
import rapthor.execution.predict.sector_model_subtraction as sector_model_subtraction
from rapthor.execution.predict.sector_model_addition import add_sector_models
from rapthor.execution.predict.sector_model_addition import get_nchunks as get_add_nchunks
from rapthor.execution.predict.sector_model_subtraction import (
    CovWeights,
    readGainFile,
    subtract_sector_models,
)
from rapthor.execution.predict.sector_model_subtraction import get_nchunks as get_subtract_nchunks
from rapthor.lib import miscellaneous as misc


def _copy_ms(source, destination):
    shutil.copytree(source, destination)
    return destination


def _write_data_column(ms_path, value):
    with pt.table(str(ms_path), readonly=False, ack=False) as table:
        data = table.getcol("DATA")
        table.putcol("DATA", np.full_like(data, value))


def _read_column(ms_path, column):
    with pt.table(str(ms_path), readonly=True, ack=False) as table:
        return table.getcol(column)


def test_add_get_nchunks(test_ms):
    with (
        patch("psutil.virtual_memory") as mock_memory,
        patch("subprocess.check_output") as mock_du,
    ):
        mock_memory.return_value.available = 7145 * 1024**2
        mock_du.return_value = b"36039\tdummy.ms\n"

        nchunks = get_add_nchunks(test_ms, nsectors=4, fraction=1.0, compressed=False)

    assert nchunks == 162


def test_add_sector_models_sums_sector_model_data_into_model_column(test_ms, tmp_path, monkeypatch):
    msin = _copy_ms(test_ms, tmp_path / "input.ms")
    model_a = _copy_ms(test_ms, tmp_path / "input.ms.sector_1_modeldata")
    model_b = _copy_ms(test_ms, tmp_path / "input.ms.sector_2_modeldata")
    _write_data_column(msin, 10.0 + 0.0j)
    _write_data_column(model_a, 2.0 + 0.0j)
    _write_data_column(model_b, 3.0 + 0.0j)
    monkeypatch.setattr(sector_model_addition, "get_nchunks", lambda *args, **kwargs: 1)

    add_sector_models(
        str(msin),
        [str(model_a), str(model_b)],
        msin_column="DATA",
        model_column="DATA",
        out_column="MODEL_DATA",
        use_compression=False,
        starttime=None,
        quiet=True,
        infix=".selfcal",
        output_dir=str(tmp_path),
    )

    output_ms = tmp_path / "input.ms.sector_1_di.ms"
    assert output_ms.is_dir()
    assert np.allclose(_read_column(output_ms, "DATA"), 10.0 + 0.0j)
    assert np.allclose(_read_column(output_ms, "MODEL_DATA"), 5.0 + 0.0j)


def test_subtract_get_nchunks(test_ms):
    with (
        patch("psutil.virtual_memory") as mock_memory,
        patch("subprocess.check_output") as mock_du,
    ):
        mock_memory.return_value.available = 7145 * 1024**2
        mock_du.return_value = b"36039\tdummy.ms\n"

        nchunks = get_subtract_nchunks(
            test_ms, nsectors=4, fraction=1.0, reweight=False, compressed=False
        )

    assert nchunks == 162


def test_subtract_sector_models_subtracts_other_sector_models(test_ms, tmp_path, monkeypatch):
    msin = _copy_ms(test_ms, tmp_path / "input.ms")
    model_a = _copy_ms(test_ms, tmp_path / "input.ms.sector_1_modeldata")
    model_b = _copy_ms(test_ms, tmp_path / "input.ms.sector_2_modeldata")
    _write_data_column(msin, 10.0 + 0.0j)
    _write_data_column(model_a, 2.0 + 0.0j)
    _write_data_column(model_b, 3.0 + 0.0j)
    monkeypatch.setattr(sector_model_subtraction, "get_nchunks", lambda *args, **kwargs: 1)

    subtract_sector_models(
        str(msin),
        [str(model_a), str(model_b)],
        msin_column="DATA",
        model_column="DATA",
        out_column="DATA",
        nr_outliers=0,
        nr_bright=0,
        use_compression=False,
        peel_outliers=False,
        peel_bright=False,
        reweight=False,
        starttime=None,
        solint_sec=60.0,
        solint_hz=0.0,
        weights_colname="CAL_WEIGHT",
        gainfile="",
        uvcut_min=80.0,
        uvcut_max=1e6,
        phaseonly=True,
        dirname=None,
        quiet=True,
        infix=".selfcal",
        output_dir=str(tmp_path),
    )

    sector_1_output = tmp_path / "input.ms.sector_1"
    sector_2_output = tmp_path / "input.ms.sector_2"
    assert sector_1_output.is_dir()
    assert sector_2_output.is_dir()
    assert np.allclose(_read_column(sector_1_output, "DATA"), 7.0 + 0.0j)
    assert np.allclose(_read_column(sector_2_output, "DATA"), 8.0 + 0.0j)


@pytest.mark.parametrize("nchunks", [1, 3, 57])
@pytest.mark.parametrize("input_column", ["DATA", "CORRECTED_DATA"])
@pytest.mark.parametrize("peeled_sources", ["outliers", "bright", "both"])
def test_peeling_preserves_subtraction_across_chunks_and_stages(
    test_ms, tmp_path, monkeypatch, nchunks, input_column, peeled_sources
):
    msin = _copy_ms(test_ms, tmp_path / "input.ms")
    with pt.table(str(msin), readonly=False, ack=False) as table:
        data = table.getcol("DATA")
        data[:] = (100 + np.arange(table.nrows()))[:, None, None]
        if input_column != "DATA":
            description = table.getcoldesc("DATA")
            description["name"] = input_column
            table.addcols(description)
        table.putcol(input_column, data)
        times = table.getcol("TIME")
    selected_rows = np.flatnonzero(np.isin(times, np.unique(times)[2:4]))
    assert selected_rows[0] > 0

    peel_outliers = peeled_sources in {"outliers", "both"}
    peel_bright = peeled_sources in {"bright", "both"}
    sector_values = [("sector_1", 2), ("sector_2", 3)]
    if peel_bright:
        sector_values.append(("bright_1", 5))
    if peel_outliers:
        sector_values.extend([("outlier_1", 7), ("outlier_2", 11)])
    models = []
    for name, value in sector_values:
        model = tmp_path / f"input.ms.slice.{name}_modeldata"
        with pt.table(str(msin), ack=False) as table:
            with table.selectrows(selected_rows.tolist()) as selection:
                copied = selection.copy(str(model), deep=True, valuecopy=True)
                copied.close()
        with pt.table(str(model), readonly=False, ack=False) as table:
            # DP3 model outputs need not contain the original input column.
            if input_column != "DATA":
                table.removecols(input_column)
            table.putcol("DATA", np.full_like(table.getcol("DATA"), value))
        models.append(str(model))
    monkeypatch.setattr(sector_model_subtraction, "get_nchunks", lambda *args, **kwargs: nchunks)

    subtract_sector_models(
        str(msin),
        models,
        msin_column=input_column,
        nr_outliers=2 if peel_outliers else 0,
        nr_bright=1 if peel_bright else 0,
        peel_outliers=peel_outliers,
        peel_bright=peel_bright,
        reweight=False,
        starttime=misc.convert_mjd2mvt(times[selected_rows[0]] - 0.01),
        solint_sec=60.0,
        solint_hz=0.0,
        infix=".slice",
        output_dir=str(tmp_path),
    )

    total_model = sum(value for _, value in sector_values)
    for name, own_model in sector_values[:2]:
        output = tmp_path / f"input.ms.slice.{name}"
        np.testing.assert_allclose(
            _read_column(output, "DATA"), data[selected_rows] - total_model + own_model
        )
        np.testing.assert_array_equal(_read_column(output, "TIME"), times[selected_rows])
    if peel_outliers:
        np.testing.assert_allclose(
            _read_column(tmp_path / "input.ms.slice_field", "DATA"), data[selected_rows] - 18
        )
    np.testing.assert_array_equal(_read_column(msin, input_column), data)


def test_cov_weights_get_nearest_frequstep_uses_channel_divisors():
    cov_weights = CovWeights.__new__(CovWeights)
    cov_weights.numchannels = 12

    assert cov_weights.get_nearest_frequstep(5.1) == 6
    assert cov_weights.get_nearest_frequstep(3.2) == 3
    assert cov_weights.freq_divisors.tolist() == [12, 6, 4, 3, 2, 1]


def test_read_gain_file_returns_unity_gains_for_phaseonly():
    nt = 10
    nchan = 1
    nbl = 1

    ant1gainarray, ant2gainarray = readGainFile(
        "unused.h5",
        None,
        nt,
        nchan,
        nbl,
        [0.0] * nt,
        1,
        "unused.ms",
        True,
        "direction",
        0,
        100,
    )

    assert np.all(ant1gainarray == 1.0)
    assert np.all(ant2gainarray == 1.0)
    assert ant1gainarray.shape == (nt * nbl, nchan)
    assert ant2gainarray.shape == (nt * nbl, nchan)

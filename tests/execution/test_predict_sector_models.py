import shutil
from unittest.mock import patch

import casacore.tables as pt
import numpy as np

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
        patch("os.popen") as mock_free,
        patch("subprocess.check_output") as mock_du,
    ):
        mock_free.return_value.readlines.return_value = [
            "               total        used        free      shared  buff/cache   available\n",
            "Mem:           15840        8695        2997        1918        6403        7145\n",
            "Swap:          20479         765       19714\n",
            "Total:         36320        9461       22711\n",
        ]
        mock_du.return_value = b"36039\tdummy.ms\n"

        nchunks = get_add_nchunks(test_ms, nsectors=4, fraction=1.0, compressed=False)

    assert nchunks == 32


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
        patch("os.popen") as mock_free,
        patch("subprocess.check_output") as mock_du,
    ):
        mock_free.return_value.readlines.return_value = [
            "               total        used        free      shared  buff/cache   available\n",
            "Mem:           15840        8695        2997        1918        6403        7145\n",
            "Swap:          20479         765       19714\n",
            "Total:         36320        9461       22711\n",
        ]
        mock_du.return_value = b"36039\tdummy.ms\n"

        nchunks = get_subtract_nchunks(
            test_ms, nsectors=4, fraction=1.0, reweight=False, compressed=False
        )

    assert nchunks == 32


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

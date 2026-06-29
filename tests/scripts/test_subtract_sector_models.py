"""
Tests for the subtract_sector_models script.
"""

import runpy
import shutil
import sys
from unittest.mock import patch

import casacore.tables as pt
import numpy as np

import rapthor.execution.predict.sector_model_subtraction as sector_model_subtraction
from rapthor.execution.predict.sector_model_subtraction import (
    CovWeights,
    get_nchunks,
    readGainFile,
    subtract_sector_models,
)


def _copy_ms(source, destination):
    shutil.copytree(source, destination)
    return destination


def _write_data_column(ms_path, value):
    with pt.table(str(ms_path), readonly=False, ack=False) as table:
        data = table.getcol("DATA")
        table.putcol("DATA", np.full_like(data, value))


def _read_data_column(ms_path):
    with pt.table(str(ms_path), readonly=True, ack=False) as table:
        return table.getcol("DATA")


def test_get_nchunks(test_ms):
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

        nchunks = get_nchunks(test_ms, nsectors=4, fraction=1.0, reweight=False, compressed=False)

    assert nchunks == 32


def test_main_subtracts_other_sector_models(test_ms, tmp_path, monkeypatch):
    msin = _copy_ms(test_ms, tmp_path / "input.ms")
    model_a = _copy_ms(test_ms, tmp_path / "input.ms.sector_1_modeldata")
    model_b = _copy_ms(test_ms, tmp_path / "input.ms.sector_2_modeldata")
    _write_data_column(msin, 10.0 + 0.0j)
    _write_data_column(model_a, 2.0 + 0.0j)
    _write_data_column(model_b, 3.0 + 0.0j)
    monkeypatch.chdir(tmp_path)
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
    )

    sector_1_output = tmp_path / "input.ms.sector_1"
    sector_2_output = tmp_path / "input.ms.sector_2"
    assert sector_1_output.is_dir()
    assert sector_2_output.is_dir()
    assert np.allclose(_read_data_column(sector_1_output), 7.0 + 0.0j)
    assert np.allclose(_read_data_column(sector_2_output), 8.0 + 0.0j)


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


def test_subtract_sector_models_cli_forwards_arguments(monkeypatch):
    calls = []

    def fake_subtract_sector_models(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(
        sector_model_subtraction,
        "subtract_sector_models",
        fake_subtract_sector_models,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "subtract_sector_models.py",
            "input.ms",
            "[model_a.ms,model_b.ms]",
            "--msin_column=CORRECTED_DATA",
            "--model_column=MODEL",
            "--out_column=SUBTRACTED",
            "--nr_outliers=1",
            "--nr_bright=2",
            "--use_compression=True",
            "--peel_outliers=True",
            "--peel_bright=True",
            "--reweight=False",
            "--starttime=123.0",
            "--solint_sec=60.0",
            "--solint_hz=1000.0",
            "--weights_colname=WEIGHT_SPECTRUM",
            "--gainfile=gains.h5",
            "--uvcut_min=100.0",
            "--uvcut_max=10000.0",
            "--phaseonly=False",
            "--dirname=Patch_0",
            "--quiet=False",
            "--infix=.selfcal",
        ],
    )

    runpy.run_module("rapthor.scripts.subtract_sector_models", run_name="__main__")

    assert calls == [
        (
            ("input.ms", ["model_a.ms", "model_b.ms"]),
            {
                "msin_column": "CORRECTED_DATA",
                "model_column": "MODEL",
                "out_column": "SUBTRACTED",
                "nr_outliers": 1,
                "nr_bright": 2,
                "use_compression": "True",
                "peel_outliers": "True",
                "peel_bright": "True",
                "reweight": "False",
                "starttime": "123.0",
                "solint_sec": 60.0,
                "solint_hz": 1000.0,
                "weights_colname": "WEIGHT_SPECTRUM",
                "gainfile": "gains.h5",
                "uvcut_min": 100.0,
                "uvcut_max": 10000.0,
                "phaseonly": "False",
                "dirname": "Patch_0",
                "quiet": "False",
                "infix": ".selfcal",
            },
        )
    ]

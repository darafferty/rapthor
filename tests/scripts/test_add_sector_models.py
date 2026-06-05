"""
Tests for the add_sector_models script.
"""

import shutil

import casacore.tables as pt
import numpy as np

from rapthor.scripts import add_sector_models
from rapthor.scripts.add_sector_models import get_nchunks, main


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


def test_get_nchunks(test_ms):
    # Test with a dummy MS file and parameters
    msin = test_ms
    nsectors = 4
    fraction = 1.0
    compressed = False

    # Mock the subprocess and os.popen calls
    import os
    import subprocess
    from unittest.mock import patch

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

        nchunks = get_nchunks(msin, nsectors, fraction, compressed)
        assert nchunks == 32, f"Expected 32 chunks, got {nchunks}"


def test_main_sums_sector_model_data_into_model_column(test_ms, tmp_path, monkeypatch):
    msin = _copy_ms(test_ms, tmp_path / "input.ms")
    model_a = _copy_ms(test_ms, tmp_path / "input.ms.sector_1_modeldata")
    model_b = _copy_ms(test_ms, tmp_path / "input.ms.sector_2_modeldata")
    _write_data_column(msin, 10.0 + 0.0j)
    _write_data_column(model_a, 2.0 + 0.0j)
    _write_data_column(model_b, 3.0 + 0.0j)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(add_sector_models, "get_nchunks", lambda *args, **kwargs: 1)

    main(
        str(msin),
        [str(model_a), str(model_b)],
        msin_column="DATA",
        model_column="DATA",
        out_column="MODEL_DATA",
        use_compression=False,
        starttime=None,
        quiet=True,
        infix=".selfcal",
    )

    output_ms = tmp_path / "input.ms.sector_1_di.ms"
    assert output_ms.is_dir()
    assert np.allclose(_read_column(output_ms, "DATA"), 10.0 + 0.0j)
    assert np.allclose(_read_column(output_ms, "MODEL_DATA"), 5.0 + 0.0j)

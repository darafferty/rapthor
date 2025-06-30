"""
Tests for the subtract_sector_models script.
"""

import pytest
from rapthor.scripts.subtract_sector_models import (CovWeights, get_nchunks,
                                                    main, readGainFile)


def test_get_nchunks(test_ms):
    # Test with a dummy MS file and parameters
    msin = test_ms
    nsectors = 4
    fraction = 1.0
    reweight = False
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

        nchunks = get_nchunks(msin, nsectors, fraction, reweight, compressed)
        assert nchunks == 32, f"Expected 32 chunks, got {nchunks}"


def test_main(test_ms):
    # # Define test parameters
    # msin = test_ms
    # model_list = ["model1.ms", "model2.ms"]
    # msin_column = "DATA"
    # model_column = "MODEL_DATA"
    # out_column = "SUBTRACTED_DATA"
    # nr_outliers = 5
    # nr_bright = 3
    # use_compression = False
    # peel_outliers = True
    # peel_bright = True
    # reweight = False
    # starttime = None
    # solint_sec = 60
    # solint_hz = 0.0
    # weights_colname = "WEIGHT"
    # gainfile = "gainfile.gain"
    # uvcut_min = 0.0
    # uvcut_max = 1000.0
    # phaseonly = False
    # dirname = "output_dir"
    # quiet = True
    # infix = "test_infix"
    # # Call the main function with test parameters
    # main(
    #     msin,
    #     model_list,
    #     msin_column,
    #     model_column,
    #     out_column,
    #     nr_outliers,
    #     nr_bright,
    #     use_compression,
    #     peel_outliers,
    #     peel_bright,
    #     reweight,
    #     starttime,
    #     solint_sec,
    #     solint_hz,
    #     weights_colname,
    #     gainfile,
    #     uvcut_min,
    #     uvcut_max,
    #     phaseonly,
    #     dirname,
    #     quiet,
    #     infix,
    # )
    pass


@pytest.fixture
def cov_weights(test_ms):
    # Create an instance of CovWeights for testing
    MSName = test_ms
    solint_sec = 60  # Example value for solution interval in seconds
    solint_hz = 0.0  # Example value for solution interval in Hz
    startrow = 0  # Starting row for the weights calculation
    nrow = 100  # Number of rows to process
    uvcut = [0, 2000]  # UV cut range
    gainfile = None  # No gain file for this test
    phaseonly = False  # Not using phase-only for this test
    dirname = None  # No specific directory for this test
    quiet = True  # Suppress output for this test
    weights = CovWeights(
        MSName,
        solint_sec,
        solint_hz,
        startrow,
        nrow,
        uvcut,
        gainfile,
        phaseonly,
        dirname,
        quiet,
    )
    yield weights


class TestCovWeights:
    def test_find_weights(self, cov_weights):
        # # Test the FindWeights method with dummy data
        # residualdata = None  # Replace with actual residual data
        # flags = None  # Replace with actual flags
        # weights = cov_weights.FindWeights(residualdata, flags)
        # # Check if weights are calculated correctly
        pass

    def test_calc_weights(self, cov_weights):
        # # Test the calcWeights method with dummy data
        # CoeffArray = None  # Replace with actual coefficient array
        # max_radius = 5000  # Example value
        # weights = cov_weights.calcWeights(CoeffArray, max_radius)
        # # Check if weights are calculated correctly
        pass


    def test_get_nearest_frequstep(self, cov_weights):
        # # Test the get_nearest_frequstep method with a dummy frequency step
        # freqstep = 100.0
        # nearest_freqstep = cov_weights.get_nearest_frequstep(freqstep)
        # # Check if the nearest frequency step is calculated correctly
        pass

def test_read_gain_file(test_ms):
    # Test the readGainFile function with dummy parameters
    gainfile = "test_gain.gain"
    ms = test_ms
    nt = 10
    nchan = 1
    nbl = 1
    tarray = [0.0] * nt
    nAnt = 1
    msname = "test_msname"
    phaseonly = False
    dirname = "output_dir"
    startrow = 0
    nrow = 100
    readGainFile(
        gainfile,
        ms,
        nt,
        nchan,
        nbl,
        tarray,
        nAnt,
        msname,
        phaseonly,
        dirname,
        startrow,
        nrow,
    )

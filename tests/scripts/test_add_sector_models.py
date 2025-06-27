"""
Tests for the add_sector_models script.
"""

from rapthor.scripts.add_sector_models import get_nchunks, main


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


def test_main(test_ms):
    # Test the main function with dummy parameters
    msin = test_ms
    msmod_list = [test_ms, test_ms]
    msin_column = "DATA"
    model_column = "DATA"
    out_column = "MODEL_DATA"
    use_compression = False
    starttime = None
    quiet = True
    infix = ""

    # We would normally check the output or state after running main.
    # Here we just ensure it runs without error.
    main(
        msin,
        msmod_list,
        msin_column,
        model_column,
        out_column,
        use_compression,
        starttime,
        quiet,
        infix,
    )

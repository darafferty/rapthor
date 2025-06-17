"""
Tests for the add_sector_models script.
"""

from pathlib import Path

from rapthor.scripts.add_sector_models import get_nchunks, main

RESOURCE_DIR = Path(__file__).parent / ".." / "resources"
TEST_MS = (RESOURCE_DIR / "test.ms").as_posix()


def test_get_nchunks():
    # Test with a dummy MS file and parameters
    msin = TEST_MS
    nsectors = 4
    fraction = 1.0
    compressed = False

    # Mock the subprocess and os.popen calls
    import os
    import subprocess
    from unittest.mock import patch

    with (
        patch("os.popen") as mock_popen,
        patch("subprocess.check_output") as mock_check_output,
    ):
        mock_popen.return_value.readlines.return_value = [
            "               total        used        free      shared  buff/cache   available\n",
            "Mem:           15840        8695        2997        1918        6403        7145\n",
            "Swap:          20479         765       19714\n",
            "Total:         36320        9461       22711\n",
        ]
        mock_check_output.return_value = b"36039\tdummy.ms\n"

        nchunks = get_nchunks(msin, nsectors, fraction, compressed)
        assert nchunks == 32, f"Expected 32 chunks, got {nchunks}"


def test_main():
    # Test the main function with dummy parameters
    msin = TEST_MS
    msmod_list = [TEST_MS, TEST_MS]
    msin_column = "DATA"
    model_column = "MODEL_DATA"
    out_column = "MODEL_DATA"
    use_compression = False
    starttime = None
    quiet = True
    infix = ""

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

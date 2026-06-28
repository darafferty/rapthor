"""
Tests for the blank_image script and image masking helper.
"""

import subprocess
import sys

import numpy as np
from astropy.io import fits

from rapthor.execution.image.masking import blank_image


def _image_data_and_header(path):
    with fits.open(path) as hdul:
        return hdul[0].data.copy(), hdul[0].header.copy()


def test_blank_image_creates_template_mask(tmp_path):
    """The helper creates a one-filled FITS template when no input image is given."""
    output_image = tmp_path / "test_blank_image.fits"

    blank_image(
        str(output_image),
        reference_ra_deg=10.684,
        reference_dec_deg=41.269,
        cellsize_deg=0.001,
        imsize="6,5",
    )

    data, header = _image_data_and_header(output_image)
    assert data.shape == (1, 1, 5, 6)
    assert np.all(data == 1.0)
    assert header["CRVAL1"] == 10.684
    assert header["CRVAL2"] == 41.269
    assert header["NAXIS1"] == 6
    assert header["NAXIS2"] == 5


def test_blank_image_cli_matches_function(tmp_path):
    """The CLI wrapper and helper produce the same template mask."""
    function_output = tmp_path / "function.fits"
    cli_output = tmp_path / "cli.fits"
    kwargs = {
        "reference_ra_deg": 10.684,
        "reference_dec_deg": 41.269,
        "cellsize_deg": 0.001,
        "imsize": "6,5",
    }

    blank_image(str(function_output), **kwargs)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "rapthor.scripts.blank_image",
            str(cli_output),
            "--reference_ra_deg=10.684",
            "--reference_dec_deg=41.269",
            "--cellsize_deg=0.001",
            "--imsize=6,5",
        ],
        check=True,
    )

    function_data, function_header = _image_data_and_header(function_output)
    cli_data, cli_header = _image_data_and_header(cli_output)
    assert np.array_equal(cli_data, function_data)
    for key in ["CRVAL1", "CRVAL2", "CDELT1", "CDELT2", "NAXIS1", "NAXIS2"]:
        assert cli_header[key] == function_header[key]

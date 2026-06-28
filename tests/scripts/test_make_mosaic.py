"""
Tests for the make_mosaic script.
"""

import subprocess
import sys

from astropy.io import fits
import numpy as np

from rapthor.execution.mosaic.images import make_mosaic


def _write_image(path, data):
    header = fits.Header()
    header["NAXIS"] = 2
    header["NAXIS1"] = data.shape[1]
    header["NAXIS2"] = data.shape[0]
    header["CTYPE1"] = "RA---SIN"
    header["CTYPE2"] = "DEC--SIN"
    header["CRPIX1"] = 1.0
    header["CRPIX2"] = 1.0
    header["CRVAL1"] = 12.0
    header["CRVAL2"] = -30.0
    header["CDELT1"] = -0.01
    header["CDELT2"] = 0.01
    fits.writeto(path, data.astype(np.float32), header, overwrite=True)


def test_main_averages_finite_regridded_images(tmp_path):
    template = tmp_path / "template.fits"
    sector_1 = tmp_path / "sector_1.fits"
    sector_2 = tmp_path / "sector_2.fits"
    output = tmp_path / "mosaic.fits"
    _write_image(template, np.zeros((4, 4), dtype=np.float32))
    _write_image(sector_1, np.full((4, 4), 2.0, dtype=np.float32))
    sector_2_data = np.full((4, 4), 4.0, dtype=np.float32)
    sector_2_data[0, 0] = np.nan
    _write_image(sector_2, sector_2_data)

    make_mosaic([str(sector_1), str(sector_2)], str(template), str(output))

    with fits.open(output) as hdul:
        assert hdul[0].data.shape == (4, 4)
        assert hdul[0].data[0, 0] == 2.0
        assert np.allclose(hdul[0].data[1:, :], 3.0)


def test_main_skip_copies_first_input_image(tmp_path):
    template = tmp_path / "template.fits"
    sector_1 = tmp_path / "sector_1.fits"
    sector_2 = tmp_path / "sector_2.fits"
    output = tmp_path / "mosaic.fits"
    _write_image(template, np.zeros((2, 2), dtype=np.float32))
    _write_image(sector_1, np.full((2, 2), 7.0, dtype=np.float32))
    _write_image(sector_2, np.full((2, 2), 9.0, dtype=np.float32))

    make_mosaic([str(sector_1), str(sector_2)], str(template), str(output), skip=True)

    with fits.open(output) as hdul:
        assert np.allclose(hdul[0].data, 7.0)


def test_make_mosaic_cli_matches_function(tmp_path):
    template = tmp_path / "template.fits"
    sector_1 = tmp_path / "sector_1.fits"
    sector_2 = tmp_path / "sector_2.fits"
    function_output = tmp_path / "function_mosaic.fits"
    cli_output = tmp_path / "cli_mosaic.fits"
    _write_image(template, np.zeros((4, 4), dtype=np.float32))
    _write_image(sector_1, np.full((4, 4), 2.0, dtype=np.float32))
    sector_2_data = np.full((4, 4), 4.0, dtype=np.float32)
    sector_2_data[0, 0] = np.nan
    _write_image(sector_2, sector_2_data)

    make_mosaic([str(sector_1), str(sector_2)], str(template), str(function_output))
    subprocess.run(
        [
            sys.executable,
            "-m",
            "rapthor.scripts.make_mosaic",
            f"{sector_1},{sector_2}",
            str(template),
            str(cli_output),
            "--skip=False",
        ],
        check=True,
    )

    with fits.open(function_output) as function_hdul, fits.open(cli_output) as cli_hdul:
        assert np.array_equal(cli_hdul[0].data, function_hdul[0].data)
        assert cli_hdul[0].header["NAXIS1"] == function_hdul[0].header["NAXIS1"]
        assert cli_hdul[0].header["NAXIS2"] == function_hdul[0].header["NAXIS2"]

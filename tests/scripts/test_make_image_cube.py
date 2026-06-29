"""
Tests for the make_image_cube script.
"""

import subprocess
import sys

from astropy.io import fits
import numpy as np

from rapthor.execution.image.cubes import make_image_cube


def _write_channel_image(path, frequency_hz, value, beam):
    header = fits.Header()
    header["NAXIS"] = 2
    header["NAXIS1"] = 3
    header["NAXIS2"] = 2
    header["CTYPE1"] = "RA---SIN"
    header["CTYPE2"] = "DEC--SIN"
    header["CRPIX1"] = 1.0
    header["CRPIX2"] = 1.0
    header["CRVAL1"] = 12.0
    header["CRVAL2"] = -30.0
    header["CDELT1"] = -0.01
    header["CDELT2"] = 0.01
    header["RESTFREQ"] = frequency_hz
    header["BMAJ"], header["BMIN"], header["BPA"] = beam
    fits.writeto(path, np.full((2, 3), value, dtype=np.float32), header, overwrite=True)


def test_main_builds_frequency_ordered_cube_and_metadata_files(tmp_path):
    high_channel = tmp_path / "high.fits"
    low_channel = tmp_path / "low.fits"
    output_cube = tmp_path / "cube.fits"
    output_beams = tmp_path / "cube_beams.txt"
    output_frequencies = tmp_path / "cube_frequencies.txt"
    _write_channel_image(high_channel, 150.0e6, 2.0, (0.02, 0.01, 45.0))
    _write_channel_image(low_channel, 140.0e6, 1.0, (0.03, 0.015, 35.0))

    make_image_cube(
        [str(high_channel), str(low_channel)],
        str(output_cube),
        output_beams_filename=str(output_beams),
        output_frequencies_filename=str(output_frequencies),
    )

    with fits.open(output_cube) as hdul:
        assert hdul[0].data.shape == (2, 2, 3)
        assert np.allclose(hdul[0].data[0], 1.0)
        assert np.allclose(hdul[0].data[1], 2.0)
        assert hdul[0].header["CTYPE3"] == "FREQ"
        assert hdul[0].header["CRVAL3"] == 140.0e6
        assert hdul[0].header["CDELT3"] == 10.0e6

    assert output_beams.read_text() == "(0.03, 0.015, 35.0), (0.02, 0.01, 45.0)"
    assert output_frequencies.read_text() == "140000000.0, 150000000.0"


def test_main_uses_default_metadata_filenames(tmp_path):
    channel = tmp_path / "channel.fits"
    output_cube = tmp_path / "cube.fits"
    _write_channel_image(channel, 140.0e6, 1.0, (0.03, 0.015, 35.0))

    make_image_cube([str(channel)], str(output_cube))

    assert output_cube.is_file()
    assert (tmp_path / "cube.fits_beams.txt").read_text() == "(0.03, 0.015, 35.0)"
    assert (tmp_path / "cube.fits_frequencies.txt").read_text() == "140000000.0"


def test_make_image_cube_cli_matches_function(tmp_path):
    high_channel = tmp_path / "high.fits"
    low_channel = tmp_path / "low.fits"
    function_cube = tmp_path / "function_cube.fits"
    function_beams = tmp_path / "function_cube_beams.txt"
    function_frequencies = tmp_path / "function_cube_frequencies.txt"
    cli_cube = tmp_path / "cli_cube.fits"
    cli_beams = tmp_path / "cli_cube_beams.txt"
    cli_frequencies = tmp_path / "cli_cube_frequencies.txt"
    _write_channel_image(high_channel, 150.0e6, 2.0, (0.02, 0.01, 45.0))
    _write_channel_image(low_channel, 140.0e6, 1.0, (0.03, 0.015, 35.0))

    make_image_cube(
        [str(high_channel), str(low_channel)],
        str(function_cube),
        output_beams_filename=str(function_beams),
        output_frequencies_filename=str(function_frequencies),
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "rapthor.scripts.make_image_cube",
            f"{high_channel},{low_channel}",
            str(cli_cube),
            f"--output_beams_filename={cli_beams}",
            f"--output_frequencies_filename={cli_frequencies}",
        ],
        check=True,
    )

    with fits.open(function_cube) as function_hdul, fits.open(cli_cube) as cli_hdul:
        assert np.array_equal(cli_hdul[0].data, function_hdul[0].data)
        assert cli_hdul[0].header["CRVAL3"] == function_hdul[0].header["CRVAL3"]
        assert cli_hdul[0].header["CDELT3"] == function_hdul[0].header["CDELT3"]
    assert cli_beams.read_text() == function_beams.read_text()
    assert cli_frequencies.read_text() == function_frequencies.read_text()

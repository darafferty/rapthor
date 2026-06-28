"""
Tests for the make_mosaic_template script.
"""

import subprocess
import sys

from astropy.io import fits
from astropy.wcs import WCS
import numpy as np

from rapthor.execution.mosaic.images import make_mosaic_template


def _header(shape=(4, 4), crval1=12.0):
    header = fits.Header()
    header["NAXIS"] = 2
    header["NAXIS1"] = shape[1]
    header["NAXIS2"] = shape[0]
    header["CTYPE1"] = "RA---SIN"
    header["CTYPE2"] = "DEC--SIN"
    header["CRPIX1"] = 1.0
    header["CRPIX2"] = 1.0
    header["CRVAL1"] = crval1
    header["CRVAL2"] = -30.0
    header["CDELT1"] = -0.01
    header["CDELT2"] = 0.01
    header["BMAJ"] = 0.03
    header["BMIN"] = 0.02
    header["BPA"] = 45.0
    header["RESTFREQ"] = 140.0e6
    return header


def _write_image(path, data, header):
    fits.writeto(path, data.astype(np.float32), header, overwrite=True)


def _write_vertices(path, header):
    wcs = WCS(header)
    x = np.array([-0.5, header["NAXIS1"] - 0.5, header["NAXIS1"] - 0.5, -0.5])
    y = np.array([-0.5, -0.5, header["NAXIS2"] - 0.5, header["NAXIS2"] - 0.5])
    ra, dec = wcs.wcs_pix2world(x, y, 0)
    np.save(path, np.transpose([ra, dec]), allow_pickle=False)


def test_main_builds_zero_template_covering_input_images(tmp_path):
    image_1 = tmp_path / "sector_1.fits"
    image_2 = tmp_path / "sector_2.fits"
    vertices_1 = tmp_path / "sector_1_vertices.npy"
    vertices_2 = tmp_path / "sector_2_vertices.npy"
    output_image = tmp_path / "template.fits"
    header_1 = _header(crval1=12.0)
    header_2 = _header(crval1=12.01)
    _write_image(image_1, np.ones((4, 4), dtype=np.float32), header_1)
    _write_image(image_2, np.ones((4, 4), dtype=np.float32), header_2)
    _write_vertices(vertices_1, header_1)
    _write_vertices(vertices_2, header_2)

    make_mosaic_template(
        [str(image_1), str(image_2)],
        [str(vertices_1), str(vertices_2)],
        str(output_image),
        skip=False,
        padding=1.0,
    )

    with fits.open(output_image) as hdul:
        assert hdul[0].data.ndim == 2
        assert hdul[0].data.size > 0
        assert np.allclose(hdul[0].data, 0.0)
        assert hdul[0].header["ORIGIN"] == "Raptor"
        assert hdul[0].header["TELESCOP"] == "LOFAR"
        assert hdul[0].header["BMAJ"] == header_1["BMAJ"]


def test_main_skip_does_not_create_template(tmp_path):
    output_image = tmp_path / "template.fits"

    make_mosaic_template(["missing.fits"], ["missing_vertices.npy"], str(output_image), skip=True)

    assert not output_image.exists()


def test_make_mosaic_template_cli_matches_function(tmp_path):
    function_output = tmp_path / "function_template.fits"
    cli_output = tmp_path / "cli_template.fits"
    image_1 = tmp_path / "sector_1.fits"
    image_2 = tmp_path / "sector_2.fits"
    vertices_1 = tmp_path / "sector_1_vertices.npy"
    vertices_2 = tmp_path / "sector_2_vertices.npy"
    header_1 = _header(crval1=12.0)
    header_2 = _header(crval1=12.01)
    _write_image(image_1, np.ones((4, 4), dtype=np.float32), header_1)
    _write_image(image_2, np.ones((4, 4), dtype=np.float32), header_2)
    _write_vertices(vertices_1, header_1)
    _write_vertices(vertices_2, header_2)

    make_mosaic_template(
        [str(image_1), str(image_2)],
        [str(vertices_1), str(vertices_2)],
        str(function_output),
        skip=False,
        padding=1.0,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "rapthor.scripts.make_mosaic_template",
            f"{image_1},{image_2}",
            f"{vertices_1},{vertices_2}",
            str(cli_output),
            "--skip=False",
            "--padding=1.0",
        ],
        check=True,
    )

    with fits.open(function_output) as function_hdul, fits.open(cli_output) as cli_hdul:
        assert np.array_equal(cli_hdul[0].data, function_hdul[0].data)
        for key in ["CRVAL1", "CRVAL2", "CDELT1", "CDELT2", "NAXIS1", "NAXIS2"]:
            assert cli_hdul[0].header[key] == function_hdul[0].header[key]

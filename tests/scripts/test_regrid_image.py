"""
Tests for the regrid_image script.
"""

from astropy.io import fits
from astropy.wcs import WCS
import numpy as np

from rapthor.scripts.regrid_image import main


def _header(shape=(4, 4)):
    header = fits.Header()
    header["NAXIS"] = 2
    header["NAXIS1"] = shape[1]
    header["NAXIS2"] = shape[0]
    header["CTYPE1"] = "RA---SIN"
    header["CTYPE2"] = "DEC--SIN"
    header["CRPIX1"] = 1.0
    header["CRPIX2"] = 1.0
    header["CRVAL1"] = 12.0
    header["CRVAL2"] = -30.0
    header["CDELT1"] = -0.01
    header["CDELT2"] = 0.01
    header["BMAJ"] = 0.03
    header["BMIN"] = 0.02
    header["BPA"] = 45.0
    header["RESTFREQ"] = 140.0e6
    return header


def _write_image(path, data, header=None):
    fits.writeto(path, data.astype(np.float32), header or _header(data.shape), overwrite=True)


def _write_vertices(path, header):
    wcs = WCS(header)
    x = np.array([-0.5, header["NAXIS1"] - 0.5, header["NAXIS1"] - 0.5, -0.5])
    y = np.array([-0.5, -0.5, header["NAXIS2"] - 0.5, header["NAXIS2"] - 0.5])
    ra, dec = wcs.wcs_pix2world(x, y, 0)
    np.save(path, np.transpose([ra, dec]), allow_pickle=False)


def test_main_reprojects_image_to_template(tmp_path):
    input_image = tmp_path / "input.fits"
    template_image = tmp_path / "template.fits"
    vertices_file = tmp_path / "vertices.npy"
    output_image = tmp_path / "regridded.fits"
    header = _header()
    data = np.arange(16, dtype=np.float32).reshape(4, 4) + 1.0
    _write_image(input_image, data, header)
    _write_image(template_image, np.zeros((4, 4), dtype=np.float32), header)
    _write_vertices(vertices_file, header)

    main(str(input_image), str(template_image), str(vertices_file), str(output_image), skip=False)

    with fits.open(output_image) as hdul:
        assert hdul[0].data.shape == (4, 4)
        assert np.nanmax(hdul[0].data) > 0
        assert hdul[0].header["CRVAL1"] == header["CRVAL1"]
        assert hdul[0].header["CRVAL2"] == header["CRVAL2"]


def test_main_skip_copies_input_image(tmp_path):
    input_image = tmp_path / "input.fits"
    template_image = tmp_path / "template.fits"
    vertices_file = tmp_path / "vertices.npy"
    output_image = tmp_path / "regridded.fits"
    data = np.full((3, 3), 5.0, dtype=np.float32)
    _write_image(input_image, data, _header(data.shape))
    _write_image(template_image, np.zeros((3, 3), dtype=np.float32), _header(data.shape))
    _write_vertices(vertices_file, _header(data.shape))

    main(str(input_image), str(template_image), str(vertices_file), str(output_image), skip=True)

    with fits.open(output_image) as hdul:
        assert np.allclose(hdul[0].data, 5.0)

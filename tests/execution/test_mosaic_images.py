from astropy.io import fits
from astropy.wcs import WCS
import numpy as np

from rapthor.execution.mosaic.images import make_mosaic, make_mosaic_template, regrid_image


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


def _write_image(path, data, header=None):
    fits.writeto(path, data.astype(np.float32), header or _header(data.shape), overwrite=True)


def _write_vertices(path, header):
    wcs = WCS(header)
    x = np.array([-0.5, header["NAXIS1"] - 0.5, header["NAXIS1"] - 0.5, -0.5])
    y = np.array([-0.5, -0.5, header["NAXIS2"] - 0.5, header["NAXIS2"] - 0.5])
    ra, dec = wcs.wcs_pix2world(x, y, 0)
    np.save(path, np.transpose([ra, dec]), allow_pickle=False)


def test_make_mosaic_template_builds_zero_template_covering_input_images(tmp_path):
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


def test_make_mosaic_template_skip_does_not_create_template(tmp_path):
    output_image = tmp_path / "template.fits"

    make_mosaic_template(["missing.fits"], ["missing_vertices.npy"], str(output_image), skip=True)

    assert not output_image.exists()


def test_regrid_image_reprojects_image_to_template(tmp_path):
    input_image = tmp_path / "input.fits"
    template_image = tmp_path / "template.fits"
    vertices_file = tmp_path / "vertices.npy"
    output_image = tmp_path / "regridded.fits"
    header = _header()
    data = np.arange(16, dtype=np.float32).reshape(4, 4) + 1.0
    _write_image(input_image, data, header)
    _write_image(template_image, np.zeros((4, 4), dtype=np.float32), header)
    _write_vertices(vertices_file, header)

    regrid_image(str(input_image), str(template_image), str(vertices_file), str(output_image))

    with fits.open(output_image) as hdul:
        assert hdul[0].data.shape == (4, 4)
        assert np.nanmax(hdul[0].data) > 0
        assert hdul[0].header["CRVAL1"] == header["CRVAL1"]
        assert hdul[0].header["CRVAL2"] == header["CRVAL2"]


def test_regrid_image_skip_copies_input_image(tmp_path):
    input_image = tmp_path / "input.fits"
    template_image = tmp_path / "template.fits"
    vertices_file = tmp_path / "vertices.npy"
    output_image = tmp_path / "regridded.fits"
    data = np.full((3, 3), 5.0, dtype=np.float32)
    header = _header(data.shape)
    _write_image(input_image, data, header)
    _write_image(template_image, np.zeros((3, 3), dtype=np.float32), header)
    _write_vertices(vertices_file, header)

    regrid_image(
        str(input_image), str(template_image), str(vertices_file), str(output_image), skip=True
    )

    with fits.open(output_image) as hdul:
        assert np.allclose(hdul[0].data, 5.0)


def test_make_mosaic_averages_finite_regridded_images(tmp_path):
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


def test_make_mosaic_skip_copies_first_input_image(tmp_path):
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

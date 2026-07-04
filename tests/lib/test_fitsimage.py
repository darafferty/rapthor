"""
Tests for the `rapthor.lib.fitsimage` module.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits as pyfits

import rapthor.lib.fitsimage as fitsimage
from rapthor.lib.fitsimage import FITSCube, FITSImage


def _write_fits_image(path, *, data=None, frequency_hz=150e6, crval=(10.0, 20.0)):
    if data is None:
        data = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=float)

    header = pyfits.Header()
    header["CRPIX1"] = 1.0
    header["CRPIX2"] = 1.0
    header["CDELT1"] = -0.01
    header["CDELT2"] = 0.01
    header["CRVAL1"] = crval[0]
    header["CRVAL2"] = crval[1]
    header["CTYPE1"] = "RA---SIN"
    header["CTYPE2"] = "DEC--SIN"
    header["RESTFREQ"] = frequency_hz
    header["BMAJ"] = 0.1
    header["BMIN"] = 0.05
    header["BPA"] = 12.5
    pyfits.writeto(path, data, header, overwrite=True)
    return path


def test_fits_image_reads_header_beam_frequency_and_statistics(tmp_path):
    image_path = _write_fits_image(tmp_path / "image.fits")

    image = FITSImage(image_path)

    assert image.imagefile == image_path
    assert image.beam == [0.1, 0.05, 12.5]
    assert image.freq == 150e6
    assert image.img_data.shape == (2, 2)
    assert image.min_value == 0.0
    assert image.max_value == 3.0
    assert image.mean_value == pytest.approx(1.5)
    assert image.median_value == pytest.approx(1.5)


def test_fits_image_flattens_higher_dimensional_image(tmp_path):
    data = np.arange(16, dtype=float).reshape(1, 2, 2, 4)
    image_path = _write_fits_image(tmp_path / "cube-like-image.fits", data=data)

    image = FITSImage(image_path)

    assert image.img_hdr["NAXIS"] == 2
    assert image.img_data.shape == (2, 4)
    assert image.img_hdr["RESTFREQ"] == 150e6


def test_fits_image_write_get_beam_and_get_wcs(tmp_path):
    image = FITSImage(_write_fits_image(tmp_path / "image.fits"))
    output_path = tmp_path / "written.fits"

    image.write(output_path)

    with pyfits.open(output_path) as hdul:
        assert hdul[0].data.shape == (2, 2)
        assert hdul[0].header["BMAJ"] == 0.1
    assert image.get_beam() == [0.1, 0.05, 12.5]
    assert list(image.get_wcs().wcs.ctype) == ["RA---SIN", "DEC--SIN"]


def test_fits_image_blank_uses_region_vertices(monkeypatch, tmp_path):
    image = FITSImage(_write_fits_image(tmp_path / "image.fits"))
    vertices_file = tmp_path / "vertices.reg"
    calls = {}

    def fake_read_vertices_x_y(path, wcs):
        calls["vertices_file"] = path
        calls["wcs"] = wcs
        return [(0, 0), (1, 0), (1, 1), (0, 1)]

    def fake_rasterize(vertices, data, blank_value):
        calls["vertices"] = vertices
        calls["blank_value"] = blank_value
        return np.full_like(data, blank_value)

    monkeypatch.setattr(fitsimage, "read_vertices_x_y", fake_read_vertices_x_y)
    monkeypatch.setattr(fitsimage, "rasterize", fake_rasterize)

    image.blank(vertices_file)

    assert calls["vertices_file"] == vertices_file
    assert calls["blank_value"] is np.nan
    assert np.isnan(image.img_data).all()


def test_fits_image_select_facet_returns_pixels_inside_polygon(tmp_path):
    data = np.arange(25, dtype=float).reshape(5, 5)
    image = FITSImage(_write_fits_image(tmp_path / "image.fits", data=data))
    pixel_vertices = np.array(
        [
            [1.0, 1.0],
            [3.0, 1.0],
            [3.0, 3.0],
            [1.0, 3.0],
        ]
    )
    image.get_wcs = lambda: SimpleNamespace(world_to_pixel_values=lambda ra, dec: (ra, dec))
    facet = SimpleNamespace(vertices=pixel_vertices)

    selected = image.select_facet(facet)

    assert selected.shape == (3, 3)
    assert selected[np.isfinite(selected)].tolist() == [12.0]


def test_fits_image_calc_noise_shift_and_weight(tmp_path):
    image = FITSImage(
        _write_fits_image(
            tmp_path / "image.fits",
            data=np.array([[-1.0, 0.0], [1.0, 2.0]], dtype=float),
        )
    )

    image.calc_noise(sampling=1)
    image.apply_shift(dra=0.1, ddec=-0.2)
    image.calc_weight()

    assert image.noise == pytest.approx(np.std(image.img_data))
    assert image.img_hdr["CRVAL1"] == pytest.approx(10.0 + 0.1 / np.cos(np.pi * 20.0 / 180.0))
    assert image.img_hdr["CRVAL2"] == pytest.approx(19.8)
    assert image.weight_data[0, 1] == 0.0
    assert image.weight_data[0, 0] == pytest.approx((1 / image.noise) ** 2)


def test_fits_cube_orders_channels_and_builds_data(tmp_path):
    high = _write_fits_image(tmp_path / "high.fits", frequency_hz=160e6)
    low = _write_fits_image(tmp_path / "low.fits", frequency_hz=140e6)

    cube = FITSCube([high, low])

    assert cube.channel_imagefiles == [low, high]
    assert cube.channel_frequencies.tolist() == [140e6, 160e6]
    assert cube.header["NAXIS"] == 3
    assert cube.header["NAXIS3"] == 2
    assert cube.header["CRVAL3"] == 140e6
    assert cube.header["CDELT3"] == 20e6
    assert cube.data.shape == (2, 2, 2)


def test_fits_cube_rejects_mismatched_channel_shapes(tmp_path):
    first = _write_fits_image(tmp_path / "first.fits", data=np.ones((2, 2)))
    second = _write_fits_image(tmp_path / "second.fits", data=np.ones((3, 3)))

    with pytest.raises(ValueError, match="Data shape for channel image"):
        FITSCube([first, second])


def test_fits_cube_write_outputs(tmp_path):
    first = _write_fits_image(tmp_path / "first.fits", frequency_hz=140e6)
    second = _write_fits_image(tmp_path / "second.fits", frequency_hz=160e6)
    cube = FITSCube([first, second])

    cube_path = tmp_path / "cube.fits"
    frequencies_path = tmp_path / "frequencies.txt"
    beams_path = tmp_path / "beams.txt"

    cube.write(cube_path)
    cube.write_frequencies(frequencies_path)
    cube.write_beams(beams_path)

    with pyfits.open(cube_path) as hdul:
        assert hdul[0].data.shape == (2, 2, 2)
        assert hdul[0].header["CTYPE3"] == "FREQ"
    assert frequencies_path.read_text() == "140000000.0, 160000000.0"
    assert beams_path.read_text() == "(0.1, 0.05, 12.5), (0.1, 0.05, 12.5)"

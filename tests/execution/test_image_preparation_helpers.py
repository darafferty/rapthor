from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from rapthor.execution.image.beam import ensure_image_beam
from rapthor.execution.image.masking import blank_image
from rapthor.execution.regions import make_ds9_region_from_skymodel

RESOURCE_DIR = Path(__file__).resolve().parents[1] / "resources"


def _image_data_and_header(path):
    with fits.open(path) as hdul:
        return hdul[0].data.copy(), hdul[0].header.copy()


def _write_image(path, *, bmaj=None, bmin=None, bpa=None):
    header = fits.Header()
    if bmaj is not None:
        header["BMAJ"] = bmaj
    if bmin is not None:
        header["BMIN"] = bmin
    if bpa is not None:
        header["BPA"] = bpa
    fits.PrimaryHDU(data=np.zeros((2, 2)), header=header).writeto(path)


def _beam_header(path):
    with fits.open(path) as hdul:
        header = hdul[0].header
        return header["BMAJ"], header["BMIN"], header["BPA"]


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


def test_ensure_image_beam_adds_missing_beam(tmp_path):
    image = tmp_path / "image.fits"
    _write_image(image)

    ensure_image_beam(str(image), 12.0)

    bmaj, bmin, bpa = _beam_header(image)
    assert bmaj == pytest.approx(12.0 / 3600)
    assert bmin == pytest.approx(12.0 / 3600)
    assert bpa == 0.0


def test_ensure_image_beam_uses_default_for_invalid_fallback_size(tmp_path):
    image = tmp_path / "image.fits"
    _write_image(image, bmaj=0.0, bmin=-1.0)

    ensure_image_beam(str(image), 0.0)

    bmaj, bmin, _ = _beam_header(image)
    assert bmaj == pytest.approx(10.0 / 3600)
    assert bmin == pytest.approx(10.0 / 3600)


def test_ensure_image_beam_preserves_valid_values(tmp_path):
    image = tmp_path / "image.fits"
    _write_image(image, bmaj=0.03, bmin=0.02, bpa=45.0)

    ensure_image_beam(str(image), 12.0)

    assert _beam_header(image) == (0.03, 0.02, 45.0)


def test_make_ds9_region_from_skymodel_writes_regions(tmp_path):
    skymodel = RESOURCE_DIR / "test_true_sky.txt"
    region_file = tmp_path / "region.reg"

    make_ds9_region_from_skymodel(
        str(skymodel),
        258.0,
        57.5,
        3.0,
        2.0,
        str(region_file),
        enclose_names=False,
    )

    region_text = region_file.read_text()
    assert region_text.startswith("# Region file format: DS9 version 4.0")
    assert "polygon(" in region_text
    assert "point(" in region_text
    assert "Patch_patch_1" in region_text

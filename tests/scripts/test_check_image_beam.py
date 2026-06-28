"""Tests for the check_image_beam script and image beam helper."""

import subprocess
import sys

import numpy as np
import pytest
from astropy.io import fits

from rapthor.execution.image.beam import ensure_image_beam


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


def test_check_image_beam_cli_matches_function(tmp_path):
    function_image = tmp_path / "function.fits"
    cli_image = tmp_path / "cli.fits"
    _write_image(function_image)
    _write_image(cli_image)

    ensure_image_beam(str(function_image), 15.0)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "rapthor.scripts.check_image_beam",
            str(cli_image),
            "15.0",
        ],
        check=True,
    )

    assert _beam_header(cli_image) == pytest.approx(_beam_header(function_image))

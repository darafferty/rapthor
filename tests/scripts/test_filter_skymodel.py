"""
Test script for filter_skymodel.py
"""

import json
import runpy
import sys

import numpy as np
from astropy.io import fits
from astropy.table import Table

import rapthor.execution.image.skymodel_filter as skymodel_filter_module
from rapthor.execution.image.skymodel_filter import filter_image_skymodel


def test_filter_skymodel_writes_empty_outputs_for_all_blank_image(tmp_path, monkeypatch):
    """All-blank images should produce valid empty outputs instead of failing."""

    def raise_all_blank(*args, **kwargs):
        raise RuntimeError("All pixels in the image are blanked.")

    monkeypatch.setattr(skymodel_filter_module, "lsmtool_filter_skymodel", raise_all_blank)

    flat_noise_image = tmp_path / "flat_noise.fits"
    true_sky_image = tmp_path / "true_sky.fits"
    true_sky_skymodel = tmp_path / "true_sky.txt"
    apparent_sky_skymodel = tmp_path / "apparent_sky.txt"
    output_root = tmp_path / "sector_1"

    fits.writeto(flat_noise_image, np.zeros((2, 2)), overwrite=True)
    fits.writeto(true_sky_image, np.zeros((2, 2)), overwrite=True)
    true_sky_skymodel.write_text("FORMAT = Name, Type, Ra, Dec, I\n")
    apparent_sky_skymodel.write_text("FORMAT = Name, Type, Ra, Dec, I\n")

    filter_image_skymodel(
        str(flat_noise_image),
        str(true_sky_image),
        str(true_sky_skymodel),
        str(apparent_sky_skymodel),
        str(output_root),
        str(tmp_path / "vertices.npy"),
        [],
    )

    assert (tmp_path / "sector_1.true_sky.txt").read_text() == "FORMAT = Name, Type, Ra, Dec, I\n"
    assert (tmp_path / "sector_1.apparent_sky.txt").read_text() == (
        "FORMAT = Name, Type, Ra, Dec, I\n"
    )
    assert fits.getdata(tmp_path / "sector_1.flat_noise_rms.fits").shape == (2, 2)
    assert fits.getdata(tmp_path / "sector_1.true_sky_rms.fits").shape == (2, 2)
    assert len(Table.read(tmp_path / "sector_1.source_catalog.fits", format="fits")) == 0
    assert json.loads((tmp_path / "sector_1.image_diagnostics.json").read_text()) == {"nsources": 0}


def test_filter_skymodel_cli_matches_function_for_all_blank_image(tmp_path, monkeypatch):
    """The CLI wrapper and helper produce the same empty outputs for blank images."""

    def raise_all_blank(*args, **kwargs):
        raise RuntimeError("All pixels in the image are blanked.")

    monkeypatch.setattr(skymodel_filter_module, "lsmtool_filter_skymodel", raise_all_blank)

    flat_noise_image = tmp_path / "flat_noise.fits"
    true_sky_image = tmp_path / "true_sky.fits"
    true_sky_skymodel = tmp_path / "true_sky.txt"
    apparent_sky_skymodel = tmp_path / "apparent_sky.txt"
    function_output_root = tmp_path / "function_sector"
    cli_output_root = tmp_path / "cli_sector"

    fits.writeto(flat_noise_image, np.zeros((2, 2)), overwrite=True)
    fits.writeto(true_sky_image, np.zeros((2, 2)), overwrite=True)
    true_sky_skymodel.write_text("FORMAT = Name, Type, Ra, Dec, I\n")
    apparent_sky_skymodel.write_text("FORMAT = Name, Type, Ra, Dec, I\n")

    filter_image_skymodel(
        str(flat_noise_image),
        str(true_sky_image),
        str(true_sky_skymodel),
        str(apparent_sky_skymodel),
        str(function_output_root),
        str(tmp_path / "vertices.npy"),
        [],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "filter_skymodel.py",
            str(flat_noise_image),
            str(true_sky_image),
            str(true_sky_skymodel),
            str(apparent_sky_skymodel),
            str(cli_output_root),
            str(tmp_path / "vertices.npy"),
            "[]",
        ],
    )

    runpy.run_module("rapthor.scripts.filter_skymodel", run_name="__main__")

    for suffix in [
        ".true_sky.txt",
        ".apparent_sky.txt",
        ".image_diagnostics.json",
    ]:
        assert (tmp_path / f"cli_sector{suffix}").read_text() == (
            tmp_path / f"function_sector{suffix}"
        ).read_text()
    assert np.array_equal(
        fits.getdata(tmp_path / "cli_sector.flat_noise_rms.fits"),
        fits.getdata(tmp_path / "function_sector.flat_noise_rms.fits"),
    )
    assert len(Table.read(tmp_path / "cli_sector.source_catalog.fits", format="fits")) == 0

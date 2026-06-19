"""
Test script for filter_skymodel.py
"""

import json

import numpy as np
from astropy.io import fits
from astropy.table import Table

import rapthor.scripts.filter_skymodel as filter_skymodel_module
from rapthor.scripts.filter_skymodel import main


def test_filter_skymodel():
    """
    Test the main function of the filter_skymodel script.
    """
    # # Define test parameters
    # flat_noise_image = "flat_noise_image.fits"
    # true_sky_image = "true_sky_image.fits"
    # true_sky_skymodel = "true_sky.skymodel"
    # output_root = "output_root"
    # vertices_file = "vertices_file.reg"
    # beamMS = "beam.ms"
    # bright_true_sky_skymodel = "bright_true_sky.skymodel"
    # threshisl = 5.0
    # threshpix = 7.5
    # rmsbox = (150, 50)
    # rmsbox_bright = (35, 7)
    # adaptive_thresh = 75.0
    # filter_by_mask = True
    # remove_negative = False
    # ncores = 8
    # main(
    #     flat_noise_image,
    #     true_sky_image,
    #     true_sky_skymodel,
    #     output_root,
    #     vertices_file,
    #     beamMS,
    #     bright_true_sky_skymodel,
    #     threshisl,
    #     threshpix,
    #     rmsbox,
    #     rmsbox_bright,
    #     adaptive_thresh,
    #     filter_by_mask,
    #     remove_negative,
    #     ncores,
    # )
    pass


def test_filter_skymodel_writes_empty_outputs_for_all_blank_image(tmp_path, monkeypatch):
    """All-blank images should produce valid empty outputs instead of failing."""

    def raise_all_blank(*args, **kwargs):
        raise RuntimeError("All pixels in the image are blanked.")

    monkeypatch.setattr(filter_skymodel_module, "filter_skymodel", raise_all_blank)

    flat_noise_image = tmp_path / "flat_noise.fits"
    true_sky_image = tmp_path / "true_sky.fits"
    true_sky_skymodel = tmp_path / "true_sky.txt"
    apparent_sky_skymodel = tmp_path / "apparent_sky.txt"
    output_root = tmp_path / "sector_1"

    fits.writeto(flat_noise_image, np.zeros((2, 2)), overwrite=True)
    fits.writeto(true_sky_image, np.zeros((2, 2)), overwrite=True)
    true_sky_skymodel.write_text("FORMAT = Name, Type, Ra, Dec, I\n")
    apparent_sky_skymodel.write_text("FORMAT = Name, Type, Ra, Dec, I\n")

    main(
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

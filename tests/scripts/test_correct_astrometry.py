"""
Tests for the script that corrects the astrometry of an image.
"""

import json
import pytest

import astropy.io.fits as fits
import numpy as np
from rapthor.lib.miscellaneous import make_template_image
from rapthor.scripts.correct_astrometry import main, validate_corrections


@pytest.fixture
def mock_astrometry_corrections(tmp_path):
    """
    Creates a mock corrections file and returns its path.

    Corrections file should store a dict with the following keys:
       "facet_name", "meanRAOffsetDeg", "meanDecOffsetDeg",
       "stdRAOffsetDeg", and "stdDecOffsetDeg",
    where each key is a list of length nfacets
    """
    # Set the shifts
    #
    # Note: The image pixel scale is 1.5 arcsec per pixel. The offsets are defined as
    # (Rapthor value) - (comparison value), so a value of:
    #   +1.5 arcsec in RA means Rapthor source is offset by -1 pixel in x coord, and
    #   +1.5 arcsec in Dec means Rapthor source is offset by +1 pixel in y coord
    corrections_dict = {
        "facet_name": ["Patch_1", "Patch_2", "Patch_3"],
        "meanRAOffsetDeg": [1.5 / 3600, -1.5 / 3600, 1.5 / 3600],
        "meanDecOffsetDeg": [-1.5 / 3600, -1.5 / 3600, 1.5 / 3600],
        "stdRAOffsetDeg": [1 / 3600, 0.5 / 3600, 0.5 / 3600],
        "stdDecOffsetDeg": [1 / 3600, 0.5 / 3600, 0.5 / 3600],
    }

    # Write out dict to a file and return its path
    corrections_file_path = tmp_path / "test_correct_astrometry_corrections.json"
    with open(corrections_file_path, "w") as fp:
        json.dump(corrections_dict, fp)
    return corrections_file_path


@pytest.fixture
def mock_fits_image(tmp_path):
    """
    Creates a mock FITS image and returns its path.

    Note: image pixel scale is set by cellsize_deg to 1.5 arcsec per pixel
    """
    # Make a blank image (all zeros)
    image_file_path = tmp_path / "test_correct_astrometry_image.fits"
    make_template_image(
        image_file_path, 215.59, -12.73, ximsize=512, yimsize=512, cellsize_deg=1.5 / 3600
    )

    # Add single-pixel sources with values of 1 for each of the patches for later checks
    with fits.open(image_file_path, mode="update") as hdu_list:
        # Note: data axes are [STOKES, FREQ, DEC, RA]
        hdu_list[0].data[0, 0, 100, 100] = 1  # Patch_1
        hdu_list[0].data[0, 0, 200, 200] = 1  # Patch_2
        hdu_list[0].data[0, 0, 300, 300] = 1  # Patch_3
    return image_file_path


@pytest.fixture
def mock_facet_region(tmp_path):
    """
    Creates a mock ds9 facet region file and returns its path.
    """
    lines = [
        "# Region file format: DS9 version 4.0\n",
        "global color=green select=1 highlite=1 edit=1 move=1 delete=1 include=1 fixed=0 source=1\n",
        "fk5\n",
        "polygon(215.7410416,-12.7740340,215.6337593,-12.7740732,215.6337775,-12.8787002,215.7411044,-12.8786610)\n",
        "point(215.6560886,-12.7944394) # text={Patch_1}\n",
        "polygon(215.6613773,-12.7028630,215.5959426,-12.7028725,215.5959441,-12.7667056,215.6613953,-12.7666961)\n",
        "point(215.6130873,-12.7526574) # text={Patch_2}\n",
        "polygon(215.5871994,-12.6229883,215.4767905,-12.6229643,215.4767426,-12.7307045,215.5871982,-12.7307286)\n",
        "point(215.5705331,-12.7111502) # text={Patch_3}\n",
    ]
    region_file_path = tmp_path / "test_correct_astrometry_region.reg"
    with open(region_file_path, "w") as fp:
        fp.writelines(lines)
    return region_file_path


def test_validate_corrections_valid():
    """
    Test the validate_corrections function on a valid dict.
    """
    valid_corrections_dict = {
        "facet_name": ["Patch_1", "Patch_2", "Patch_3"],
        "meanRAOffsetDeg": [1 / 3600, -2 / 3600, 0.5 / 3600],
        "meanDecOffsetDeg": [-0.5 / 3600, -1.5 / 3600, 1 / 3600],
        "stdRAOffsetDeg": [1 / 3600, 0.5 / 3600, 0.5 / 3600],
        "stdDecOffsetDeg": [1 / 3600, 0.5 / 3600, 0.5 / 3600],
    }
    validate_corrections(valid_corrections_dict)


def test_validate_corrections_inconsistent_lengths():
    """
    Test the validate_corrections function on a dict with values of inconsistent lengths.
    """
    invalid_corrections_dict = {
        "facet_name": ["Patch_1", "Patch_2", "Patch_3"],
        "meanRAOffsetDeg": [1 / 3600, -2 / 3600],
        "meanDecOffsetDeg": [-0.5 / 3600, -1.5 / 3600, 1 / 3600],
        "stdRAOffsetDeg": [1 / 3600, 0.5 / 3600, 0.5 / 3600],
        "stdDecOffsetDeg": [1 / 3600, 0.5 / 3600, 0.5 / 3600],
    }
    with pytest.raises(ValueError, match="Corrections should have equal length"):
        validate_corrections(invalid_corrections_dict)


def test_validate_corrections_missing_key():
    """
    Test the validate_corrections function on a dict that's missing a required key.
    """
    invalid_corrections_dict = {
        "facet_name": ["Patch_1", "Patch_2", "Patch_3"],
        "meanDecOffsetDeg": [-0.5 / 3600, -1.5 / 3600, 1 / 3600],
        "stdRAOffsetDeg": [1 / 3600, 0.5 / 3600, 0.5 / 3600],
        "stdDecOffsetDeg": [1 / 3600, 0.5 / 3600, 0.5 / 3600],
    }
    with pytest.raises(ValueError, match="Missing key in corrections dict: meanRAOffsetDeg"):
        validate_corrections(invalid_corrections_dict)


def test_main(mock_fits_image, mock_facet_region, mock_astrometry_corrections, tmp_path):
    """
    Test the main function of the correct_astrometry script.
    """
    # Define test parameters
    input_image = mock_fits_image
    region_file = mock_facet_region
    corrections_file = mock_astrometry_corrections
    output_image = tmp_path / "test_correct_astrometry.fits"
    overwrite = True

    # Run the script
    main(
        input_image,
        region_file,
        corrections_file,
        output_image,
        overwrite,
    )

    # Check if the output image was created
    assert output_image.exists(), "Astrometry-corrected image was not created."

    # Check if the shifts were done correctly by verifying that the sources are in the expected
    # locations. Sources should be shifted by the opposite of the offsets in the corrections dict,
    # so their positions should be shifted as follows (in pixels):
    #    Patch_1 (was at [100, 100]): +1 in x (RA), +1 in y (Dec) to [101, 101]
    #    Patch_2 (was at [200, 200]): -1 in x, +1 in y to [199, 201]
    #    Patch_3 (was at [300, 300]): +1 in x, -1 in y to [301, 299]
    with fits.open(output_image) as hdu_list:
        # Note: data axes are [STOKES, FREQ, DEC, RA]
        assert np.isclose(hdu_list[0].data[0, 0, 101, 101], 1), "Source not at expected position."
        assert np.isclose(hdu_list[0].data[0, 0, 201, 199], 1), "Source not at expected position."
        assert np.isclose(hdu_list[0].data[0, 0, 299, 301], 1), "Source not at expected position."

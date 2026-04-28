"""
Tests for the script that corrects the astrometry of an image.
"""

import json
import pytest

import astropy.io.fits as fits
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
    corrections_dict = {
        "facet_name": ["Patch_1", "Patch_2", "Patch_3"],
        "meanRAOffsetDeg": [1 / 3600, -2 / 3600, 0.5 / 3600],
        "meanDecOffsetDeg": [-0.5 / 3600, -1.5 / 3600, 1 / 3600],
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
    """
    image_file_path = tmp_path / "test_correct_astrometry_image.fits"
    make_template_image(
        image_file_path, 215.59, -12.73, ximsize=512, yimsize=512, cellsize_deg=0.000417
    )
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
        "polygon(215.59054,-12.735746,215.59082,-12.735629,215.5907,-12.735354,215.59042,-12.735471)\n",
        "point(215.59062, -12.735519) # text={Patch_1}\n",
        "polygon(215.59107,-12.73545,215.59151,-12.735265,215.59133,-12.734826,215.59089,-12.735011)\n",
        "point(215.59123, -12.735146) # text={Patch_2}\n",
        "polygon(215.59049,-12.73536,215.59088,-12.735192,215.59071,-12.734794,215.59032,-12.734962)\n",
        "point(215.59055, -12.735169) # text={Patch_3}\n",
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
    try:
        main(
            input_image,
            region_file,
            corrections_file,
            output_image,
            overwrite,
        )

        # Check if the output image was created
        assert output_image.exists(), "Astrometry-corrected image was not created."
    finally:
        # Clean up temporary files
        input_image.unlink(missing_ok=True)
        region_file.unlink(missing_ok=True)
        corrections_file.unlink(missing_ok=True)
        output_image.unlink(missing_ok=True)

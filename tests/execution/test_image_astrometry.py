import json

import pytest

from rapthor.execution.image.astrometry import (
    astrometry_corrected_image_path,
    correct_astrometry_image,
    validate_astrometry_corrections,
)


def test_astrometry_corrected_image_path_adds_astrometry_infix():
    assert (
        astrometry_corrected_image_path("/work/sector_1-MFS-I-image-pb.fits").as_posix()
        == "/work/sector_1-MFS-I-image-pb-ast.fits"
    )
    assert (
        astrometry_corrected_image_path("/work/sector_1-MFS-I-image-pb.fits.fz").as_posix()
        == "/work/sector_1-MFS-I-image-pb-ast.fits.fz"
    )


def test_correct_astrometry_image_copies_input_when_no_offsets_are_available(tmp_path):
    input_image = tmp_path / "sector_1-MFS-I-image-pb.fits"
    input_image.write_text("pb image")

    output_image = correct_astrometry_image(input_image, region_file=None, corrections_file=None)

    assert output_image == tmp_path / "sector_1-MFS-I-image-pb-ast.fits"
    assert output_image.read_text() == "pb image"


def test_correct_astrometry_image_treats_empty_offsets_as_no_correction(tmp_path):
    input_image = tmp_path / "sector_1-MFS-I-image-pb.fits"
    offsets_file = tmp_path / "sector_1.astrometry_offsets.json"
    input_image.write_text("pb image")
    offsets_file.write_text("{}")

    output_image = correct_astrometry_image(
        input_image, region_file=None, corrections_file=offsets_file
    )

    assert output_image.read_text() == "pb image"


def test_validate_astrometry_corrections_rejects_missing_keys():
    corrections = {
        "facet_name": ["field"],
        "meanRAOffsetDeg": [0.1],
        "meanDecOffsetDeg": [0.1],
        "stdRAOffsetDeg": [0.01],
    }

    with pytest.raises(ValueError, match="stdDecOffsetDeg"):
        validate_astrometry_corrections(corrections)


def test_validate_astrometry_corrections_rejects_mismatched_lengths():
    corrections = {
        "facet_name": ["field", "other"],
        "meanRAOffsetDeg": [0.1],
        "meanDecOffsetDeg": [0.1],
        "stdRAOffsetDeg": [0.01],
        "stdDecOffsetDeg": [0.01],
    }

    with pytest.raises(ValueError, match="equal length"):
        validate_astrometry_corrections(corrections)


def test_validate_astrometry_corrections_accepts_diagnostics_schema():
    corrections = {
        "facet_name": ["field"],
        "meanRAOffsetDeg": [0.1],
        "meanDecOffsetDeg": [0.2],
        "stdRAOffsetDeg": [0.01],
        "stdDecOffsetDeg": [0.02],
        "meanClippedRAOffsetDeg": [0.1],
        "stdClippedRAOffsetDeg": [0.01],
        "meanClippedDecOffsetDeg": [0.2],
        "stdClippedDecOffsetDeg": [0.02],
    }

    assert validate_astrometry_corrections(json.loads(json.dumps(corrections))) == corrections

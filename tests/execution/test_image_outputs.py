from pathlib import Path

import pytest

from rapthor.execution.image.outputs import (
    file_records_for_required_patterns,
    first_existing_file,
    mfs_extra_image_patterns,
    optional_first_existing_file,
    require_directory,
    require_file,
)
from rapthor.lib.records import directory_record, file_record


def test_image_output_helpers_require_expected_paths(tmp_path):
    image = tmp_path / "sector_1-MFS-I-image.fits"
    image.write_text("image")
    ms = tmp_path / "sector_1_concat.ms"
    ms.mkdir()

    assert require_file(str(image), "image") == file_record(image)
    assert require_directory(str(ms), "measurement set") == directory_record(ms)

    with pytest.raises(FileNotFoundError, match="missing image"):
        require_file(str(tmp_path / "missing.fits"), "missing image")
    with pytest.raises(FileNotFoundError, match="missing directory"):
        require_directory(str(tmp_path / "missing.ms"), "missing directory")


def test_image_output_helpers_find_matching_files_in_sorted_order(tmp_path):
    second = tmp_path / "sector_1-MFS-U-image.fits"
    first = tmp_path / "sector_1-MFS-Q-image.fits"
    second.write_text("image")
    first.write_text("image")
    patterns = [str(tmp_path / "sector_1-MFS-[QUV]-image.fits")]

    assert first_existing_file(patterns, "extra image") == file_record(first)
    assert file_records_for_required_patterns(patterns, "extra image") == [
        file_record(first),
        file_record(second),
    ]
    assert optional_first_existing_file([str(tmp_path / "missing-*.fits")]) is None


def test_mfs_extra_image_patterns_include_compressed_suffix(tmp_path):
    patterns = [
        Path(pattern).name
        for pattern in mfs_extra_image_patterns("sector_1", str(tmp_path), compressed=True)
    ]

    assert "sector_1-MFS-[QUV]-image.fits.fz" in patterns
    assert "sector_1-MFS-*dirty.fits.fz" in patterns

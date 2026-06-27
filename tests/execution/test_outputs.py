import pytest

from rapthor.execution.outputs import (
    OutputRecordError,
    directory_record,
    directory_record_path,
    file_record,
    file_record_path,
    is_output_record,
    optional_directory_record_path,
    optional_file_record_path,
    output_record_path,
    validate_output_record,
)


def test_file_and_directory_records_are_finalizer_compatible(tmp_path):
    file_path = tmp_path / "image.fits"
    dir_path = tmp_path / "measurement.ms"

    assert file_record(file_path) == {"class": "File", "path": file_path.as_posix()}
    assert directory_record(dir_path) == {"class": "Directory", "path": dir_path.as_posix()}


def test_validate_output_record_accepts_nested_records(tmp_path):
    output = [
        [file_record(tmp_path / "image.fits"), directory_record(tmp_path / "data.ms")],
        [None],
    ]

    assert validate_output_record(output, allow_none=True) is output


def test_validate_output_record_rejects_invalid_records():
    with pytest.raises(OutputRecordError, match="Invalid output record"):
        validate_output_record({"class": "File"})


def test_is_output_record_rejects_empty_path():
    assert not is_output_record({"class": "File", "path": ""})


def test_required_record_path_helpers_extract_paths(tmp_path):
    file_path = tmp_path / "image.fits"
    dir_path = tmp_path / "measurement.ms"

    assert file_record_path(file_record(file_path)) == file_path.as_posix()
    assert directory_record_path(directory_record(dir_path)) == dir_path.as_posix()
    assert output_record_path(file_record(file_path), "File") == file_path.as_posix()


def test_optional_record_path_helpers_accept_none(tmp_path):
    file_path = tmp_path / "image.fits"
    dir_path = tmp_path / "measurement.ms"

    assert optional_file_record_path(None) is None
    assert optional_directory_record_path(None) is None
    assert optional_file_record_path(file_record(file_path)) == file_path.as_posix()
    assert optional_directory_record_path(directory_record(dir_path)) == dir_path.as_posix()


def test_record_path_helpers_reject_wrong_or_malformed_records(tmp_path):
    with pytest.raises(OutputRecordError, match="Expected a File output record"):
        file_record_path(directory_record(tmp_path / "data.ms"))

    with pytest.raises(OutputRecordError, match="Expected a Directory output record"):
        directory_record_path({"class": "Directory", "path": ""})

from pathlib import Path

import pytest

import rapthor.lib.records as records_module
from rapthor.lib.records import (
    OutputRecordError,
    directory_record,
    directory_record_path,
    file_record,
    file_record_path,
    is_output_record,
    move_record_recursive,
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

    assert optional_file_record_path(None) is None
    assert optional_file_record_path(file_record(file_path)) == file_path.as_posix()


def test_record_path_helpers_reject_wrong_or_malformed_records(tmp_path):
    with pytest.raises(OutputRecordError, match="Expected a File output record"):
        file_record_path(directory_record(tmp_path / "data.ms"))

    with pytest.raises(OutputRecordError, match="Expected a Directory output record"):
        directory_record_path({"class": "Directory", "path": ""})


def test_move_record_recursive_moves_files_and_returns_their_new_records(tmp_path):
    (tmp_path / "a.json").write_text("a")
    (tmp_path / "b.pdf").write_text("b")
    dest_dir = tmp_path / "plots"
    moved = [file_record(dest_dir / "a.json"), [file_record(dest_dir / "b.pdf")], None]

    records = [file_record(tmp_path / "a.json"), [file_record(tmp_path / "b.pdf")], None]
    assert move_record_recursive(records, str(dest_dir)) == moved
    # Moving the new records again, as a resumed run does, leaves the files in place
    assert move_record_recursive(moved, str(dest_dir)) == moved

    assert (dest_dir / "a.json").read_text() == "a"
    assert (dest_dir / "b.pdf").read_text() == "b"
    assert not (tmp_path / "a.json").exists()


def test_clean_record_moves_a_directory_aside_before_removing_it(tmp_path, monkeypatch):
    directory = tmp_path / "prep.ms"
    directory.mkdir()
    (directory / "table.dat").write_text("data")
    record = records_module.directory_record(directory)
    partial = Path(records_module.partial_product_path(directory))

    def stop(path):
        raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(records_module.shutil, "rmtree", stop)
        with pytest.raises(KeyboardInterrupt):
            records_module.clean_if_file_or_directory_record(record)
    # A removal that is stopped leaves no incomplete directory under its final name
    assert not directory.exists()
    assert (partial / "table.dat").is_file()

    directory.mkdir()
    records_module.clean_if_file_or_directory_record(record)
    assert not directory.exists()
    assert not partial.exists()

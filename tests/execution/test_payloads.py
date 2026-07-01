from pathlib import Path

import pytest

from rapthor.execution.payloads import (
    PayloadSerializationError,
    assert_serializable_payload,
    optional_file_path,
    optional_string,
    validate_basename,
    validate_int_list,
    validate_string_list,
)
from rapthor.lib.records import file_record


def test_assert_serializable_payload_accepts_plain_values():
    payload = {
        "path": "/data/input.ms",
        "threads": 4,
        "enabled": True,
        "items": [{"name": "sector_0", "weight": 1.5}],
    }

    assert assert_serializable_payload(payload) is payload


def test_assert_serializable_payload_rejects_path_objects():
    with pytest.raises(PayloadSerializationError, match="payload.path"):
        assert_serializable_payload({"path": Path("/data/input.ms")})


def test_assert_serializable_payload_rejects_non_string_mapping_keys():
    with pytest.raises(PayloadSerializationError, match="non-string key"):
        assert_serializable_payload({1: "bad"})


def test_assert_serializable_payload_rejects_domain_like_objects():
    class FieldLike:
        pass

    with pytest.raises(PayloadSerializationError, match="FieldLike"):
        assert_serializable_payload({"field": FieldLike()})


def test_validate_basename_accepts_plain_filename():
    assert validate_basename("output.ms", "output_filename") == "output.ms"


@pytest.mark.parametrize("filename", ["", "/tmp/output.ms", "nested/output.ms"])
def test_validate_basename_rejects_empty_or_non_basename(filename):
    with pytest.raises(ValueError, match="output_filename must be"):
        validate_basename(filename, "output_filename")


def test_optional_file_path_accepts_file_record_path_string_or_none():
    path = "/data/model.fits"

    assert optional_file_path(file_record(path), "model") == path
    assert optional_file_path(path, "model") == path
    assert optional_file_path(None, "model") is None


def test_optional_file_path_rejects_non_file_payload():
    with pytest.raises(ValueError, match="model must be a File record, path string, or None"):
        optional_file_path({"class": "Directory", "path": "/data"}, "model")


@pytest.mark.parametrize("value", [None, "", "None"])
def test_optional_string_returns_none_for_unset_sentinels(value):
    assert optional_string(value) is None


def test_optional_string_returns_string_for_set_values():
    assert optional_string("applycal") == "applycal"
    assert optional_string(7) == "7"


def test_validate_string_list_accepts_non_empty_strings():
    values = ["sector_1.ms", "sector_2.ms"]

    assert validate_string_list(values, "sector_filenames") == values


def test_validate_string_list_rejects_non_string_or_empty_values():
    with pytest.raises(ValueError, match="sector_filenames must be a list of strings"):
        validate_string_list(["sector_1.ms", ""], "sector_filenames")


def test_validate_string_list_can_require_non_empty_list():
    with pytest.raises(ValueError, match="sector_filenames must be a non-empty list of strings"):
        validate_string_list([], "sector_filenames", allow_empty=False)


def test_validate_string_list_uses_non_empty_message_for_malformed_required_list():
    with pytest.raises(ValueError, match="sector_filenames must be a non-empty list of strings"):
        validate_string_list(["sector_1.ms", 7], "sector_filenames", allow_empty=False)


def test_validate_int_list_accepts_exact_length():
    assert validate_int_list([1, 2], "wsclean_imsize", length=2) == [1, 2]


def test_validate_int_list_rejects_wrong_length():
    with pytest.raises(ValueError, match="wsclean_imsize must contain exactly 2 entries"):
        validate_int_list([1], "wsclean_imsize", length=2)

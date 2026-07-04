from pathlib import Path

import pytest

from rapthor.execution.calibrate.validation import validate_calibrate_payload
from rapthor.execution.image.validation import validate_image_payload
from rapthor.execution.payloads import (
    PayloadSerializationError,
    assert_serializable_payload,
    optional_file_path,
    optional_string,
    validate_basename,
    validate_int_list,
    validate_required_list,
    validate_string_list,
)
from rapthor.lib.records import file_record
from tests.execution.payload_factories import (
    representative_calibrate_payload,
    representative_image_payload,
)


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
        """Domain-like object that must not be accepted in worker payloads."""

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


def test_validate_required_list_accepts_non_empty_list():
    values = ["model-0.fits", "model-1.fits"]

    assert validate_required_list(values, "model_images") == values


def test_validate_required_list_can_require_exact_length():
    with pytest.raises(ValueError, match="model_image_ra_dec must contain exactly 2 entries"):
        validate_required_list(["12h00m00s"], "model_image_ra_dec", length=2)


@pytest.mark.parametrize("value", [None, [], "not-a-list"])
def test_validate_required_list_rejects_empty_or_non_list(value):
    with pytest.raises(ValueError, match="model_images must be a non-empty list"):
        validate_required_list(value, "model_images")


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


def test_validate_image_payload_validates_nested_sector_contract():
    payload = representative_image_payload()

    validated = validate_image_payload(payload)

    assert validated["sectors"][0]["prepare_tasks"][0]["msout"] == "obs.prep.ms"
    assert validated["sectors"][0]["image_cube_specs"][0]["filename"] == (
        "sector_1_I_freq_cube.fits"
    )


def test_validate_image_payload_rejects_non_mapping_prepare_task():
    payload = representative_image_payload()
    payload["sectors"][0]["prepare_tasks"] = ["obs.prep.ms"]

    with pytest.raises(ValueError, match=r"sectors\[0\].prepare_tasks\[0\] must be a mapping"):
        validate_image_payload(payload)


def test_validate_calibrate_payload_validates_solve_slot_contract():
    payload = representative_calibrate_payload()

    validated = validate_calibrate_payload(payload)

    assert validated["chunks"][0]["solve_slots"][0]["h5parm"] == "solve1.h5"


def test_validate_calibrate_payload_rejects_non_list_solve_slot_values():
    payload = representative_calibrate_payload()
    payload["chunks"][0]["solve_slots"][0]["solutions_per_direction"] = "1"

    with pytest.raises(ValueError, match="solutions_per_direction must be a list"):
        validate_calibrate_payload(payload)


def test_validate_calibrate_payload_validates_image_predict_contract():
    payload = representative_calibrate_payload()
    payload["image_based_predict"] = True
    payload["image_predict"] = {
        "skymodel": None,
        "model_image_root": "model",
        "model_image_ra_dec": ["12h00m00s", "+10d00m00s"],
        "model_image_imsize": [1024],
        "model_images": ["model-term-0.fits"],
        "facet_region_file": "facet.reg",
        "facet_region_path": "/work/calibrate_1/facet.reg",
    }

    with pytest.raises(
        ValueError,
        match="image_predict.model_image_imsize must contain exactly 2 entries",
    ):
        validate_calibrate_payload(payload)

from pathlib import Path

import pytest

from rapthor.execution.payloads import PayloadSerializationError, assert_serializable_payload


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

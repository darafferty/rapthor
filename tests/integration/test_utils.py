"""Tests for integration test helper functions."""

import configparser
from .utils import update_parset_path
import pytest


def test_update_parset_path(tmp_path):
    """Test the update_parset_path helper function."""
    parset_content = """[section1]
                        param1 = value1
                        param2 = value2
                        [section2]
                        param3 = value3
                        param4 = value4"""
    parset_path = tmp_path / "test.parset"
    parset_path.write_text(parset_content)
    updated_parset_path = update_parset_path(
        parset_path, {"param1": "new_value1", "param3": "new_value3"}
    )
    updated_parset = configparser.ConfigParser()
    updated_parset.read(updated_parset_path)
    assert updated_parset["section1"]["param1"] == "new_value1"
    assert updated_parset["section1"]["param2"] == "value2"
    assert updated_parset["section2"]["param3"] == "new_value3"
    assert updated_parset["section2"]["param4"] == "value4"


def test_update_parset_path_missing_param(tmp_path):
    """Test the update_parset_path helper function with a missing parameter."""
    parset_content = """[section1]
                        param1 = value1
                        param2 = value2
                        [section2]
                        param3 = value3
                        param4 = value4"""
    parset_path = tmp_path / "test.parset"
    parset_path.write_text(parset_content)
    with pytest.raises(ValueError, match="Parameters .* not found in parset."):
        update_parset_path(parset_path, {"param1": "new_value1", "param5": "new_value5"})

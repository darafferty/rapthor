"""Tests for integration test helper functions."""

import configparser

import pytest

from .utils import parse_combine_h5parms_args_from_log, update_parset_path


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


def test_parse_combine_h5parms_args_from_log(tmp_path):
    """Test parsing the non-DP3 combine_h5parms command from a CWL log."""
    log_path = tmp_path / "combine.log"
    log_path.write_text(
        "$ combine_h5parms.py fast.h5 medium.h5 combined.h5 p1p2_scalar "
        "--reweight=False --cal_names=patch1,patch2 --cal_fluxes=1.0,2.0\n"
    )

    args = parse_combine_h5parms_args_from_log(log_path)

    assert args == {
        "inh5parm1": "fast.h5",
        "inh5parm2": "medium.h5",
        "outh5parm": "combined.h5",
        "mode": "p1p2_scalar",
        "reweight": "False",
        "cal_names": "patch1,patch2",
        "cal_fluxes": "1.0,2.0",
    }


def test_parse_combine_h5parms_args_from_log_without_command_raises(tmp_path):
    """Test that missing combine_h5parms commands are reported clearly."""
    log_path = tmp_path / "not_combine.log"
    log_path.write_text("$ DP3 steps=[solve1]\n")

    with pytest.raises(ValueError, match="No combine_h5parms.py command found"):
        parse_combine_h5parms_args_from_log(log_path)

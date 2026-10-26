"""
Tests for the wsclean_predict script.
"""

import argparse

import pytest

from rapthor.scripts.wsclean_predict import main, ra_dec_pair


@pytest.mark.parametrize(
    "value, expected",
    [
        ("8:37:21.48,-32.53.29.04", ["8:37:21.48", "-32.53.29.04"]),  # southern field
        ("17:15:01.03,56.45.51.41", ["17:15:01.03", "56.45.51.41"]),  # northern field
    ],
)
def test_ra_dec_pair(value, expected):
    """
    Test that ra_dec_pair splits an 'RA,Dec' string into [RA, Dec].
    """
    assert ra_dec_pair(value) == expected


@pytest.mark.parametrize("value", ["8:37:21.48", "8:37:21.48,-32.53.29.04,1"])
def test_ra_dec_pair_raises_error_if_not_two_values(value):
    """
    Test that ra_dec_pair raises an error unless exactly two values are given.
    """
    with pytest.raises(argparse.ArgumentTypeError, match="expected 'RA,Dec'"):
        ra_dec_pair(value)


def test_main_passes_negative_dec_to_predict(tmp_path, monkeypatch, mocker):
    """
    Test that a negative Dec, given on the command line as the CWL step does,
    is parsed by main and passed unchanged to predict.
    """
    msin = tmp_path / "input.ms"
    msin.mkdir()
    region = tmp_path / "facets_ds9.reg"
    region.touch()
    skymodel = tmp_path / "calibration_skymodel.txt"
    skymodel.touch()
    monkeypatch.chdir(tmp_path)  # main writes msout_names.json to the working directory

    # Mock the steps that need real data or WSClean
    mocker.patch("rapthor.scripts.wsclean_predict.read_ds9_region_file", return_value=[])
    mocker.patch("rapthor.scripts.wsclean_predict.make_writable", return_value=str(msin))
    mocker.patch(
        "rapthor.scripts.wsclean_predict.optimal_rendering_parameters",
        return_value=(100, 1e-5),
    )
    predict_mock = mocker.patch("rapthor.scripts.wsclean_predict.predict", return_value=0)
    monkeypatch.setattr(
        "sys.argv",
        [
            "wsclean_predict.py",
            "--ra_dec=8:37:21.480000,-32.53.29.040000",
            "--region",
            str(region),
            "--msin",
            str(msin),
            "--skymodel",
            str(skymodel),
            "--frequency_bandwidth",
            "150234485.5",
            "1000000.0",
            "--imsize",
            "100",
            "100",
            "--cellsize",
            "0.0004166666666666667",
        ],
    )

    assert main() == 0
    predict_mock.assert_called_once()
    ra_dec = predict_mock.call_args.args[3]
    assert ra_dec == ["8:37:21.480000", "-32.53.29.040000"]

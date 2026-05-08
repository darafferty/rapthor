"""
Test cases for the `rapthor.process` module.
"""

import pytest

from rapthor.process import (
    _do_calibrate_mode,
    chunk_observations,
    do_final_pass,
    make_report,
    run,
    run_steps,
)


def test_run(parset_file=None, logging_level="info"):
    pass


def test_run_steps(field=None, steps=None, final=False):
    pass


def test_do_final_pass(field=None, selfcal_steps=None, final_step=None):
    pass


def test_chunk_observations(field=None, steps=None, data_fraction=None):
    pass


def test_make_report(field=None, outfile=None):
    pass


@pytest.mark.parametrize(
    "strategy, expected",
    [
        (
            {
                "di": [],
                "dd": ["fast_phase", "full_jones"],
            },
            {"di": False, "dd": True},
        ),  # No DI calibration
        (
            {
                "di": ["fast_phase"],
                "dd": [],
            },
            {"di": True, "dd": False},
        ),  # Fast DI calibration
        (
            {
                "di": ["full_jones"],
                "dd": [],
            },
            {"di": True, "dd": False},
        ),  # Full DI calibration
        (
            {
                "di": [],
                "dd": ["fast_phase", "full_jones"],
            },
            {"di": False, "dd": True},
        ),  # Fast DD calibration
        (
            {
                "di": ["fast_phase"],
                "dd": ["full_jones"],
            },
            {"di": True, "dd": True},
        ),  # Full Jones calibration
        (
            {
                "di": [],
                "dd": [],
            },
            {"di": False, "dd": False},
        ),  # No DD calibration
    ],
)
def test_do_calibrate_mode(strategy, expected):
    """Test function that determines whether or not to do DI or DD calibration"""
    assert _do_calibrate_mode(strategy) == expected


def test_do_calibrate_mode_with_unrecognized_modes_raises_error():
    """Test that _do_calibrate_mode raises a ValueError when no calibration modes are present"""
    with pytest.raises(
        ValueError,
        match=r"Calibration strategy {'unknown_mode': \['fast_phase', 'full_jones'\]} does not contain any of the calibration modes \['di', 'dd'\]",
    ):
        _do_calibrate_mode({"unknown_mode": ["fast_phase", "full_jones"]})

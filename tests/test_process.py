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


@pytest.mark.parametrize(
    "calibration_strategy, expected_calls",
    [
        ({"di": ["full_jones"]}, [("predict", "di", 1), ("calibrate", "di", 1)]),
        ({"dd": ["fast_phase"]}, [("calibrate", "dd", 1)]),
        (
            {"di": ["full_jones"], "dd": ["fast_phase"]},
            [("predict", "di", 1), ("calibrate", "di", 1), ("calibrate", "dd", 1)],
        ),
        (
            {"dd": ["fast_phase"], "di": ["full_jones"]},
            [("calibrate", "dd", 1), ("predict", "di", 1), ("calibrate", "di", 1)],
        ),
    ],
)
def test_run_steps_preserves_calibration_strategy_order(
    monkeypatch, calibration_strategy, expected_calls
):
    """Test that run_steps preserves the DI/DD ordering from calibration_strategy."""

    calls = []

    class RecordingOperation:
        operation_name = None

        def __init__(self, mode, field, index):
            self.mode = mode
            self.index = index

        def run(self):
            calls.append((self.operation_name, self.mode, self.index))

    class RecordingPredict(RecordingOperation):
        operation_name = "predict"

    class RecordingCalibrate(RecordingOperation):
        operation_name = "calibrate"

    class Field:
        cycle_number = 1
        dde_mode = "single"
        do_predict = False
        do_image = False
        do_check = False

        def update(self, step, index, final=False):
            self.__dict__.update(step)

    monkeypatch.setattr("rapthor.process.Predict", RecordingPredict)
    monkeypatch.setattr("rapthor.process.Calibrate", RecordingCalibrate)

    run_steps(
        Field(),
        [
            {
                "do_calibrate": True,
                "calibration_strategy": calibration_strategy,
            }
        ],
    )

    assert calls == expected_calls

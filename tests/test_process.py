"""
Test cases for the `rapthor.process` module.
"""

import pytest

from rapthor.lib.calibration_memory import CalibrationMemoryRiskError
from rapthor.process import (
    _do_calibrate_mode,
    check_preflight_calibration_memory,
    chunk_observations,
    do_final_pass,
    make_report,
    run,
    run_steps,
)


class _MinimalRunField:
    """Field test double for checks performed before pipeline operations start."""

    epoch_observations = []

    def __init__(self, parset):
        self.parset = parset


class _RunStepsField:
    """Field test double for calibration-cycle orchestration tests."""

    cycle_number = 1
    dde_mode = "single"
    num_patches = 1
    do_predict = False
    do_image = False
    do_check = False

    def __init__(self, calls=None):
        self.calls = calls

    def update(self, step, index, final=False):
        if self.calls is not None:
            self.calls.append(("update", index))
        self.__dict__.update(step)

    def set_obs_parameters(self):
        if self.calls is not None:
            self.calls.append(("observation-parameters", self.cycle_number))


def test_check_preflight_calibration_memory_checks_each_cycle(monkeypatch):
    calls = []

    class Field:
        max_directions = 6

    monkeypatch.setattr(
        "rapthor.process.check_calibration_memory",
        lambda field, cycle, directions, step: calls.append((cycle, directions, step)),
    )
    steps = [{"max_directions": 3}, {}]

    check_preflight_calibration_memory(Field(), steps)

    assert calls == [(1, 3, steps[0]), (2, 6, steps[1])]


def test_run_validates_strategy_before_preflight(monkeypatch):
    calls = []
    parset = {
        "generate_initial_skymodel": False,
        "strategy": "synthetic",
    }
    strategy_steps = [{"do_calibrate": False}]

    monkeypatch.setattr("rapthor.process.parset_read", lambda _: parset)
    monkeypatch.setattr("rapthor.process._logging.set_level", lambda _: None)
    monkeypatch.setattr("rapthor.process.Field", _MinimalRunField)
    monkeypatch.setattr("rapthor.process.set_strategy", lambda _: strategy_steps)
    monkeypatch.setattr(
        "rapthor.process.validate_strategy",
        lambda steps, settings: calls.append("validate"),
    )
    monkeypatch.setattr(
        "rapthor.process.check_preflight_calibration_memory",
        lambda field, steps: calls.append("preflight"),
    )
    monkeypatch.setattr(
        "rapthor.process.do_final_pass",
        lambda field, selfcal, final: calls.append("final-decision") or False,
    )
    monkeypatch.setattr("rapthor.process.make_report", lambda _: None)

    run("synthetic.parset")

    assert calls == ["validate", "preflight", "final-decision"]


def test_run_stops_when_preflight_reports_high_risk(monkeypatch):
    parset = {
        "generate_initial_skymodel": False,
        "strategy": "synthetic",
    }

    def fail_preflight(*args):
        raise CalibrationMemoryRiskError("high OOM risk")

    monkeypatch.setattr("rapthor.process.parset_read", lambda _: parset)
    monkeypatch.setattr("rapthor.process._logging.set_level", lambda _: None)
    monkeypatch.setattr("rapthor.process.Field", _MinimalRunField)
    monkeypatch.setattr("rapthor.process.set_strategy", lambda _: [{"do_calibrate": True}])
    monkeypatch.setattr("rapthor.process.validate_strategy", lambda *_: None)
    monkeypatch.setattr("rapthor.process.check_preflight_calibration_memory", fail_preflight)
    monkeypatch.setattr("rapthor.process.do_final_pass", pytest.fail)
    monkeypatch.setattr("rapthor.process.run_steps", pytest.fail)
    monkeypatch.setattr("rapthor.process.ImageInitial", pytest.fail)

    with pytest.raises(CalibrationMemoryRiskError, match="high OOM risk"):
        run("synthetic.parset")


def test_run_preflight_checks_repeated_final_configuration_once(monkeypatch):
    preflight_calls = []
    run_steps_calls = []
    parset = {
        "final_data_fraction": 1.0,
        "generate_initial_skymodel": False,
        "ntimes_to_repeat_final_cycle": 2,
        "strategy": "synthetic",
    }
    final_step = {"do_calibrate": True}

    class Field:
        epoch_observations = []
        make_quv_images = False
        dde_mode = "single"

        def __init__(self, field_parset):
            self.parset = field_parset

    monkeypatch.setattr("rapthor.process.parset_read", lambda _: parset)
    monkeypatch.setattr("rapthor.process._logging.set_level", lambda _: None)
    monkeypatch.setattr("rapthor.process.Field", Field)
    monkeypatch.setattr("rapthor.process.set_strategy", lambda _: [final_step])
    monkeypatch.setattr("rapthor.process.validate_strategy", lambda *_: None)
    monkeypatch.setattr(
        "rapthor.process.check_preflight_calibration_memory",
        lambda field, steps: preflight_calls.append(steps),
    )
    monkeypatch.setattr("rapthor.process.do_final_pass", lambda *_: True)
    monkeypatch.setattr("rapthor.process.chunk_observations", lambda *_: None)
    monkeypatch.setattr(
        "rapthor.process.run_steps",
        lambda field, steps, final=False: run_steps_calls.append((steps, final)),
    )
    monkeypatch.setattr("rapthor.process.make_report", lambda _: None)

    run("synthetic.parset")

    assert preflight_calls == [[final_step]]
    assert run_steps_calls == [([final_step], True)] * 3


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

    monkeypatch.setattr("rapthor.process.Predict", RecordingPredict)
    monkeypatch.setattr("rapthor.process.Calibrate", RecordingCalibrate)
    monkeypatch.setattr(
        "rapthor.process.check_calibration_memory", lambda field, index, directions: None
    )

    run_steps(
        _RunStepsField(),
        [
            {
                "do_calibrate": True,
                "calibration_strategy": calibration_strategy,
            }
        ],
    )

    assert calls == expected_calls


def test_run_steps_checks_cycle_memory_after_update_before_calibration(monkeypatch):
    calls = []

    class RecordingCalibrate:
        def __init__(self, mode, field, index):
            self.mode = mode
            self.index = index

        def run(self):
            calls.append(("calibrate", self.index))

    def check_memory(field, cycle_number, directions):
        calls.append(("memory", cycle_number))

    monkeypatch.setattr("rapthor.process.Calibrate", RecordingCalibrate)
    monkeypatch.setattr("rapthor.process.check_calibration_memory", check_memory)

    run_steps(
        _RunStepsField(calls),
        [{"do_calibrate": True, "calibration_strategy": {"dd": ["fast_phase"]}}],
    )

    assert calls == [
        ("update", 1),
        ("observation-parameters", 1),
        ("memory", 1),
        ("calibrate", 1),
    ]


def test_run_steps_stops_after_resolving_parameters_when_cycle_risk_is_high(monkeypatch):
    calls = []

    def fail_memory_check(field, cycle_number, directions):
        calls.append(("memory", cycle_number))
        raise CalibrationMemoryRiskError("high OOM risk")

    monkeypatch.setattr("rapthor.process.check_calibration_memory", fail_memory_check)
    monkeypatch.setattr("rapthor.process.Predict", pytest.fail)
    monkeypatch.setattr("rapthor.process.Calibrate", pytest.fail)

    with pytest.raises(CalibrationMemoryRiskError, match="high OOM risk"):
        run_steps(
            _RunStepsField(calls),
            [{"do_calibrate": True, "calibration_strategy": {"dd": ["fast_phase"]}}],
        )

    assert calls == [("update", 1), ("observation-parameters", 1), ("memory", 1)]

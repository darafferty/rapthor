"""
Test cases for the `rapthor.process` module.
"""

import logging
from types import SimpleNamespace

import pytest

from rapthor.lib.calibration_memory import check_calibration_memory
from rapthor.process import (
    _do_calibrate_mode,
    check_preflight_calibration_memory,
    chunk_observations,
    do_final_pass,
    make_report,
    run,
    run_steps,
)


def _memory_observation(
    name="observation.ms",
    *,
    channels=2,
    stations=2,
    sampling_interval=4,
    parameters=None,
):
    return SimpleNamespace(
        name=name,
        numchannels=channels,
        stations=[f"station-{index}" for index in range(stations)],
        timepersample=sampling_interval,
        parameters=parameters or {},
    )


def _memory_field(*, observations=None, calibration_strategy=None):
    field = SimpleNamespace(
        observations=observations or [_memory_observation()],
        calibration_strategy=calibration_strategy,
        do_calibrate=True,
        do_slowgain_solve=False,
        do_fulljones_solve=False,
        fast_timestep_sec=8,
        medium_timestep_sec=12,
        slow_timestep_sec=16,
        fulljones_timestep_sec=20,
        max_directions=5,
        parset={"cluster_specific": {"mem_per_node_gb": 0}},
    )
    field.set_obs_parameters = lambda: None
    return field


def _memory_step(**overrides):
    return {
        "do_calibrate": True,
        "calibration_strategy": {"dd": ["fast_phase"]},
        "fast_timestep_sec": 8,
        **overrides,
    }


def test_preflight_calibration_memory_uses_max_directions_and_largest_solve():
    field = _memory_field()
    step = {
        "do_calibrate": True,
        "calibration_strategy": {"dd": ["fast_phase", "slow_gains"]},
        "fast_timestep_sec": 8,
        "slow_timestep_sec": 20,
    }

    assessment = check_calibration_memory(
        field,
        1,
        7,
        step,
    )

    assert assessment.mode == "dd"
    assert assessment.solve_type == "slow_gains"
    assert assessment.directions == 7
    assert assessment.baselines == 3
    assert assessment.solution_interval_seconds == 20
    assert assessment.memory.time_steps == 5


def test_resolved_calibration_memory_uses_actual_intervals_and_largest_task():
    first = _memory_observation(
        "first.ms",
        channels=2,
        parameters={"solint_fast_timestep": [2]},
    )
    second = _memory_observation(
        "second.ms",
        channels=8,
        parameters={"solint_fast_timestep": [3, 3]},
    )
    field = _memory_field(
        observations=[first, second],
        calibration_strategy={"dd": ["fast_phase"]},
    )

    assessment = check_calibration_memory(field, 2, 4)

    assert assessment.observation_name == "second.ms"
    assert assessment.directions == 4
    assert assessment.solution_interval_seconds == 12
    assert assessment.peak_memory_gb == pytest.approx(3 * 8 * 3 * (4 + 1) * 80 / 1e9)


def test_calibration_memory_uses_one_direction_for_di():
    field = _memory_field(calibration_strategy={"di": ["full_jones"]})
    step = {
        "do_calibrate": True,
        "calibration_strategy": field.calibration_strategy,
        "fulljones_timestep_sec": 20,
    }

    assessment = check_calibration_memory(field, 1, 50, step)

    assert assessment.mode == "di"
    assert assessment.directions == 1
    assert assessment.solve_type == "full_jones"


def test_calibration_memory_logs_max_directions_for_di(caplog):
    field = _memory_field(calibration_strategy={"di": ["full_jones"]})
    step = {
        "do_calibrate": True,
        "calibration_strategy": field.calibration_strategy,
        "fulljones_timestep_sec": 20,
        "max_directions": 4,
    }

    with caplog.at_level(logging.INFO, logger="rapthor"):
        check_calibration_memory(field, 1, 4, step)

    assert "with 1 direction(s) (max_directions=4)" in caplog.text


def test_di_only_calibration_memory_does_not_require_dd_directions():
    field = _memory_field(calibration_strategy={"dd": [], "di": ["full_jones"]})
    step = {
        "do_calibrate": True,
        "calibration_strategy": field.calibration_strategy,
        "fulljones_timestep_sec": 20,
    }

    assessment = check_calibration_memory(field, 1, None, step)

    assert assessment.mode == "di"
    assert assessment.directions == 1


def test_calibration_memory_skips_non_calibration_cycle():
    field = _memory_field()

    assessment = check_calibration_memory(field, 1, 5, {"do_calibrate": False})

    assert assessment is None


def test_calibration_memory_skips_explicit_strategy_without_solves():
    field = _memory_field(calibration_strategy={"di": [], "dd": []})

    assessment = check_calibration_memory(field, 1, 5)

    assert assessment is None


def test_check_calibration_memory_prefers_configured_limit(monkeypatch, caplog):
    field = _memory_field()
    field.parset["cluster_specific"]["mem_per_node_gb"] = 32
    monkeypatch.setattr("rapthor.lib.calibration_memory.get_available_memory", pytest.fail)

    with caplog.at_level(logging.INFO, logger="rapthor"):
        check_calibration_memory(field, 1, 5, _memory_step())

    assert "configured per-node memory" in caplog.text


def test_check_calibration_memory_uses_current_machine(monkeypatch, caplog):
    field = _memory_field()
    monkeypatch.setattr("rapthor.lib.calibration_memory.get_available_memory", lambda: 24)

    with caplog.at_level(logging.INFO, logger="rapthor"):
        check_calibration_memory(field, 1, 5, _memory_step())

    assert "24.000 GB of memory available on current machine" in caplog.text


def test_check_calibration_memory_handles_probe_failure(monkeypatch, caplog):
    field = _memory_field()

    def fail_memory_probe():
        raise RuntimeError("memory probe failed")

    monkeypatch.setattr("rapthor.lib.calibration_memory.get_available_memory", fail_memory_probe)

    with caplog.at_level(logging.WARNING, logger="rapthor"):
        check_calibration_memory(field, 1, 5, _memory_step())

    assert "capacity comparison skipped" in caplog.text
    assert "memory available on current machine could not be determined" in caplog.text


def test_calibration_memory_warns_when_limit_is_exceeded(caplog):
    field = _memory_field()
    field.parset["cluster_specific"]["mem_per_node_gb"] = 0.000001

    with caplog.at_level(logging.DEBUG, logger="rapthor"):
        check_calibration_memory(field, 3, 5, _memory_step())

    assert "likely out of memory" in caplog.text
    assert "cycle 3" in caplog.text
    assert "fast_phase" in caplog.text
    assert "configured per-node memory" in caplog.text


def test_calibration_memory_logs_headroom_when_estimate_fits(caplog):
    field = _memory_field()
    field.parset["cluster_specific"]["mem_per_node_gb"] = 1

    with caplog.at_level(logging.INFO, logger="rapthor"):
        check_calibration_memory(field, 1, 5, _memory_step())

    assert "headroom" in caplog.text
    assert "likely out of memory" not in caplog.text


def test_calibration_memory_treats_exact_limit_as_fitting(caplog):
    field = _memory_field()
    field.parset["cluster_specific"]["mem_per_node_gb"] = 3 * 2 * 2 * (5 + 1) * 80 / 1e9

    with caplog.at_level(logging.INFO, logger="rapthor"):
        check_calibration_memory(field, 1, 5, _memory_step())

    assert "0.000 GB headroom" in caplog.text
    assert "likely out of memory" not in caplog.text


def test_calibration_memory_logs_estimate_without_memory_limit(monkeypatch, caplog):
    field = _memory_field()

    def fail_memory_probe():
        raise RuntimeError("memory probe failed")

    monkeypatch.setattr("rapthor.lib.calibration_memory.get_available_memory", fail_memory_probe)

    with caplog.at_level(logging.WARNING, logger="rapthor"):
        check_calibration_memory(field, 1, 5, _memory_step())

    assert "capacity comparison skipped" in caplog.text
    assert "memory available on current machine could not be determined" in caplog.text


def test_check_preflight_calibration_memory_checks_each_calibration_cycle(caplog):
    field = _memory_field()
    field.parset["cluster_specific"]["mem_per_node_gb"] = 1
    steps = [
        {
            "do_calibrate": True,
            "calibration_strategy": {"dd": ["fast_phase"]},
            "fast_timestep_sec": 8,
            "max_directions": directions,
        }
        for directions in (3, 5)
    ]

    with caplog.at_level(logging.INFO, logger="rapthor"):
        check_preflight_calibration_memory(field, steps)

    assert caplog.text.count("pre-flight max_directions upper bound") == 2
    assert "cycle 1" in caplog.text
    assert "cycle 2" in caplog.text
    assert "max_directions=3" in caplog.text
    assert "max_directions=5" in caplog.text


def test_check_preflight_calibration_memory_uses_field_default_max_directions(caplog):
    field = _memory_field()
    field.max_directions = 6
    field.parset["cluster_specific"]["mem_per_node_gb"] = 1

    with caplog.at_level(logging.INFO, logger="rapthor"):
        check_preflight_calibration_memory(
            field,
            [
                {
                    "do_calibrate": True,
                    "calibration_strategy": {"dd": ["fast_phase"]},
                    "fast_timestep_sec": 8,
                }
            ],
        )

    assert "6 direction(s) (max_directions=6)" in caplog.text


def test_check_preflight_calibration_memory_skips_limit_lookup_without_solves(monkeypatch):
    field = _memory_field()
    monkeypatch.setattr("rapthor.lib.calibration_memory.get_available_memory", pytest.fail)

    check_preflight_calibration_memory(
        field,
        [{"do_calibrate": True, "calibration_strategy": {"di": [], "dd": []}}],
    )


def test_check_calibration_memory_is_advisory(monkeypatch, caplog):
    field = _memory_field()

    def fail_estimate(*args, **kwargs):
        raise ValueError("invalid estimate input")

    monkeypatch.setattr("rapthor.lib.calibration_memory.estimate_dp3_peak_memory", fail_estimate)

    with caplog.at_level(logging.WARNING, logger="rapthor"):
        check_calibration_memory(field, 1, 5, _memory_step())

    assert "Could not complete the advisory DP3 calibration memory" in caplog.text
    assert "Processing will continue" in caplog.text


def test_check_calibration_memory_resolves_observation_parameters(caplog):
    observation = _memory_observation()
    field = _memory_field(
        observations=[observation],
        calibration_strategy={"dd": ["fast_phase"]},
    )
    field.parset["cluster_specific"]["mem_per_node_gb"] = 1
    field.num_patches = 4
    calls = []

    def set_obs_parameters():
        calls.append("set")
        observation.parameters["solint_fast_timestep"] = [2]

    field.set_obs_parameters = set_obs_parameters

    with caplog.at_level(logging.INFO, logger="rapthor"):
        check_calibration_memory(field, 2, 4)

    assert calls == ["set"]
    assert "resolved facet count" in caplog.text
    assert "4 direction(s)" in caplog.text


def test_resolved_calibration_memory_assessment_is_advisory(monkeypatch, caplog):
    observation = _memory_observation()
    field = _memory_field(
        observations=[observation],
        calibration_strategy={"dd": ["fast_phase"]},
    )
    field.num_patches = 4

    def set_obs_parameters():
        observation.parameters["solint_fast_timestep"] = [2]

    def fail_estimate(*args, **kwargs):
        raise KeyError("missing interval")

    field.set_obs_parameters = set_obs_parameters
    monkeypatch.setattr("rapthor.lib.calibration_memory.estimate_dp3_peak_memory", fail_estimate)

    with caplog.at_level(logging.WARNING, logger="rapthor"):
        check_calibration_memory(field, 1, 4)

    assert (
        "Could not complete the advisory DP3 calibration memory resolved facet count" in caplog.text
    )
    assert "Processing will continue" in caplog.text


def test_calibration_memory_skips_observation_setup_without_solves():
    field = _memory_field(calibration_strategy={"di": [], "dd": []})
    field.set_obs_parameters = pytest.fail

    check_calibration_memory(field, 1, 5)


def test_run_validates_strategy_before_preflight(monkeypatch):
    calls = []
    parset = {
        "generate_initial_skymodel": False,
        "strategy": "synthetic",
    }
    strategy_steps = [{"do_calibrate": False}]

    class Field:
        epoch_observations = []

        def __init__(self, field_parset):
            self.parset = field_parset

    monkeypatch.setattr("rapthor.process.parset_read", lambda _: parset)
    monkeypatch.setattr("rapthor.process._logging.set_level", lambda _: None)
    monkeypatch.setattr("rapthor.process.Field", Field)
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

    class Field:
        cycle_number = 1
        dde_mode = "single"
        num_patches = 1
        do_predict = False
        do_image = False
        do_check = False

        def update(self, step, index, final=False):
            self.__dict__.update(step)

    monkeypatch.setattr("rapthor.process.Predict", RecordingPredict)
    monkeypatch.setattr("rapthor.process.Calibrate", RecordingCalibrate)
    monkeypatch.setattr(
        "rapthor.process.check_calibration_memory", lambda field, index, directions: None
    )

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


def test_run_steps_checks_cycle_memory_after_update_before_calibration(monkeypatch):
    calls = []

    class Field:
        cycle_number = 1
        dde_mode = "single"
        num_patches = 1
        do_predict = False
        do_image = False
        do_check = False

        def update(self, step, index, final=False):
            calls.append(("update", index))
            self.__dict__.update(step)

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
        Field(),
        [{"do_calibrate": True, "calibration_strategy": {"dd": ["fast_phase"]}}],
    )

    assert calls == [("update", 1), ("memory", 1), ("calibrate", 1)]

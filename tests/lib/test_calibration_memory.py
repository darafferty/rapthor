"""Tests for DP3 calibration memory assessment and enforcement."""

import logging
from types import SimpleNamespace

import pytest

from rapthor.lib.calibration_memory import (
    CalibrationMemoryRiskError,
    check_calibration_memory,
)


def _observation(
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
        parameters=parameters if parameters is not None else {},
    )


def _field(
    *,
    observations=None,
    calibration_strategy=None,
    fail_on_risk=False,
    memory_limit_gb=0,
):
    return SimpleNamespace(
        observations=observations if observations is not None else [_observation()],
        calibration_strategy=calibration_strategy,
        do_calibrate=True,
        do_slowgain_solve=False,
        do_fulljones_solve=False,
        fast_timestep_sec=8,
        medium_timestep_sec=12,
        slow_timestep_sec=16,
        fulljones_timestep_sec=20,
        max_directions=5,
        parset={
            "cluster_specific": {
                "fail_on_calibration_oom_risk": fail_on_risk,
                "mem_per_node_gb": memory_limit_gb,
            }
        },
    )


def _step(**overrides):
    return {
        "do_calibrate": True,
        "calibration_strategy": {"dd": ["fast_phase"]},
        "fast_timestep_sec": 8,
        **overrides,
    }


def test_preflight_uses_max_directions_and_largest_solve():
    field = _field()

    estimate = check_calibration_memory(
        field,
        1,
        7,
        {
            "do_calibrate": True,
            "calibration_strategy": {"dd": ["fast_phase", "slow_gains"]},
            "fast_timestep_sec": 8,
            "slow_timestep_sec": 20,
        },
    )

    assert estimate.mode == "dd"
    assert estimate.solve_type == "slow_gains"
    assert estimate.directions == 7
    assert estimate.baselines == 3
    assert estimate.solution_interval_seconds == 20
    assert estimate.memory.time_steps == 5


def test_resolved_check_uses_actual_intervals_and_largest_task():
    first = _observation(
        "first.ms",
        channels=2,
        parameters={"solint_fast_timestep": [2]},
    )
    second = _observation(
        "second.ms",
        channels=8,
        parameters={"solint_fast_timestep": [3, 3]},
    )
    field = _field(
        observations=[first, second],
        calibration_strategy={"dd": ["fast_phase"]},
    )

    estimate = check_calibration_memory(field, 2, 4)

    assert estimate.observation_name == "second.ms"
    assert estimate.directions == 4
    assert estimate.solution_interval_seconds == 12
    assert estimate.peak_memory_gb == pytest.approx(3 * 8 * 3 * (4 + 1) * 80 / 1e9)


def test_di_calibration_uses_one_direction_and_logs_configured_max(caplog):
    field = _field(calibration_strategy={"di": ["full_jones"]})
    step = {
        "do_calibrate": True,
        "calibration_strategy": field.calibration_strategy,
        "fulljones_timestep_sec": 20,
        "max_directions": 4,
    }

    with caplog.at_level(logging.INFO, logger="rapthor"):
        estimate = check_calibration_memory(field, 1, 50, step)

    assert estimate.mode == "di"
    assert estimate.directions == 1
    assert estimate.solve_type == "full_jones"
    assert "with 1 direction(s) (max_directions=4)" in caplog.text


def test_di_only_calibration_does_not_require_dd_directions():
    field = _field(calibration_strategy={"dd": [], "di": ["full_jones"]})

    estimate = check_calibration_memory(
        field,
        1,
        None,
        {
            "do_calibrate": True,
            "calibration_strategy": field.calibration_strategy,
            "fulljones_timestep_sec": 20,
        },
    )

    assert estimate.mode == "di"
    assert estimate.directions == 1


@pytest.mark.parametrize(
    "step, strategy",
    [
        ({"do_calibrate": False}, None),
        (None, {"di": [], "dd": []}),
    ],
)
def test_check_skips_cycles_without_calibration_solves(step, strategy):
    field = _field(calibration_strategy=strategy)

    assert check_calibration_memory(field, 1, 5, step) is None


def test_check_prefers_configured_limit(monkeypatch, caplog):
    field = _field(memory_limit_gb=32)
    monkeypatch.setattr("rapthor.lib.calibration_memory.get_available_memory", pytest.fail)

    with caplog.at_level(logging.INFO, logger="rapthor"):
        check_calibration_memory(field, 1, 5, _step())

    assert "configured per-node memory" in caplog.text


def test_check_uses_current_machine_when_limit_is_zero(monkeypatch, caplog):
    field = _field()
    monkeypatch.setattr("rapthor.lib.calibration_memory.get_available_memory", lambda: 24)

    with caplog.at_level(logging.INFO, logger="rapthor"):
        check_calibration_memory(field, 1, 5, _step())

    assert "24.000 GB of memory available on current machine" in caplog.text


def test_over_limit_warns_and_continues_by_default(caplog):
    field = _field(memory_limit_gb=0.000001)

    with caplog.at_level(logging.DEBUG, logger="rapthor"):
        estimate = check_calibration_memory(field, 3, 5, _step())

    assert estimate is not None
    assert "likely out of memory" in caplog.text
    assert "cycle 3" in caplog.text
    assert "fast_phase" in caplog.text
    assert "configured per-node memory" in caplog.text


@pytest.mark.parametrize(
    "step, parameters, expected_stage",
    [
        (_step(), {}, "pre-flight max_directions upper bound"),
        (None, {"solint_fast_timestep": [2]}, "resolved facet count"),
    ],
    ids=["preflight", "resolved"],
)
def test_strict_mode_raises_for_known_high_risk(step, parameters, expected_stage):
    observation = _observation(parameters=parameters)
    field = _field(
        observations=[observation],
        calibration_strategy={"dd": ["fast_phase"]},
        fail_on_risk=True,
        memory_limit_gb=0.000001,
    )

    with pytest.raises(CalibrationMemoryRiskError) as exc_info:
        check_calibration_memory(field, 3, 5, step)

    message = str(exc_info.value)
    for expected in (
        expected_stage,
        "cycle 3",
        "fast_phase",
        "observation.ms",
        "estimated",
        "configured per-node memory",
        "likely out of memory by",
        "fail_on_calibration_oom_risk=False",
    ):
        assert expected in message


@pytest.mark.parametrize("limit_scale", [1, 2], ids=["exact_limit", "below_limit"])
def test_strict_mode_allows_estimate_that_fits(limit_scale, caplog):
    exact_limit = 3 * 2 * 2 * (5 + 1) * 80 / 1e9
    field = _field(fail_on_risk=True, memory_limit_gb=exact_limit * limit_scale)

    with caplog.at_level(logging.INFO, logger="rapthor"):
        estimate = check_calibration_memory(field, 1, 5, _step())

    assert estimate is not None
    assert "headroom" in caplog.text
    assert "likely out of memory" not in caplog.text


def test_strict_mode_remains_advisory_when_capacity_probe_fails(monkeypatch, caplog):
    field = _field(fail_on_risk=True)

    def fail_memory_probe():
        raise RuntimeError("memory probe failed")

    monkeypatch.setattr("rapthor.lib.calibration_memory.get_available_memory", fail_memory_probe)

    with caplog.at_level(logging.WARNING, logger="rapthor"):
        estimate = check_calibration_memory(field, 1, 5, _step())

    assert estimate is not None
    assert "capacity comparison skipped" in caplog.text
    assert "memory available on current machine could not be determined" in caplog.text


@pytest.mark.parametrize(
    "step, parameters, expected_stage",
    [
        (_step(), None, "pre-flight max_directions upper bound"),
        (None, {"solint_fast_timestep": [2]}, "resolved facet count"),
    ],
    ids=["preflight", "resolved"],
)
def test_strict_mode_remains_advisory_when_estimation_fails(
    monkeypatch,
    caplog,
    step,
    parameters,
    expected_stage,
):
    field = _field(
        observations=[_observation(parameters=parameters)],
        calibration_strategy={"dd": ["fast_phase"]},
        fail_on_risk=True,
    )

    def fail_estimate(*args, **kwargs):
        raise ValueError("invalid estimate input")

    monkeypatch.setattr("rapthor.lib.calibration_memory.estimate_dp3_peak_memory", fail_estimate)

    with caplog.at_level(logging.WARNING, logger="rapthor"):
        estimate = check_calibration_memory(field, 1, 5, step)

    assert estimate is None
    assert f"Could not complete the advisory DP3 calibration memory {expected_stage}" in caplog.text
    assert "Processing will continue" in caplog.text


def test_resolved_check_consumes_parameters_without_mutating_field():
    field = _field(
        observations=[_observation(parameters={"solint_fast_timestep": [2]})],
        calibration_strategy={"dd": ["fast_phase"]},
    )
    field.set_obs_parameters = pytest.fail

    estimate = check_calibration_memory(field, 2, 4)

    assert estimate.solution_interval_seconds == 8

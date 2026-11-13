"""Tests for DP3 calibration memory assessment and enforcement."""

import logging
from types import SimpleNamespace

import numpy as np
import pytest

from rapthor.lib.calibration_memory import (
    BdaSettings,
    CalibrationMemoryRiskError,
    _bda_channels,
    _bda_time_factor,
    _bda_visibility_samples,
    check_calibration_memory,
)


def _observation(
    name="observation.ms",
    *,
    channels=2,
    stations=2,
    sampling_interval=4,
    parameters=None,
    positions=None,
    baselines=None,
):
    # By default, the MS has every baseline, including the autocorrelations
    if baselines is None:
        baselines = np.column_stack(np.triu_indices(stations))
    return SimpleNamespace(
        name=name,
        numchannels=channels,
        channels_are_regular=True,
        stations=[f"station-{index}" for index in range(stations)],
        antenna_positions=positions if positions is not None else [[0.0, 0.0, 0.0]] * stations,
        baselines=np.asarray(baselines),
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
        generate_screens=False,
        use_image_based_predict=False,
        use_wsclean_predict=False,
        calibrate_bda_timebase=0.0,
        calibrate_bda_frequencybase=0.0,
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
    step = {
        "do_calibrate": True,
        "calibration_strategy": {"dd": ["fast_phase"]},
        "fast_timestep_sec": 8,
        "max_directions": 5,
    }
    step.update(overrides)
    return step


def test_preflight_uses_max_directions_and_largest_solve():
    estimate = check_calibration_memory(
        _field(),
        1,
        7,
        _step(
            calibration_strategy={"dd": ["fast_phase", "slow_gains"]},
            slow_timestep_sec=20,
        ),
    )

    assert estimate.mode == "dd"
    assert estimate.solve_type == "slow_gains"
    assert estimate.directions == 7
    assert estimate.baselines == 3
    assert estimate.solution_interval_seconds == 20
    assert estimate.time_steps == 5


def test_resolved_check_uses_actual_intervals_and_largest_task():
    first = _observation("first.ms", parameters={"solint_fast_timestep": [2]})
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


def test_di_calibration_uses_one_direction(caplog):
    field = _field(calibration_strategy={"di": ["full_jones"]})

    with caplog.at_level(logging.INFO, logger="rapthor"):
        estimate = check_calibration_memory(
            field,
            1,
            None,
            _step(
                calibration_strategy=field.calibration_strategy,
                fulljones_timestep_sec=20,
                max_directions=4,
            ),
        )

    assert estimate.directions == 1
    assert estimate.solve_type == "full_jones"
    assert "with 1 direction(s) (max_directions=4)" in caplog.text


@pytest.mark.parametrize(
    "step, strategy",
    [
        ({"do_calibrate": False}, None),
        (None, {"di": [], "dd": []}),
    ],
)
def test_check_skips_cycles_without_calibration_solves(step, strategy):
    assert check_calibration_memory(_field(calibration_strategy=strategy), 1, 5, step) is None


def test_check_prefers_configured_limit(monkeypatch, caplog):
    field = _field(memory_limit_gb=32)
    monkeypatch.setattr("rapthor.lib.calibration_memory.get_available_memory", pytest.fail)

    with caplog.at_level(logging.INFO, logger="rapthor"):
        check_calibration_memory(field, 1, 5, _step())

    assert "configured per-node memory" in caplog.text


def test_check_uses_current_machine_when_limit_is_zero(monkeypatch, caplog):
    monkeypatch.setattr("rapthor.lib.calibration_memory.get_available_memory", lambda: 24)

    with caplog.at_level(logging.INFO, logger="rapthor"):
        check_calibration_memory(_field(), 1, 5, _step())

    assert "24.000 GB of memory available on current machine" in caplog.text


def test_over_limit_warns_and_continues_by_default(caplog):
    with caplog.at_level(logging.DEBUG, logger="rapthor"):
        estimate = check_calibration_memory(
            _field(memory_limit_gb=0.000001),
            3,
            5,
            _step(),
        )

    assert estimate is not None
    assert "likely out of memory" in caplog.text
    assert "cycle 3" in caplog.text
    assert "fast_phase" in caplog.text


@pytest.mark.parametrize(
    "step, parameters, expected_stage",
    [
        (_step(), {}, "pre-flight max_directions upper bound"),
        (None, {"solint_fast_timestep": [2]}, "resolved facet count"),
    ],
    ids=["preflight", "resolved"],
)
def test_strict_mode_raises_for_known_high_risk(step, parameters, expected_stage):
    field = _field(
        observations=[_observation(parameters=parameters)],
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
        "likely out of memory by",
        "fail_on_calibration_oom_risk=False",
    ):
        assert expected in message


def test_strict_mode_allows_estimate_that_fits(caplog):
    exact_limit = 3 * 2 * 2 * (5 + 1) * 80 / 1e9
    field = _field(fail_on_risk=True, memory_limit_gb=exact_limit)

    with caplog.at_level(logging.INFO, logger="rapthor"):
        estimate = check_calibration_memory(field, 1, 5, _step())

    assert estimate is not None
    assert "headroom" in caplog.text
    assert "likely out of memory" not in caplog.text


def test_strict_mode_remains_advisory_when_capacity_probe_fails(monkeypatch, caplog):
    def fail_memory_probe():
        raise RuntimeError("memory probe failed")

    monkeypatch.setattr("rapthor.lib.calibration_memory.get_available_memory", fail_memory_probe)

    with caplog.at_level(logging.WARNING, logger="rapthor"):
        estimate = check_calibration_memory(
            _field(fail_on_risk=True),
            1,
            5,
            _step(),
        )

    assert estimate is not None
    assert "capacity comparison skipped" in caplog.text


def test_estimation_failure_remains_advisory(monkeypatch, caplog):
    def fail_estimate(*args, **kwargs):
        raise ValueError("invalid estimate input")

    monkeypatch.setattr("rapthor.lib.calibration_memory.estimate_dp3_peak_memory", fail_estimate)

    with caplog.at_level(logging.WARNING, logger="rapthor"):
        estimate = check_calibration_memory(_field(fail_on_risk=True), 1, 5, _step())

    assert estimate is None
    assert "Could not complete the advisory DP3 calibration memory pre-flight" in caplog.text
    assert "Processing will continue" in caplog.text


# Three stations on a line give baselines of 0 m (three autocorrelations), 100 m,
# 900 m and 1000 m
LINE_POSITIONS = [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [1000.0, 0.0, 0.0]]


def _bda_field(**settings):
    parameters = {"solint_fast_timestep": [6], "bda_maxinterval": [16.0], "bda_minchannels": [2]}
    observation = _observation(
        channels=8, stations=3, parameters=parameters, positions=LINE_POSITIONS
    )
    field = _field(observations=[observation], calibration_strategy={"dd": ["fast_phase"]})
    field.calibrate_bda_timebase = 2000.0
    field.calibrate_bda_frequencybase = 1000.0
    for name, value in settings.items():
        setattr(field, name, value)
    return field


@pytest.mark.parametrize(
    "settings, expected_samples",
    [
        # 6 time slots of 4 s and 8 channels. Time factors, limited to 16 s:
        # 4, 4 (0 m, 100 m) and 2 (900 m, 1000 m), so 6 + 2 * 4 - 1 = 13 slots
        # are held. Channels, at least 2: 2, 2, 8 and 8.
        # Samples: 3 * 2 * 4 + 2 * 4 + 8 * 7 + 8 * 7
        pytest.param({}, 144, id="bda"),
        # Time factors as above, all 8 channels: 3 * 8 * 4 + 8 * 4 + 8 * 7 + 8 * 7
        pytest.param({"calibrate_bda_frequencybase": 0.0}, 240, id="timebase-only"),
        # Time factors of 1, so 6 + 1 = 7 slots are held. Channels as in "bda":
        # 3 * 2 * 7 + 2 * 7 + 8 * 7 + 8 * 7
        pytest.param({"calibrate_bda_timebase": 0.0}, 168, id="frequencybase-only"),
    ],
)
def test_resolved_dd_check_uses_baseline_dependent_averaging(settings, expected_samples):
    """The samples per baseline follow DP3's BDAAverager and BdaSolverBuffer."""
    estimate = check_calibration_memory(_bda_field(**settings), 2, 4)

    assert estimate.visibility_samples == expected_samples
    assert estimate.peak_memory_gb == pytest.approx(expected_samples * (4 + 1) * 80 / 1e9)


@pytest.mark.parametrize(
    "settings, strategy_step, calibration_strategy",
    [
        pytest.param({}, None, {"di": ["fast_phase"]}, id="di"),
        pytest.param({"use_image_based_predict": True}, None, None, id="image-based-predict"),
        pytest.param({"use_wsclean_predict": True}, None, None, id="wsclean"),
        pytest.param({"generate_screens": True}, None, None, id="screens"),
        pytest.param(
            {"calibrate_bda_timebase": 0.0, "calibrate_bda_frequencybase": 0.0},
            None,
            None,
            id="disabled",
        ),
        # The BDA limits are not known yet, so preflight assumes no BDA
        pytest.param({}, _step(fast_timestep_sec=24), None, id="preflight"),
    ],
)
def test_check_uses_all_samples_when_calibration_does_not_use_bda(
    settings, strategy_step, calibration_strategy
):
    field = _bda_field(**settings)
    if calibration_strategy is not None:
        field.calibration_strategy = calibration_strategy

    estimate = check_calibration_memory(field, 2, 4, strategy_step)

    assert estimate.visibility_samples == 6 * 8 * 6


def test_check_uses_all_samples_with_irregular_channels():
    field = _bda_field()
    field.observations[0].channels_are_regular = False

    estimate = check_calibration_memory(field, 2, 4)

    assert estimate.visibility_samples == 6 * 8 * 6


BDA = BdaSettings(timebase=2000.0, frequencybase=1000.0, maxinterval=16.0, minchannels=2)


@pytest.mark.parametrize(
    "length, settings, expected_factor",
    [
        pytest.param(900.0, {}, 2, id="baseline-limit"),
        # 4 time slots of 4 s are exactly the 16 s maximum interval
        pytest.param(500.0, {}, 4, id="exactly-maxinterval"),
        pytest.param(100.0, {}, 4, id="maxinterval-limit"),
        pytest.param(0.0, {}, 4, id="autocorrelation"),
        pytest.param(100.0, {"timebase": 0.0}, 1, id="timebase-off"),
    ],
)
def test_bda_time_factor_follows_dp3(length, settings, expected_factor):
    bda = BdaSettings(**{**BDA.__dict__, **settings})

    assert _bda_time_factor(length, 4.0, bda) == expected_factor


@pytest.mark.parametrize(
    "length, settings, expected_channels",
    [
        pytest.param(450.0, {}, 4, id="baseline-limit"),
        pytest.param(100.0, {}, 2, id="minchannels"),
        # 900 m with a 500 m base would keep 15 of the 8 channels
        pytest.param(900.0, {"frequencybase": 500.0}, 8, id="longer-than-frequencybase"),
        pytest.param(100.0, {"frequencybase": 0.0}, 8, id="frequencybase-off"),
    ],
)
def test_bda_channels_follow_dp3(length, settings, expected_channels):
    bda = BdaSettings(**{**BDA.__dict__, **settings})

    assert _bda_channels(length, 8, bda) == expected_channels


def test_bda_samples_include_the_next_interval_held_by_dp3():
    """
    DP3 solves an interval only once every baseline has a row that starts after
    it. With time factors of 7 (autocorrelation) and 1, an 8-slot interval ends
    after the 7-slot row starting at slot 14, so 8 + 2 * 7 - 1 = 21 slots are held.
    """
    bda = BdaSettings(timebase=2000.0, frequencybase=0.0, maxinterval=28.0, minchannels=1)

    samples = _bda_visibility_samples([0.0, 1500.0], 8, 8, 4.0, bda)

    assert samples == 8 * 3 + 8 * 21


def test_bda_estimate_can_exceed_the_unaveraged_one():
    """BDA that barely averages still holds the next interval's rows in DP3."""
    parameters = {"solint_fast_timestep": [8], "bda_maxinterval": [28.0], "bda_minchannels": [1]}
    observation = _observation(
        channels=8, stations=2, parameters=parameters, positions=[[0.0] * 3, [1500.0, 0.0, 0.0]]
    )
    field = _field(observations=[observation], calibration_strategy={"dd": ["fast_phase"]})
    field.calibrate_bda_timebase = 2000.0

    estimate = check_calibration_memory(field, 2, 4)

    # Two autocorrelations with 3 rows and one baseline with 21 rows of 8
    # channels, more than 3 baselines * 8 channels * 8 time slots without BDA
    assert estimate.visibility_samples == 2 * 8 * 3 + 8 * 21
    assert estimate.visibility_samples > 3 * 8 * 8


def test_bda_estimate_uses_only_the_baselines_in_the_ms():
    """
    DP3 averages the baselines present in the MS. An MS without autocorrelations
    (as LINC produces) must not hold the next interval's rows for the time factor
    that the 0 m autocorrelations would get.
    """
    # Stations 3 km apart: every cross baseline gets a time factor of 1
    positions = [[0.0] * 3, [3000.0, 0.0, 0.0], [6000.0, 0.0, 0.0]]
    parameters = {"solint_fast_timestep": [6], "bda_maxinterval": [16.0], "bda_minchannels": [1]}
    observation = _observation(
        channels=8,
        stations=3,
        parameters=parameters,
        positions=positions,
        baselines=[[0, 1], [0, 2], [1, 2]],
    )
    field = _field(observations=[observation], calibration_strategy={"dd": ["fast_phase"]})
    field.calibrate_bda_timebase = 2000.0

    estimate = check_calibration_memory(field, 2, 4)

    # 6 + 2 * 1 - 1 = 7 time slots of 8 channels on 3 baselines. With the
    # autocorrelations, the time factor of 4 (16 s) would have held 13 slots
    assert estimate.visibility_samples == 3 * 8 * 7

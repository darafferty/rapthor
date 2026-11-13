"""DP3 calibration memory checks and optional high-risk enforcement."""

import logging
import math
from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np

from rapthor.lib.calibration import CALIBRATION_SOLVE_METADATA
from rapthor.lib.cluster import DP3MemoryEstimate, estimate_dp3_peak_memory, get_available_memory
from rapthor.lib.strategy import default_calibration_strategy

log = logging.getLogger("rapthor")

FAIL_ON_CALIBRATION_OOM_RISK = "fail_on_calibration_oom_risk"


class CalibrationMemoryRiskError(RuntimeError):
    """Raised when strict memory checking identifies a likely calibration OOM."""


@dataclass(frozen=True)
class CalibrationMemoryEstimate:
    """Inputs and memory terms for one DP3 calibration task."""

    mode: str
    solve_type: str
    observation_name: str
    directions: int
    baselines: int
    channels: int
    sampling_interval_seconds: float
    solution_interval_seconds: float
    time_steps: int
    visibility_samples: int
    memory: DP3MemoryEstimate

    @property
    def peak_memory_gb(self) -> float:
        """Estimated peak memory in decimal gigabytes."""
        return self.memory.peak_memory_gb


@dataclass(frozen=True)
class BdaSettings:
    """Baseline-dependent averaging settings of the DP3 calibration steps."""

    timebase: float
    frequencybase: float
    maxinterval: float
    minchannels: int


@dataclass(frozen=True)
class CalibrationMemoryAssessment:
    """Capacity comparison for the largest DP3 calibration task in one cycle."""

    estimate: CalibrationMemoryEstimate
    cycle_number: int
    stage: str
    max_directions: int
    memory_limit_gb: Optional[float]
    memory_source: str

    @property
    def margin_gb(self) -> Optional[float]:
        """Available headroom, or ``None`` when capacity is unknown."""
        if self.memory_limit_gb is None:
            return None
        return self.memory_limit_gb - self.estimate.peak_memory_gb

    @property
    def high_risk(self) -> bool:
        """Whether the estimate is strictly greater than the known limit."""
        return self.margin_gb is not None and self.margin_gb < 0

    @property
    def task_details(self) -> str:
        """Describe the task and its estimated peak memory."""
        estimate = self.estimate
        return (
            f"DP3 calibration memory {self.stage} for cycle {self.cycle_number}: "
            f"{estimate.mode.upper()} {estimate.solve_type} on "
            f"{estimate.observation_name} with {estimate.directions} direction(s) "
            f"(max_directions={self.max_directions}) "
            f"is estimated at {estimate.peak_memory_gb:.3f} GB"
        )

    @property
    def capacity_message(self) -> str:
        """Describe the capacity result used for logging and enforcement."""
        margin = self.margin_gb
        if margin is None:
            return f"{self.task_details}; capacity comparison skipped because {self.memory_source}"

        details = (
            f"{self.task_details} against {self.memory_limit_gb:.3f} GB of {self.memory_source}"
        )
        if margin < 0:
            return f"{details}; likely out of memory by {-margin:.3f} GB"
        return f"{details}; {margin:.3f} GB headroom"

    @property
    def failure_message(self) -> str:
        """Return the actionable error emitted when strict checking is enabled."""
        return (
            f"{self.capacity_message}. Rapthor is stopping because "
            f"{FAIL_ON_CALIBRATION_OOM_RISK}=True. Set "
            f"{FAIL_ON_CALIBRATION_OOM_RISK}=False to continue despite this risk."
        )


def check_calibration_memory(
    field: object,
    cycle_number: int,
    dd_directions: Optional[int],
    strategy_step: Optional[Mapping[str, object]] = None,
) -> Optional[CalibrationMemoryEstimate]:
    """Estimate and log the largest DP3 calibration task for one cycle.

    ``strategy_step`` is supplied for a preflight check. Without it, the current
    resolved observation parameters and field settings are used. Calculation
    failures remain advisory. A known over-limit estimate raises
    :class:`CalibrationMemoryRiskError` when the corresponding cluster option is
    enabled.
    """
    stage = _memory_check_stage(strategy_step)
    try:
        assessment = _assess_calibration_memory(
            field,
            cycle_number,
            dd_directions,
            strategy_step,
            stage,
        )
        if assessment is None:
            return None
        _log_calibration_memory(assessment)
        fail_on_risk = field.parset["cluster_specific"].get(FAIL_ON_CALIBRATION_OOM_RISK, False)
    except Exception as error:
        log.warning(
            "Could not complete the advisory DP3 calibration memory %s for cycle %s: %s. "
            "Processing will continue.",
            stage,
            cycle_number,
            error,
        )
        log.debug("DP3 calibration memory check failure", exc_info=True)
        return None

    if fail_on_risk and assessment.high_risk:
        raise CalibrationMemoryRiskError(assessment.failure_message)
    return assessment.estimate


def _memory_check_stage(strategy_step: Optional[Mapping[str, object]]) -> str:
    """Describe whether a check uses configured or resolved calibration inputs."""
    if strategy_step is not None:
        return "pre-flight max_directions upper bound"
    return "resolved facet count"


def _assess_calibration_memory(
    field: object,
    cycle_number: int,
    dd_directions: Optional[int],
    strategy_step: Optional[Mapping[str, object]],
    stage: str,
) -> Optional[CalibrationMemoryAssessment]:
    """Build the capacity assessment for the largest task in one cycle."""
    if not _strategy_setting(field, strategy_step, "do_calibrate"):
        return None

    strategy = _strategy_setting(field, strategy_step, "calibration_strategy")
    strategy = strategy or default_calibration_strategy()
    if not any(strategy.values()):
        return None

    estimate = _largest_calibration_memory_estimate(
        field,
        strategy,
        dd_directions,
        strategy_step,
    )
    if estimate is None:
        return None

    memory_limit, memory_source = _memory_limit(field)
    return CalibrationMemoryAssessment(
        estimate=estimate,
        cycle_number=cycle_number,
        stage=stage,
        max_directions=_strategy_setting(field, strategy_step, "max_directions"),
        memory_limit_gb=memory_limit,
        memory_source=memory_source,
    )


def _strategy_setting(
    field: object,
    strategy_step: Optional[Mapping[str, object]],
    name: str,
) -> object:
    """Read a setting from a strategy step, falling back to the field default."""
    if strategy_step is not None and name in strategy_step:
        return strategy_step[name]
    return getattr(field, name)


def _largest_calibration_memory_estimate(
    field: object,
    strategy: Mapping[str, list[str]],
    dd_directions: Optional[int],
    strategy_step: Optional[Mapping[str, object]],
) -> Optional[CalibrationMemoryEstimate]:
    """Return the largest task estimate without logging or changing field state."""
    largest = None

    for mode, solves in strategy.items():
        if not solves:
            continue
        directions = 1 if mode == "di" else int(dd_directions)
        if directions <= 0:
            continue

        for solve_type in solves:
            for observation in field.observations:
                solution_interval_seconds = _solution_interval_seconds(
                    field,
                    observation,
                    solve_type,
                    strategy_step,
                )
                estimate = _estimate_calibration_task(
                    mode,
                    solve_type,
                    observation,
                    directions,
                    solution_interval_seconds,
                    _bda_settings(field, observation, mode, strategy_step),
                )
                if largest is None or estimate.peak_memory_gb > largest.peak_memory_gb:
                    largest = estimate

    return largest


def _solution_interval_seconds(
    field: object,
    observation: object,
    solve_type: str,
    strategy_step: Optional[Mapping[str, object]],
) -> float:
    """Resolve one solve interval from preflight settings or observation state."""
    metadata = CALIBRATION_SOLVE_METADATA[solve_type]
    if strategy_step is not None:
        return _strategy_setting(field, strategy_step, metadata.target_timestep_key)

    solution_interval_samples = max(observation.parameters[metadata.timestep_key])
    return solution_interval_samples * observation.timepersample


def _bda_settings(
    field: object,
    observation: object,
    mode: str,
    strategy_step: Optional[Mapping[str, object]],
) -> Optional[BdaSettings]:
    """Return the BDA settings of the solves, or None when they do not use BDA.

    As in the Calibrate operation, only the DD solves use BDA, and only with
    regular channels and without image-based or WSClean prediction (screens use
    image-based prediction). A preflight check runs before the BDA limits are
    known, so it assumes no BDA.
    """
    timebase = field.calibrate_bda_timebase
    frequencybase = field.calibrate_bda_frequencybase
    if (
        strategy_step is not None
        or mode != "dd"
        or (timebase <= 0 and frequencybase <= 0)
        or field.generate_screens
        or field.use_image_based_predict
        or field.use_wsclean_predict
        or not all(obs.channels_are_regular for obs in field.observations)
    ):
        return None
    # All time chunks of an observation share one value; the shortest interval
    # and the most channels are the conservative choice
    return BdaSettings(
        timebase=timebase,
        frequencybase=frequencybase,
        maxinterval=min(observation.parameters["bda_maxinterval"]),
        minchannels=max(observation.parameters["bda_minchannels"]),
    )


def _bda_time_factor(length: float, sampling_interval_seconds: float, bda: BdaSettings) -> int:
    """Return the BDA time factor of a baseline, as DP3's BDAAverager.cc sets it."""
    time_factor = math.floor(bda.timebase / max(length, 0.1))
    # Rapthor always sets maxinterval to at least 1 s, but DP3 ignores a value of 0
    if bda.maxinterval > 0 and time_factor * sampling_interval_seconds > bda.maxinterval:
        time_factor = math.floor(bda.maxinterval / sampling_interval_seconds)
    return max(time_factor, 1)


def _bda_channels(length: float, channels: int, bda: BdaSettings) -> int:
    """Return the BDA channel count of a baseline, as DP3's BDAAverager.cc sets it."""
    if bda.frequencybase <= 0:
        return channels
    baseline_channels = math.ceil(length / bda.frequencybase * channels)
    if baseline_channels > channels:
        return channels
    return max(baseline_channels, bda.minchannels)


def _bda_visibility_samples(
    lengths: list[float],
    channels: int,
    time_steps: int,
    sampling_interval_seconds: float,
    bda: BdaSettings,
) -> int:
    """Return the samples DP3 holds for one solution interval after BDA."""
    time_factors = [_bda_time_factor(length, sampling_interval_seconds, bda) for length in lengths]
    # DP3 solves an interval once every baseline has a complete row that starts
    # after it (BdaSolverBuffer.h:80-86, BDAAverager.cc:298-299), so up to
    # 2 * max_factor - 1 time steps of the next interval are held as well. Rows
    # of the previous interval that share an output buffer with rows of this one
    # (BdaSolverBuffer.cc:179-204) are not included
    held_time_steps = time_steps + 2 * max(time_factors) - 1
    return sum(
        _bda_channels(length, channels, bda) * math.ceil(held_time_steps / time_factor)
        for length, time_factor in zip(lengths, time_factors)
    )


def _estimate_calibration_task(
    mode: str,
    solve_type: str,
    observation: object,
    directions: int,
    solution_interval_seconds: float,
    bda: Optional[BdaSettings],
) -> CalibrationMemoryEstimate:
    """Estimate one observation/solve task using resolved scientific inputs."""
    station_count = len(observation.stations)
    baselines = station_count * (station_count + 1) // 2
    channels = observation.numchannels
    sampling_interval_seconds = observation.timepersample
    time_steps = math.ceil(solution_interval_seconds / sampling_interval_seconds)
    visibility_samples = baselines * channels * time_steps
    if bda is not None:
        # The baselines in the MS, as DP3 averages them. The autocorrelations
        # only count when the MS has them
        positions = np.asarray(observation.antenna_positions, dtype=float)
        first, second = np.asarray(observation.baselines, dtype=int).T
        lengths = np.linalg.norm(positions[second] - positions[first], axis=1)
        # With BDA, DP3 also holds rows of the next interval, so this can exceed
        # the estimate without BDA
        visibility_samples = _bda_visibility_samples(
            list(lengths), channels, time_steps, sampling_interval_seconds, bda
        )
    memory = estimate_dp3_peak_memory(visibility_samples=visibility_samples, directions=directions)
    return CalibrationMemoryEstimate(
        mode=mode,
        solve_type=solve_type,
        observation_name=observation.name,
        directions=directions,
        baselines=baselines,
        channels=channels,
        sampling_interval_seconds=sampling_interval_seconds,
        solution_interval_seconds=solution_interval_seconds,
        time_steps=time_steps,
        visibility_samples=visibility_samples,
        memory=memory,
    )


def _memory_limit(field):
    """Return the applicable memory limit and its user-facing source."""
    configured_limit = field.parset["cluster_specific"]["mem_per_node_gb"]
    if configured_limit > 0:
        return configured_limit, "configured per-node memory"

    try:
        return get_available_memory(), "memory available on current machine"
    except Exception:
        return None, "memory available on current machine could not be determined"


def _log_calibration_memory(assessment):
    """Log the capacity result and detailed calculation terms."""
    estimate = assessment.estimate
    if assessment.memory_limit_gb is None or assessment.high_risk:
        log.warning(assessment.capacity_message)
    else:
        log.info(assessment.capacity_message)

    log.debug(
        "DP3 memory terms for cycle %s: baselines=%s, channels=%s, sampling_interval=%.3f "
        "s, solution_interval=%.3f s, time_steps=%s, visibility_samples=%s, "
        "visibility_copies=%.3f GB, weights=%.3f GB, weighted_data=%.3f GB",
        assessment.cycle_number,
        estimate.baselines,
        estimate.channels,
        estimate.sampling_interval_seconds,
        estimate.solution_interval_seconds,
        estimate.time_steps,
        estimate.visibility_samples,
        estimate.memory.visibility_copies_gb,
        estimate.memory.weights_gb,
        estimate.memory.weighted_data_gb,
    )

"""DP3 calibration memory checks and optional high-risk enforcement."""

import logging
from dataclasses import dataclass
from typing import Optional

from rapthor.lib.calibration import (
    INTERVAL_KEYS_BY_SOLVE,
    TARGET_TIMESTEP_BY_SOLVE,
    resolve_calibration_strategy,
)
from rapthor.lib.cluster import DP3MemoryEstimate, estimate_dp3_peak_memory, get_available_memory

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
    memory: DP3MemoryEstimate

    @property
    def peak_memory_gb(self) -> float:
        """Estimated peak memory in decimal gigabytes."""
        return self.memory.peak_memory_gb


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


def _setting(field, step, name):
    """Read a setting from a strategy step, falling back to the field default."""
    return step[name] if step is not None and name in step else getattr(field, name)


def _largest_calibration_memory_estimate(field, strategy, dd_directions, step):
    """Return the largest task estimate without logging or changing field state."""
    resolved = step is None
    largest = None

    for mode, solves in strategy.items():
        if not solves:
            continue
        directions = 1 if mode == "di" else int(dd_directions)
        if directions <= 0:
            continue

        for solve_type in solves:
            for observation in field.observations:
                if resolved:
                    timestep_key = INTERVAL_KEYS_BY_SOLVE[solve_type][0]
                    solution_interval = max(observation.parameters[timestep_key])
                    solution_interval *= observation.timepersample
                else:
                    timestep_key = TARGET_TIMESTEP_BY_SOLVE[solve_type]
                    solution_interval = _setting(field, step, timestep_key)

                station_count = len(observation.stations)
                baselines = station_count * (station_count + 1) // 2
                estimate = CalibrationMemoryEstimate(
                    mode=mode,
                    solve_type=solve_type,
                    observation_name=observation.name,
                    directions=directions,
                    baselines=baselines,
                    channels=observation.numchannels,
                    sampling_interval_seconds=observation.timepersample,
                    solution_interval_seconds=solution_interval,
                    memory=estimate_dp3_peak_memory(
                        baselines=baselines,
                        channels=observation.numchannels,
                        solution_interval_seconds=solution_interval,
                        sampling_interval_seconds=observation.timepersample,
                        directions=directions,
                    ),
                )
                if largest is None or estimate.peak_memory_gb > largest.peak_memory_gb:
                    largest = estimate

    return largest


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
        "s, solution_interval=%.3f s, time_steps=%s, visibility_copies=%.3f GB, "
        "weights=%.3f GB, weighted_data=%.3f GB",
        assessment.cycle_number,
        estimate.baselines,
        estimate.channels,
        estimate.sampling_interval_seconds,
        estimate.solution_interval_seconds,
        estimate.memory.time_steps,
        estimate.memory.visibility_copies_gb,
        estimate.memory.weights_gb,
        estimate.memory.weighted_data_gb,
    )


def check_calibration_memory(field, cycle_number, dd_directions, step=None):
    """Estimate and log the largest DP3 calibration task for one cycle.

    ``step`` is supplied for a pre-flight check. Without it, the current resolved
    observation parameters and field settings are used. Calculation failures remain
    advisory. A known over-limit estimate raises :class:`CalibrationMemoryRiskError`
    when the corresponding cluster option is enabled.
    """
    stage = "pre-flight max_directions upper bound" if step is not None else "resolved facet count"
    assessment = None
    fail_on_risk = False
    try:
        if not _setting(field, step, "do_calibrate"):
            return None

        strategy, _ = resolve_calibration_strategy(
            calibration_strategy=_setting(field, step, "calibration_strategy"),
            do_slowgain_solve=_setting(field, step, "do_slowgain_solve"),
            do_fulljones_solve=_setting(field, step, "do_fulljones_solve"),
        )
        if not any(strategy.values()):
            return None

        estimate = _largest_calibration_memory_estimate(field, strategy, dd_directions, step)
        if estimate is None:
            return None

        max_directions = _setting(field, step, "max_directions")
        memory_limit, memory_source = _memory_limit(field)
        assessment = CalibrationMemoryAssessment(
            estimate=estimate,
            cycle_number=cycle_number,
            stage=stage,
            max_directions=max_directions,
            memory_limit_gb=memory_limit,
            memory_source=memory_source,
        )
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

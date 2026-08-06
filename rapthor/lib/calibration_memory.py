"""DP3 calibration memory assessment and advisory logging."""

import logging
from dataclasses import dataclass

from rapthor.lib.calibration import (
    INTERVAL_KEYS_BY_SOLVE,
    TARGET_TIMESTEP_BY_SOLVE,
    resolve_calibration_strategy,
)
from rapthor.lib.cluster import DP3MemoryEstimate, estimate_dp3_peak_memory, get_available_memory

log = logging.getLogger("rapthor")


@dataclass(frozen=True)
class CalibrationMemoryAssessment:
    """Largest estimated DP3 calibration task for one processing cycle."""

    cycle_number: int
    mode: str
    solve_type: str
    observation_name: str
    directions: int
    baselines: int
    channels: int
    sampling_interval_seconds: float
    solution_interval_seconds: float
    estimate: DP3MemoryEstimate


def _strategy_value(field, step, name):
    if step is not None and name in step:
        return step[name]
    return getattr(field, name)


def resolve_field_calibration_strategy(field, step=None, resolved=False):
    """Resolve explicit or legacy solve lists from a field and optional strategy step."""
    calibration_strategy = _strategy_value(field, step, "calibration_strategy")
    if resolved and getattr(field, "_calibration_strategy_defaulted", False):
        calibration_strategy = None
    strategy, _ = resolve_calibration_strategy(
        calibration_strategy=calibration_strategy,
        do_slowgain_solve=_strategy_value(field, step, "do_slowgain_solve"),
        do_fulljones_solve=_strategy_value(field, step, "do_fulljones_solve"),
    )
    return strategy


def has_calibration_solves(field, step=None, resolved=False):
    """Return whether a field or strategy step requests at least one solve."""
    strategy = resolve_field_calibration_strategy(field, step, resolved)
    return any(strategy.values())


def assess_calibration_memory(
    field,
    *,
    cycle_number,
    dd_directions,
    step=None,
    resolved=False,
):
    """Return the largest per-task DP3 calibration memory estimate for a cycle."""
    if not _strategy_value(field, step, "do_calibrate"):
        return None

    calibration_strategy = resolve_field_calibration_strategy(field, step, resolved)
    assessments = []
    for mode, solves in calibration_strategy.items():
        if not solves:
            continue
        directions = 1 if mode == "di" else int(dd_directions)
        if directions <= 0:
            continue
        for solve_type in solves:
            for observation in field.observations:
                if resolved:
                    timestep_key = INTERVAL_KEYS_BY_SOLVE[solve_type][0]
                    solution_timesteps = max(observation.parameters[timestep_key])
                    solution_interval_seconds = solution_timesteps * observation.timepersample
                else:
                    target_key = TARGET_TIMESTEP_BY_SOLVE[solve_type]
                    solution_interval_seconds = _strategy_value(field, step, target_key)

                station_count = len(observation.stations)
                baselines = station_count * (station_count + 1) // 2
                estimate = estimate_dp3_peak_memory(
                    baselines=baselines,
                    channels=observation.numchannels,
                    solution_interval_seconds=solution_interval_seconds,
                    sampling_interval_seconds=observation.timepersample,
                    directions=directions,
                )
                assessments.append(
                    CalibrationMemoryAssessment(
                        cycle_number=cycle_number,
                        mode=mode,
                        solve_type=solve_type,
                        observation_name=observation.name,
                        directions=directions,
                        baselines=baselines,
                        channels=observation.numchannels,
                        sampling_interval_seconds=observation.timepersample,
                        solution_interval_seconds=solution_interval_seconds,
                        estimate=estimate,
                    )
                )

    if not assessments:
        return None
    return max(assessments, key=lambda assessment: assessment.estimate["peak_memory_gb"])


def get_calibration_memory_limit(field):
    """Return the applicable memory limit and a user-facing description of its source."""
    configured_limit_gb = field.parset["cluster_specific"]["mem_per_node_gb"]
    if configured_limit_gb > 0:
        return configured_limit_gb, "configured per-node memory"
    try:
        return get_available_memory(), "memory available on current machine"
    except Exception:
        return None, "memory available on current machine could not be determined"


def log_calibration_memory_assessment(
    assessment,
    *,
    memory_limit_gb,
    memory_source,
    stage,
):
    """Log an advisory capacity result for a DP3 calibration memory estimate."""
    peak_memory_gb = assessment.estimate["peak_memory_gb"]
    task_details = (
        f"DP3 calibration memory {stage} for cycle {assessment.cycle_number}: "
        f"{assessment.mode.upper()} {assessment.solve_type} on "
        f"{assessment.observation_name} with {assessment.directions} direction(s) "
        f"is estimated at {peak_memory_gb:.2f} GB"
    )
    if memory_limit_gb is None:
        log.warning(
            "%s; capacity comparison skipped because %s",
            task_details,
            memory_source,
        )
    else:
        memory_margin_gb = memory_limit_gb - peak_memory_gb
        details = f"{task_details} against {memory_limit_gb:.2f} GB of {memory_source}"
        if memory_margin_gb < 0:
            log.warning("%s; likely out of memory by %.2f GB", details, -memory_margin_gb)
        else:
            log.info("%s; %.2f GB headroom", details, memory_margin_gb)

    log.debug(
        "DP3 memory terms for cycle %s: baselines=%s, channels=%s, sampling_interval=%.3f "
        "s, solution_interval=%.3f s, time_steps=%s, visibility_copies=%.3f GB, "
        "weights=%.3f GB, weighted_data=%.3f GB",
        assessment.cycle_number,
        assessment.baselines,
        assessment.channels,
        assessment.sampling_interval_seconds,
        assessment.solution_interval_seconds,
        assessment.estimate["time_steps"],
        assessment.estimate["visibility_copies_gb"],
        assessment.estimate["weights_gb"],
        assessment.estimate["weighted_data_gb"],
    )

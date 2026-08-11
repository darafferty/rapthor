"""Advisory DP3 calibration memory checks."""

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


def _log_calibration_memory(
    estimate, cycle_number, stage, max_directions, memory_limit, memory_source
):
    """Log the capacity result and detailed calculation terms."""
    task_details = (
        f"DP3 calibration memory {stage} for cycle {cycle_number}: "
        f"{estimate.mode.upper()} {estimate.solve_type} on "
        f"{estimate.observation_name} with {estimate.directions} direction(s) "
        f"(max_directions={max_directions}) "
        f"is estimated at {estimate.peak_memory_gb:.3f} GB"
    )
    if memory_limit is None:
        log.warning("%s; capacity comparison skipped because %s", task_details, memory_source)
    else:
        margin = memory_limit - estimate.peak_memory_gb
        details = f"{task_details} against {memory_limit:.3f} GB of {memory_source}"
        if margin < 0:
            log.warning("%s; likely out of memory by %.3f GB", details, -margin)
        else:
            log.info("%s; %.3f GB headroom", details, margin)

    log.debug(
        "DP3 memory terms for cycle %s: baselines=%s, channels=%s, sampling_interval=%.3f "
        "s, solution_interval=%.3f s, time_steps=%s, visibility_copies=%.3f GB, "
        "weights=%.3f GB, weighted_data=%.3f GB",
        cycle_number,
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

    ``step`` is supplied for a pre-flight check. Without it, observation parameters
    are resolved first and the current field settings are used. The check is advisory:
    errors are logged and do not interrupt processing.
    """
    stage = "pre-flight max_directions upper bound" if step is not None else "resolved facet count"
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

        if step is None:
            field.set_obs_parameters()

        estimate = _largest_calibration_memory_estimate(field, strategy, dd_directions, step)
        if estimate is None:
            return None

        max_directions = _setting(field, step, "max_directions")
        memory_limit, memory_source = _memory_limit(field)
        _log_calibration_memory(
            estimate,
            cycle_number,
            stage,
            max_directions,
            memory_limit,
            memory_source,
        )
        return estimate
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

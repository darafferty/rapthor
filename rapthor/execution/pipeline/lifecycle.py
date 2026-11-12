"""Scheduler-independent process lifecycle helpers."""

import logging
import math
import os

import numpy as np

from rapthor.lib.calibration import CALIBRATION_SOLVE_METADATA
from rapthor.lib.strategy import default_calibration_strategy

log = logging.getLogger("rapthor")


def do_final_pass(field, selfcal_steps, final_step):
    """
    Check the processing state to determine whether a final pass is needed.

    A final pass is needed when selfcal was not done, when the final data
    fraction or QUV settings require another pass, or when the final strategy
    step differs from the last selfcal step.
    """
    if not selfcal_steps:
        return True

    if field.do_check and (field.selfcal_state.diverged or field.selfcal_state.failed):
        log.warning(
            "Selfcal diverged or failed, so skipping final cycle (with a data fraction of %.2f)",
            field.parset["final_data_fraction"],
        )
        return False

    if final_step == selfcal_steps[field.cycle_number - 1]:
        return (
            not np.isclose(
                field.parset["final_data_fraction"],
                field.parset["selfcal_data_fraction"],
            )
            or field.make_quv_images
        )

    return True


def chunk_observations(field, steps, data_fraction, selection_steps=None):
    """
    Chunk observations in time for calibration, data-fraction, or multi-node runs.

    The resulting data fraction may differ from observation to observation
    depending on the length of each observation and the largest requested
    calibration solution interval.

    Parameters
    ----------
    field : Field object
        The Field object for this run
    steps : list of dict
        The strategy steps whose solution intervals set the chunks
    data_fraction : float
        The fraction of the data to use
    selection_steps : list of dict, optional
        The strategy steps whose solution intervals set the time ranges selected when
        the data fraction is less than one, so that these steps all use the same data.
        The selection is kept by rechunk_observations. If None, the steps are used
    """
    _select_data(field, selection_steps or steps, data_fraction)
    field.update_observations(_planned_observations(field, steps, data_fraction))


def rechunk_observations(field, steps, index, data_fraction):
    """
    Chunk observations again for a later selfcal cycle, if its solves need it.

    Each cycle is chunked for its own calibration solution intervals, within the
    time ranges selected for all selfcal cycles when the first cycle was chunked
    (see chunk_observations). The current observations are kept
    when the chunks do not change, as they may point to data with the outliers
    peeled, and in a cycle without calibration, which applies the solutions of the
    cycles before it (and so needs their chunks). New chunks start again from the
    input data, so the outliers are peeled again if any cycle so far peeled them.

    Parameters
    ----------
    field : Field object
        The Field object for this run
    steps : list of dict
        The selfcal strategy steps
    index : int
        The index in steps of the step of the cycle to chunk
    data_fraction : float
        The selfcal data fraction

    Returns
    -------
    step : dict
        The strategy step to run the cycle with. When the observations are chunked
        again, this is a copy with the peeling flag set (the strategy is unchanged)
    """
    step = steps[index]
    if not step["do_calibrate"]:
        return step

    observations = _planned_observations(field, [step], data_fraction)
    if _chunk_layout(observations) == _chunk_layout(field.observations):
        return step

    field.update_observations(observations)
    # The peeling flag is only required in imaging cycles
    peel_outliers = any(
        earlier_step.get("peel_outliers", False) for earlier_step in steps[: index + 1]
    )
    if peel_outliers:
        # Restore the outliers to the calibration sky model, as in the final cycle
        field.imaged_sources_only = False
    return {**step, "peel_outliers": peel_outliers}


def make_report(field, outfile=None):
    """
    Make a summary report of QA metrics for the run.

    Parameters
    ----------
    field : Field object
        The Field object for this run
    outfile : str
        The filename of the output file
    """
    output_lines = []
    output_lines.extend(_selfcal_report_lines(field))
    output_lines.append("\n")
    output_lines.extend(_calibration_report_lines(field))
    output_lines.append("\n")
    for sector in field.imaging_sectors:
        output_lines.extend(_image_report_lines(sector))
        output_lines.append("\n")

    if outfile is None:
        outfile = os.path.join(field.parset["dir_working"], "logs", "diagnostics.txt")
    with open(outfile, "w") as f:
        f.writelines(output_lines)


def _solution_intervals(field, steps):
    """Return the target interval in seconds and the DD factor of each solve run."""
    intervals = []
    for step in steps:
        if step["do_calibrate"]:
            strategy = step.get("calibration_strategy") or default_calibration_strategy()
            intervals.extend(
                _solution_interval(field, step, solve)
                for solves in strategy.values()
                for solve in solves
            )
    return intervals


def _solution_interval(field, step, solve):
    key = CALIBRATION_SOLVE_METADATA[solve].target_timestep_key
    target_timestep = step.get(key, getattr(field, key))
    # The DD interval factor lengthens all but the full-Jones solution intervals,
    # unless screens are generated (see Observation.set_calibration_parameters)
    if solve == "full_jones" or field.generate_screens:
        return target_timestep, 1
    return target_timestep, field.dd_interval_factor


def _select_data(field, steps, data_fraction):
    """Set the data fraction and the length of the selected ranges of each observation."""
    # The selection is made once, as the steps' intervals can fall back to field
    # settings that later cycles change
    intervals = _solution_intervals(field, steps)
    solve_time = max((target * factor for target, factor in intervals), default=None)
    _set_observation_data_fractions(field, data_fraction, solve_time)
    for obs in field.full_observations:
        obs.selection_timesteps = _chunk_timesteps(obs, intervals, data_fraction)


def _planned_observations(field, steps, data_fraction):
    max_nodes = field.parset["cluster_specific"]["max_nodes"]
    if data_fraction >= 1.0 and max_nodes <= 1:
        return field.full_observations

    intervals = _solution_intervals(field, steps)
    max_chunks = None if data_fraction < 1.0 else max_nodes
    return [
        chunk
        for obs in field.full_observations
        for chunk in obs.chunk_observation(
            # Without solves of their own, steps keep the selected ranges whole
            _chunk_timesteps(obs, intervals, data_fraction)
            if intervals
            else obs.selection_timesteps,
            max_chunks=max_chunks,
            selection_timesteps=obs.selection_timesteps,
        )
    ]


def _chunk_timesteps(obs, intervals, data_fraction):
    """Return the longest solution interval in time slots, as DP3 uses it."""
    if intervals:
        return max(obs.solution_interval_timesteps(*interval) for interval in intervals)
    # Without calibration, select 600 s of data at a time, or allow any chunk length
    return math.ceil(600.0 / obs.timepersample) if data_fraction < 1.0 else 1


def _chunk_layout(observations):
    # Different solution intervals can give the same chunks, for example when the
    # number of nodes limits them, so compare the chunks themselves
    return [(obs.name, obs.starttime, obs.endtime) for obs in observations]


def _set_observation_data_fractions(field, data_fraction, solve_time):
    for obs in field.full_observations:
        obs.data_fraction = data_fraction
        if solve_time is None:
            continue

        min_fraction = min(1.0, solve_time / (obs.endtime - obs.starttime))
        if data_fraction < min_fraction:
            obs.log.warning(
                "The specified value of data_fraction (%0.3f) results in "
                "a total time for this observation that is less than the "
                "largest potential calibration timestep (%.3f s). The data "
                "fraction will be increased to %0.3f to attempt to meet "
                "the timestep requirement.",
                data_fraction,
                solve_time,
                min_fraction,
            )
            obs.data_fraction = min_fraction


def _selfcal_report_lines(field):
    output_lines = ["Selfcal diagnostics:\n"]
    if field.selfcal_state:
        if field.selfcal_state.diverged:
            output_lines.append(
                f"  Selfcal diverged in cycle {field.cycle_number}. "
                "The final cycle was therefore skipped.\n"
            )
        elif field.selfcal_state.failed:
            output_lines.append(
                f"  Selfcal failed due to excessively high noise in cycle {field.cycle_number}. "
                "The final cycle was therefore skipped.\n"
            )
        elif field.do_final:
            output_lines.append(
                f"  Selfcal converged in cycle {field.cycle_number - 1} "
                "and a further, final cycle was done.\n"
            )
        else:
            output_lines.append(
                f"  Selfcal converged in cycle {field.cycle_number}. "
                "A final cycle was not done as it was not needed.\n"
            )
    else:
        output_lines.append("  No selfcal performed.\n")
    return output_lines


def _calibration_report_lines(field):
    output_lines = ["Calibration diagnostics:\n"]
    if not field.calibration_diagnostics:
        output_lines.append("  No calibration done.\n")
        return output_lines

    for index, diagnostics in enumerate(field.calibration_diagnostics):
        if index == 0:
            output_lines.append("  Fraction of solutions flagged:\n")
        output_lines.append(
            f"    cycle {diagnostics['cycle_number']}: "
            f"{diagnostics['solution_flagged_fraction']:.1f}\n"
        )
    return output_lines


def _image_report_lines(sector):
    output_lines = [f"Image diagnostics for {sector.name}:\n"]
    if not sector.diagnostics:
        output_lines.append("  No imaging done.\n")
        return output_lines

    min_rms_lines = ["  Minimum image noise (uJy/beam):\n"]
    median_rms_lines = ["  Median image noise (uJy/beam):\n"]
    dynamic_range_lines = ["  Image dynamic range:\n"]
    nsources_lines = ["  Number of sources found by PyBDSF:\n"]
    for diagnostics in sector.diagnostics:
        min_rms_lines.append(
            f"    cycle {diagnostics['cycle_number']}: "
            f"{diagnostics['min_rms_flat_noise'] * 1e6:.1f} (non-PB-corrected), "
            f"{diagnostics['min_rms_true_sky'] * 1e6:.1f} (PB-corrected), "
            f"{diagnostics['theoretical_rms'] * 1e6:.1f} (theoretical)\n"
        )
        median_rms_lines.append(
            f"    cycle {diagnostics['cycle_number']}: "
            f"{diagnostics['median_rms_flat_noise'] * 1e6:.1f} (non-PB-corrected), "
            f"{diagnostics['median_rms_true_sky'] * 1e6:.1f} (PB-corrected)\n"
        )
        dynamic_range_lines.append(
            f"    cycle {diagnostics['cycle_number']}: "
            f"{diagnostics['dynamic_range_global_true_sky']:.1f}\n"
        )
        nsources_lines.append(
            f"    cycle {diagnostics['cycle_number']}: {diagnostics['nsources']}\n"
        )

    output_lines.extend(min_rms_lines)
    output_lines.extend(median_rms_lines)
    output_lines.extend(dynamic_range_lines)
    output_lines.extend(nsources_lines)
    return output_lines

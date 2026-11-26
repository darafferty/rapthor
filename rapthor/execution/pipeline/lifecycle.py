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
    field.update_observations(_chunked_observations(_chunk_plan(field, steps)))


def rechunk_observations(field, steps, index):
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

    Returns
    -------
    step : dict
        The strategy step to run the cycle with. When the observations are chunked
        again, this is a copy with the peeling flag set (the strategy is unchanged)
    """
    step = steps[index]
    if not step["do_calibrate"]:
        return step

    plan = _chunk_plan(field, [step])
    if _is_chunked_as_planned(field.observations, plan):
        return step

    field.update_observations(_chunked_observations(plan))
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
    # A step inherits the settings it omits from the steps before it, as field.update
    # carries them forward, and from the field's current settings before those
    settings = {}
    for step in steps:
        settings.update(step)
        if step["do_calibrate"]:
            strategy = step.get("calibration_strategy") or default_calibration_strategy()
            intervals.extend(
                _solution_interval(field, settings, solve)
                for solve in _timed_solves(field, strategy)
            )
    return intervals


def _timed_solves(field, strategy):
    """Yield the solves whose solution intervals the calibration uses."""
    for mode, solves in strategy.items():
        if mode == "dd" and field.generate_screens and solves:
            # Screens are solved at the interval of the first DD solve (its slot sets
            # solint_solve1_timestep, see Calibrate._apply_solve_plan_inputs), and of the
            # slow gains for amplitudes (see execution.calibrate.solves)
            yield solves[0]
            if "slow_gains" in solves:
                yield "slow_gains"
        else:
            yield from solves


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
        # Without calibration, select 600 s of data at a time, or allow any chunk length
        default = math.ceil(600.0 / obs.timepersample) if data_fraction < 1.0 else 1
        obs.selection_timesteps = _chunk_timesteps(obs, intervals, default)


def _chunk_plan(field, steps):
    """Return the observation, and the name, start and end time, of each planned chunk."""
    intervals = _solution_intervals(field, steps)
    # A node count of 0 (unset) can reach here from the execution settings
    max_nodes = max(1, field.parset["cluster_specific"]["max_nodes"])
    return [
        (obs, times)
        for obs in field.full_observations
        for times in _observation_chunk_times(obs, intervals, max_nodes)
    ]


def _observation_chunk_times(obs, intervals, max_nodes):
    """Return the name, start and end time of each chunk of an observation."""

    def chunk_times(chunk_timesteps):
        return obs.chunk_times(
            chunk_timesteps, max_chunks=max_nodes, selection_timesteps=obs.selection_timesteps
        )

    # The selection was made for the longest solution interval of all of the steps
    # it is shared by. Chunks of that interval are the same in each of these steps, so
    # a step uses chunks of its own (shorter) interval only if there are more of them
    # and the longest is shorter, as otherwise they would be chunked (and peeled) again
    # for no more nodes or no faster solves. Steps without solves of their own keep the
    # selected ranges whole
    chunk_timesteps = _chunk_timesteps(obs, intervals, obs.selection_timesteps)
    own_chunk_times = chunk_times(chunk_timesteps)
    if chunk_timesteps >= obs.selection_timesteps:
        return own_chunk_times
    shared_chunk_times = chunk_times(obs.selection_timesteps)
    more_chunks = len(own_chunk_times) > len(shared_chunk_times)
    shorter_chunks = _longest_chunk_timesteps(obs, own_chunk_times) < _longest_chunk_timesteps(
        obs, shared_chunk_times
    )
    return own_chunk_times if more_chunks and shorter_chunks else shared_chunk_times


def _longest_chunk_timesteps(obs, chunk_times):
    # Count the time slots, as the start and end times are computed differently
    return max(
        round((endtime - starttime) / obs.timepersample) for _, starttime, endtime in chunk_times
    )


def _chunked_observations(plan):
    return [obs.chunk(*times) for obs, times in plan]


def _chunk_timesteps(obs, intervals, default):
    """Return the longest solution interval in time slots, as DP3 uses it (or default)."""
    return max(
        (obs.solution_interval_timesteps(*interval) for interval in intervals), default=default
    )


def _is_chunked_as_planned(observations, plan):
    # Different solution intervals can give the same chunks, for example when the
    # number of nodes limits them, so compare the chunks themselves. They are compared
    # as planned, before making the chunks scans the data
    return [obs.planned_times for obs in observations] == [times for _, times in plan]


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

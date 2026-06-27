"""Scheduler-independent process lifecycle helpers."""

import logging
import os

import numpy as np

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


def chunk_observations(field, steps, data_fraction):
    """
    Chunk observations in time for calibration, data-fraction, or multi-node runs.

    The resulting data fraction may differ from observation to observation
    depending on the length of each observation and the largest requested
    calibration solution interval.
    """
    solve_time = _calibration_solve_time(field, steps)
    chunk_time = _chunk_time_for_run(field, data_fraction, solve_time)
    if chunk_time is None:
        field.update_observations(field.full_observations)
        return

    _set_observation_data_fractions(field, data_fraction, solve_time)
    field.chunk_observations(chunk_time)


def _calibration_solve_time(field, steps):
    if not steps or not any(step["do_calibrate"] for step in steps):
        return None

    fast_solint = max(step.get("fast_timestep_sec", 0) for step in steps)
    slow_solint = max(step.get("slow_timestep_sec", 0) for step in steps)
    max_dd_timestep = max(fast_solint, slow_solint)
    max_di_timestep = max(step.get("fulljones_timestep_sec", 0) for step in steps)

    return max(max_dd_timestep * field.dd_interval_factor, max_di_timestep)


def _chunk_time_for_run(field, data_fraction, solve_time):
    max_nodes = field.parset["cluster_specific"]["max_nodes"]
    if data_fraction < 1.0:
        return solve_time or 600.0

    if max_nodes <= 1:
        return None

    split_time = min(
        (obs.endtime - obs.starttime) / max_nodes - obs.timepersample / 10
        for obs in field.full_observations
    )
    return max(solve_time or 0, split_time)


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

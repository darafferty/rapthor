"""
Module that performs the processing
"""

import logging
import os

import numpy as np

from rapthor.operations.calibrate import Calibrate
from rapthor.operations.image import Image, ImageNormalize
from rapthor.operations.mosaic import Mosaic
from rapthor.operations.predict import Predict

log = logging.getLogger("rapthor")


def run(parset_file, logging_level="info"):
    """
    Process a dataset using the Prefect/Dask execution path.

    Parameters
    ----------
    parset_file : str
        Filename of parset containing processing parameters
    logging_level : str, optional
        One of 'debug', 'info', 'warning' in decreasing order of verbosity
    """
    from rapthor.execution.flows.process import process_flow

    return process_flow(parset_file, logging_level=logging_level)


def run_steps(field, steps, final=False):
    """
    Run one group of processing steps through the Prefect process-step scheduler.

    The public helper remains here for compatibility with older callers and
    tests, but operation ordering now lives in
    ``rapthor.execution.flows.process.run_process_steps``.

    Parameters
    ----------
    field : Field object
        The Field object for this run
    steps : list of dict
        List of strategy step dicts containing the processing parameters
    final : bool, optional
        If True, process as the final pass
    """
    from rapthor.execution.flows.process import ProcessOperationFactories, run_process_steps

    factories = ProcessOperationFactories(
        predict=Predict,
        calibrate=Calibrate,
        image=Image,
        mosaic=Mosaic,
        image_normalize=ImageNormalize,
    )
    return run_process_steps(field, steps, final=final, operation_factories=factories)


def do_final_pass(field, selfcal_steps, final_step):
    """
    Check the processing state to determine whether a final pass is needed

    A final pass is needed when:
        - selfcal was not done
        - selfcal was done, but:
            - the final data fraction is different from the selfcal one, or
            - QUV images are to be made, or
            - the parameters for the final pass differ from those of the last
              cycle of selfcal

    Parameters
    ----------
    field : Field object
        The Field object for this run
    selfcal_steps : list of dicts
        List of strategy step dicts containing the selfcal processing parameters
    final_step : dict
        Dict containing the processing parameters for the final pass

    Returns
    -------
    final_pass : bool
        True is a final pass is needed and False if not
    """
    if not selfcal_steps:
        # No selfcal was done, final pass needed
        final_pass = True
    else:
        # Selfcal was done
        if field.do_check and (field.selfcal_state.diverged or field.selfcal_state.failed):
            # Selfcal was found to have diverged or failed, so don't do the final pass
            # even if required otherwise
            log.warning(
                "Selfcal diverged or failed, so skipping final cycle (with a "
                "data fraction of %.2f)",
                field.parset["final_data_fraction"],
            )
            final_pass = False
        elif final_step == selfcal_steps[field.cycle_number - 1]:
            # Selfcal successful, but the strategy parameters of the final pass are
            # identical to those of the last step of selfcal. Only do final pass if
            # required by other settings
            if (
                not np.isclose(
                    field.parset["final_data_fraction"],
                    field.parset["selfcal_data_fraction"],
                )
                or field.make_quv_images
            ):
                # Parset parameters require final pass
                final_pass = True
            else:
                # Final pass not needed
                final_pass = False
        else:
            # Selfcal successful, and the strategy parameters of the final pass differ
            # from those of the last step of selfcal
            final_pass = True

    return final_pass


def chunk_observations(field, steps, data_fraction):
    """
    Chunks observations in time

    The resulting data fraction may differ from observation to observation
    depending on the length of each observation.

    Parameters
    ----------
    field : Field object
        The Field object for this run
    steps : list of dicts
        List of strategy step dicts containing the processing parameters
    data_fraction : float
        The target data fraction
    """
    # Find the minimum duration that is needed for the solves
    if steps and any(step["do_calibrate"] for step in steps):
        # When calibration is to be done, use the solution intervals to
        # set the minimum duration
        fast_solint = max(step.get("fast_timestep_sec", 0) for step in steps)
        slow_solint = max(step.get("slow_timestep_sec", 0) for step in steps)
        max_dd_timestep = max(fast_solint, slow_solint)
        max_di_timestep = max(step.get("fulljones_timestep_sec", 0) for step in steps)

        # For DD solves, include the effect of DD solution intervals (given by
        # dd_interval_factor), which increases the solution intervals. This effect
        # does not apply to the DI solves
        solve_time = max(max_dd_timestep * field.dd_interval_factor, max_di_timestep)  # sec
    else:
        # No calibration to be done
        solve_time = None

    # Set the chunking time. Chunking is only needed when the data fraction
    # is less than one (so that the uv coverage can be optimized in this case) or when
    # there is more than one node (so the processing can be parallelized efficiently)
    max_nodes = field.parset["cluster_specific"]["max_nodes"]
    if data_fraction < 1.0:
        # Use the minmum duration set by the calibration. If no calibration is to be
        # done, set the minimum duration to a typical value (600 s) that should result
        # in enough chunks to obtain good uv coverage
        chunk_time = solve_time or 600.0
    elif max_nodes > 1:
        # Set the minimum duration that results in at least as many chunks for each
        # observation as there are nodes (for parallelization over nodes)
        #
        # Note: we reduce the duration slightly to avoid making fewer chunks than
        # desired due to rounding done during the chunking
        split_time = min(
            (obs.endtime - obs.starttime) / max_nodes - obs.timepersample / 10
            for obs in field.full_observations
        )

        # Use the largest of the solve and split times as the chunking time
        chunk_time = max(solve_time or 0, split_time)
    else:
        # Chunking not needed: use the original (full) observations
        field.update_observations(field.full_observations)
        return

    # Before chunking, set the data fraction per observation, increasing it if needed
    # to meet the solve requirements
    for obs in field.full_observations:
        obs.data_fraction = data_fraction
        if solve_time is not None:
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

    field.chunk_observations(chunk_time)


def make_report(field, outfile=None):
    """
    Make a summary report of QA metrics for the run

    Parameters
    ----------
    field : Field object
        The Field object for this run
    outfile : str
        The filename of the output file
    """
    # Report selfcal convergence
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
        else:
            if field.do_final:
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
    output_lines.append("\n")

    # Report calibration diagnostics: these are stored in field.calibration_diagnostics
    output_lines.append("Calibration diagnostics:\n")
    if not field.calibration_diagnostics:
        output_lines.append("  No calibration done.\n")
    else:
        for index, diagnostics in enumerate(field.calibration_diagnostics):
            if index == 0:
                output_lines.append("  Fraction of solutions flagged:\n")
            output_lines.append(
                f"    cycle {diagnostics['cycle_number']}: "
                f"{diagnostics['solution_flagged_fraction']:.1f}\n"
            )
    output_lines.append("\n")

    # Report imaging diagnostics: these are stored for each sector and cycle in
    # sector.diagnostics
    for sector in field.imaging_sectors:
        output_lines.append(f"Image diagnostics for {sector.name}:\n")
        if not sector.diagnostics:
            output_lines.append("  No imaging done.\n")
        else:
            for index, diagnostics in enumerate(sector.diagnostics):
                if index == 0:
                    min_rms_lines = ["  Minimum image noise (uJy/beam):\n"]
                    median_rms_lines = ["  Median image noise (uJy/beam):\n"]
                    dynamic_range_lines = ["  Image dynamic range:\n"]
                    nsources_lines = ["  Number of sources found by PyBDSF:\n"]
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
        output_lines.append("\n")

    # Open output file
    if outfile is None:
        outfile = os.path.join(field.parset["dir_working"], "logs", "diagnostics.txt")
    with open(outfile, "w") as f:
        f.writelines(output_lines)


def _do_calibrate_mode(calibration_strategy):
    """
    Determine whether DI and/or DD calibration is enabled.

    Parameters
    ----------
    calibration_strategy : dict
        The calibration strategy for this run
    """
    from rapthor.execution.flows.process import _do_calibrate_mode as process_do_calibrate_mode

    return process_do_calibrate_mode(calibration_strategy)

"""
Module that performs the processing
"""
import logging
import os
from rapthor import _logging
from rapthor.lib.parset import parset_read
from rapthor.lib.strategy import set_strategy
from rapthor.operations.concatenate import Concatenate
from rapthor.operations.calibrate import CalibrateDD, CalibrateDI
from rapthor.operations.image import Image, ImageInitial, ImageNormalize
from rapthor.operations.mosaic import Mosaic
from rapthor.operations.predict import PredictDD, PredictDI, PredictNC
from rapthor.lib.field import Field
import numpy as np

log = logging.getLogger('rapthor')


def run(parset_file, logging_level='info'):
    """
    Processes a dataset using direction-dependent calibration and imaging

    Parameters
    ----------
    parset_file : str
        Filename of parset containing processing parameters
    logging_level : str, optional
        One of 'degug', 'info', 'warning' in decreasing order of verbosity
    """
    # Read parset
    parset = parset_read(parset_file)

    # Set up logger
    log.info("Setting log level to %s", logging_level.upper())
    _logging.set_level(logging_level)

    # Initialize field object and do concatenation if needed
    field = Field(parset)
    if any([len(obs) > 1 for obs in field.epoch_observations]):
        log.info("MS files with different frequencies found for one "
                 "or more epochs. Concatenation over frequency will be done.")
        op = Concatenate(field, 1)
        op.run()

    # Set the processing strategy
    strategy_steps = set_strategy(field)
    if strategy_steps:
        selfcal_steps = strategy_steps[:-1]  # can be an empty list (when no selfcal needed)
        final_step = strategy_steps[-1]
    else:
        log.warning("The strategy '{}' does not define any processing steps. No "
                    "processing can be done.".format(parset['strategy']))
        return

    # Generate an initial sky model from the input data if needed
    if parset['generate_initial_skymodel']:
        if not any([step['do_calibrate'] for step in strategy_steps]):
            log.warning("Generation of an initial sky model has been activated but "
                        "the strategy '{}' does not contain any calibration steps. "
                        "Skipping the initial skymodel generation...".format(parset['strategy']))
            field.parset['generate_initial_skymodel'] = False
        else:
            field.define_full_field_sector(radius=parset['generate_initial_skymodel_radius'])
            log.info("Imaging full field to generate an initial sky model...")
            chunk_observations(field, [], parset['generate_initial_skymodel_data_fraction'])
            op = ImageInitial(field)
            op.run()

    # Run the self calibration
    if selfcal_steps:
        log.info("Starting self calibration with a data fraction of "
                 "{0:.2f}".format(parset['selfcal_data_fraction']))

        # Set the data chunking to match the longest solution interval set in
        # the strategy
        chunk_observations(field, selfcal_steps, parset['selfcal_data_fraction'])
        run_steps(field, selfcal_steps)

    # Run a final pass if needed
    field.do_final = do_final_pass(field, selfcal_steps, final_step)
    if field.do_final:
        if selfcal_steps:
            if not any([len(obs) > 1 for obs in field.epoch_observations]):
                # Use concatenation was not done the user input data column
                # for the final calibration run
                field.data_colname = parset['data_colname']

            # If selfcal was done, set peel_outliers to that of the initial cycle, since the
            # observations will be regenerated and outliers (if any) need to be peeled again
            final_step['peel_outliers'] = selfcal_steps[0]['peel_outliers']
            log.info("Starting final cycle with a data fraction of "
                     "{0:.2f}".format(parset['final_data_fraction']))
            field.cycle_number += 1
        else:
            if not final_step['do_calibrate']:
                if not parset["input_h5parm"]:
                    raise ValueError("The stratgey indicates that no calibration is to be done "
                                     "but no calibration solutions were provided. Please provide "
                                     "the solutions with the input_h5parm parameter")
                elif (
                    (final_step['peel_outliers'] or final_step['peel_bright_sources']) and
                    not parset["input_skymodel"]
                ):
                    raise ValueError("Peeling of outliers or bright sources was activated but no "
                                     "sky model was provided. Please provide a sky model with the "
                                     "input_skymodel parameter")
                else:
                    # Turn off conflicting flags
                    field.parset['generate_initial_skymodel'] = False
                    field.parset['download_initial_skymodel'] = False
            log.info("Using a data fraction of {0:.2f}".format(parset['final_data_fraction']))

        if field.make_quv_images:
            log.info("Stokes I, Q, U, and V images will be made")
        if field.dde_mode == 'hybrid':
            log.info("Screens will be used for calibration and imaging (since dde_mode = "
                     "'hybrid' and this is the final cycle)")
            if final_step['peel_outliers']:
                # Currently, when screens are used peeling cannot be done
                log.warning("Peeling of outliers is currently not supported when using "
                            "screens. Peeling will be skipped")
                final_step['peel_outliers'] = False

        # Set the data chunking to match the longest solution interval set in
        # the strategy
        chunk_observations(field, [final_step], parset['final_data_fraction'])

        run_steps(field, [final_step], final=True)

    # Make a summary report for the run and finish
    make_report(field)
    log.info("Rapthor has finished :)")


def run_steps(field, steps, final=False):
    """
    Runs the steps in a reduction

    This function runs the operations in the correct order and handles all the
    bookkeeping for the processing

    Parameters
    ----------
    field : Field object
        The Field object for this run
    steps : list of dict
        List of strategy step dicts containing the processing parameters
    final : bool, optional
        If True, process as the final pass
    """
    # Run the self calibration part of the strategy (if any)
    for index, step in enumerate(steps):

        # Update the field object for the current step
        cycle_number = index + field.cycle_number
        field.update(step, cycle_number, final=final)

        # Calibrate
        if field.do_calibrate:
            # Set whether screens should be generated
            field.generate_screens = True if (field.dde_mode == 'hybrid' and final) else False

            if field.peel_non_calibrator_sources:
                # Predict and subtract non-calibrator sources before calibration
                op = PredictNC(field, cycle_number)
                op.run()

            # Calibrate (direction-dependent)
            op = CalibrateDD(field, cycle_number)
            op.run()

            # Calibrate (direction-independent)
            if field.do_fulljones_solve:
                op = PredictDI(field, cycle_number)
                op.run()
                op = CalibrateDI(field, cycle_number)
                op.run()

        # Predict and subtract the sector models
        # Note: DD predict is not yet supported when screens are used
        if field.do_predict and not field.generate_screens:
            op = PredictDD(field, cycle_number)
            op.run()

        # Image and mosaic the sectors
        if field.do_image:
            # Normalize the flux scale
            if field.do_normalize:
                # Define sector and set the Stokes polarization to I
                field.define_normalize_sector()
                field.image_pol = 'I'

                op = ImageNormalize(field, cycle_number)
                op.run()

            # Set the Stokes polarizations for imaging
            field.image_pol = 'IQUV' if (field.make_quv_images and final) else 'I'

            # Set whether screens should be applied
            field.apply_screens = True if (field.dde_mode == 'hybrid' and final) else False

            # Set whether the final major iteration is skipped (note: it is never skipped
            # for the final iteration)
            field.skip_final_major_iteration = False if final else field.parset['imaging_specific']['skip_final_major_iteration']

            op = Image(field, cycle_number)
            op.run()

            op = Mosaic(field, cycle_number)
            op.run()

        # Check for selfcal convergence/divergence
        if field.do_check and not final:
            log.info("Checking selfcal convergence...")
            selfcal_state = field.check_selfcal_progress()
            if not any(selfcal_state):
                # Continue selfcal
                log.info("Improvement in image noise, dynamic range, and/or number of "
                         "sources exceeds that set by the convergence ratio of "
                         "{0}.".format(field.convergence_ratio))
                log.info("Continuing selfcal...")
            else:
                # Stop selfcal
                if selfcal_state.converged:
                    log.info("Selfcal has converged (improvement in image noise, dynamic "
                             "range, and number of sources does not exceed that set by the "
                             "convergence ratio of {0})".format(field.convergence_ratio))
                if selfcal_state.diverged:
                    log.warning("Selfcal has diverged (ratio of current image noise "
                                "to previous value is > {})".format(field.divergence_ratio))
                if selfcal_state.failed:
                    log.warning("Selfcal has failed due to high noise (ratio of current image noise "
                                "to theoretical value is > {})".format(field.failure_ratio))
                log.info("Stopping selfcal at cycle {0} of {1}".format(cycle_number, len(steps)))
                break
        else:
            selfcal_state = None

    field.selfcal_state = selfcal_state
    field.cycle_number = cycle_number


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
            log.warning("Selfcal diverged or failed, so skipping final cycle (with a data "
                        "fraction of {0:.2f})".format(field.parset['final_data_fraction']))
            final_pass = False
        elif final_step == selfcal_steps[field.cycle_number-1]:
            # Selfcal successful, but the strategy parameters of the final pass are
            # identical to those of the last step of selfcal. Only do final pass if
            # required by other settings
            if not np.isclose(field.parset['final_data_fraction'],
                              field.parset['selfcal_data_fraction']) or field.make_quv_images:
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
    # Find the overall minimum duration that can be used and still satisfy the
    # specified solution intervals
    min_time = None
    if steps and any([step['do_calibrate'] for step in steps]):
        fast_solint = max([step['fast_timestep_sec'] if 'fast_timestep_sec'
                           in step else 0 for step in steps])
        slow_solint = max([step['slow_timestep_sec'] if 'slow_timestep_sec'
                           in step else 0 for step in steps])
        max_dd_timestep = max(fast_solint, slow_solint)
        max_di_timestep = field.fulljones_timestep_sec
        min_time = max(max_dd_timestep * field.dd_interval_factor, max_di_timestep)

    for obs in field.full_observations:
        tot_time = obs.endtime - obs.starttime
        obs.data_fraction = data_fraction
        if min_time:
            min_fraction = min(1.0, min_time/tot_time)
            if data_fraction < min_fraction:
                obs.log.warning('The specified value of data_fraction ({0:0.3f}) results in a '
                                'total time for this observation that is less than the largest '
                                'potential calibration timestep ({1} s). The data fraction will be '
                                'increased to {2:0.3f} to atempt to meet the timestep '
                                'requirement.'.format(data_fraction, min_time, min_fraction))
                obs.data_fraction = min_fraction

    field.chunk_observations(min_time)


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
    output_lines = ['Selfcal diagnostics:\n']
    if field.selfcal_state:
        if field.selfcal_state.diverged:
            output_lines.append(f'  Selfcal diverged in cycle {field.cycle_number}. '
                                'The final cycle was therefore skipped.\n')
        elif field.selfcal_state.failed:
            output_lines.append(f'  Selfcal failed due to excessively high noise in cycle {field.cycle_number}. '
                                'The final cycle was therefore skipped.\n')
        else:
            if field.do_final:
                output_lines.append(f'  Selfcal converged in cycle {field.cycle_number - 1} '
                                    'and a further, final cycle was done.\n')
            else:
                output_lines.append(f'  Selfcal converged in cycle {field.cycle_number}. '
                                    'A final cycle was not done as it was not needed.\n')
    else:
        output_lines.append('  No selfcal performed.\n')
    output_lines.append('\n')

    # Report calibration diagnostics: these are stored in field.calibration_diagnostics
    output_lines.append('Calibration diagnostics:\n')
    if not field.calibration_diagnostics:
        output_lines.append(f'  No calibration done.\n')
    else:
        for index, diagnostics in enumerate(field.calibration_diagnostics):
            if index == 0:
                output_lines.append(f'  Fraction of solutions flagged:\n')
            output_lines.append(f"    cycle {diagnostics['cycle_number']}: "
                                f"{diagnostics['solution_flagged_fraction']:.1f}\n")
    output_lines.append('\n')

    # Report imaging diagnostics: these are stored for each sector and cycle in
    # sector.diagnostics
    for sector in field.imaging_sectors:
        output_lines.append(f'Image diagnostics for {sector.name}:\n')
        if not sector.diagnostics:
            output_lines.append(f'  No imaging done.\n')
        else:
            for index, diagnostics in enumerate(sector.diagnostics):
                if index == 0:
                    min_rms_lines = ["  Minimum image noise (uJy/beam):\n"]
                    median_rms_lines = ["  Median image noise (uJy/beam):\n"]
                    dynamic_range_lines = ["  Image dynamic range:\n"]
                    nsources_lines = ["  Number of sources found by PyBDSF:\n"]
                min_rms_lines.append(f"    cycle {diagnostics['cycle_number']}: "
                                     f"{diagnostics['min_rms_flat_noise']*1e6:.1f} (non-PB-corrected), "
                                     f"{diagnostics['min_rms_true_sky']*1e6:.1f} (PB-corrected), "
                                     f"{diagnostics['theoretical_rms']*1e6:.1f} (theoretical)\n")
                median_rms_lines.append(f"    cycle {diagnostics['cycle_number']}: "
                                        f"{diagnostics['median_rms_flat_noise']*1e6:.1f} (non-PB-corrected), "
                                        f"{diagnostics['median_rms_true_sky']*1e6:.1f} (PB-corrected)\n")
                dynamic_range_lines.append(f"    cycle {diagnostics['cycle_number']}: "
                                           f"{diagnostics['dynamic_range_global_true_sky']:.1f}\n")
                nsources_lines.append(f"    cycle {diagnostics['cycle_number']}: {diagnostics['nsources']}\n")
            output_lines.extend(min_rms_lines)
            output_lines.extend(median_rms_lines)
            output_lines.extend(dynamic_range_lines)
            output_lines.extend(nsources_lines)
        output_lines.append('\n')

    # Open output file
    if outfile is None:
        outfile = os.path.join(field.parset["dir_working"], 'logs', 'diagnostics.txt')
    with open(outfile, 'w') as f:
        f.writelines(output_lines)

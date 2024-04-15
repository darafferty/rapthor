"""
Module that preforms the processing
"""
import logging
from rapthor import _logging
from rapthor.lib.parset import parset_read
from rapthor.lib.strategy import set_strategy
from rapthor.operations.concatenate import Concatenate
from rapthor.operations.calibrate import CalibrateDD, CalibrateDI
from rapthor.operations.image import Image
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
    parset['logging_level'] = logging_level
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

    # Run the self calibration
    if selfcal_steps:
        log.info("Starting self calibration with a data fraction of "
                 "{0:.2f}".format(parset['selfcal_data_fraction']))
        run_steps(field, selfcal_steps)

    # Run a final pass if needed
    if do_final_pass(field, selfcal_steps, final_step):
        if selfcal_steps:
            # If selfcal was done, set peel_outliers to that of initial iteration, since the
            # observations will be regenerated and outliers (if any) need to be peeled again
            final_step['peel_outliers'] = selfcal_steps[0]['peel_outliers']
            log.info("Starting final iteration with a data fraction of "
                     "{0:.2f}".format(parset['final_data_fraction']))
            field.cycle_number += 1
        else:
            log.info("Using a data fraction of {0:.2f}".format(parset['final_data_fraction']))
        if field.make_quv_images:
            log.info("Stokes I, Q, U, and V images will be made")
        run_steps(field, [final_step], final=True)

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
        if field.do_predict:
            op = PredictDD(field, cycle_number)
            op.run()

        # Image and mosaic the sectors
        if field.do_image:
            # Set the Stokes polarizations for imaging
            field.image_pol = 'IQUV' if (field.make_quv_images and final) else 'I'

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
                log.info("Stopping selfcal at iteration {0} of {1}".format(index+1, len(steps)))
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
            log.warning("Selfcal diverged or failed, so skipping final iteration (with a data "
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

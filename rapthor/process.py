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
from rapthor.operations.predict import PredictDD, PredictDI
from rapthor.lib.field import Field
import numpy as np

log = logging.getLogger('rapthor')


def run(parset_file, logging_level='info'):
    """
    Processes a dataset using direction-dependent calibration and imaging

    This function runs the operations in the correct order and handles all the
    bookkeeping for the processing

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
    selfcal_steps = strategy_steps[:-1]
    final_step = strategy_steps[-1]

    # Run the self calibration part of the strategy (if any)
    for index, step in enumerate(selfcal_steps):

        # Update the field object for the current step
        field.update(step, index+1)

        # Calibrate (direction-dependent)
        if field.do_calibrate:
            op = CalibrateDD(field, index+1)
            op.run()

            # Calibrate (direction-independent)
            if field.do_fulljones_solve:
                op = PredictDI(field, index+1)
                op.run()
                op = CalibrateDI(field, index+1)
                op.run()

        # Predict and subtract the sector models
        if field.do_predict:
            op = PredictDD(field, index+1)
            op.run()

        # Image and mosaic the sectors
        if field.do_image:
            # Since we're doing selfcal, ensure that only Stokes I is imaged
            field.image_pol = 'I'

            op = Image(field, index+1)
            op.run()

            op = Mosaic(field, index+1)
            op.run()

        # Check for selfcal convergence/divergence
        if field.do_check:
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
                log.info("Stopping selfcal at iteration {0} of {1}".format(index+1, len(strategy_steps)))
                break

    # Run a final pass if needed. It is needed when:
    #    - selfcal was not done
    #    - selfcal was done, but:
    #        - the final data fraction is different from the selfcal one, or
    #        - QUV images are to be made
    if not selfcal_steps:
        do_final_pass = True
        index = 0
    elif (not np.isclose(parset['final_data_fraction'], parset['selfcal_data_fraction']) or
            field.make_quv_images):
        if field.do_check and (selfcal_state.diverged or selfcal_state.failed):
            # If selfcal was found to have diverged or failed, don't do the final pass
            # even if required otherwise
            log.warning("Selfcal diverged or failed, so skipping final iteration (with a data "
                        "fraction of {0:.2f})".format(parset['final_data_fraction']))
            do_final_pass = False
        else:
            do_final_pass = True
            index += 1  # increment index for final iteration
    else:
        do_final_pass = False

    if do_final_pass:
        if selfcal_steps:
            # If selfcal was done, set peel_outliers to that of initial iteration, since the
            # observations will be regenerated and outliers (if any) need to be peeled again
            final_step['peel_outliers'] = selfcal_steps[0]['peel_outliers']
            log.info("Starting final iteration with a data fraction of "
                     "{0:.2f}".format(parset['final_data_fraction']))
        else:
            log.info("Using a data fraction of {0:.2f}".format(parset['final_data_fraction']))
        if field.make_quv_images:
            log.info("Stokes I, Q, U, and V images will be made")

        # Now start the final processing pass
        field.update(final_step, index+1, final=True)

        # Calibrate (direction-dependent)
        if field.do_calibrate:
            op = CalibrateDD(field, index+1)
            op.run()

            # Calibrate (direction-independent)
            if field.do_fulljones_solve:
                op = PredictDI(field, index+1)
                op.run()
                op = CalibrateDI(field, index+1)
                op.run()

        # Predict and subtract the sector models
        if field.do_predict:
            op = PredictDD(field, index+1)
            op.run()

        # Image and mosaic the sectors
        if field.do_image:
            # Set the Stokes polarizations for imaging
            field.image_pol = 'IQUV' if field.make_quv_images else 'I'

            op = Image(field, index+1)
            op.run()

            op = Mosaic(field, index+1)
            op.run()

    log.info("Rapthor has finished :)")

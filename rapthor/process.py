"""
Module that preforms the processing
"""
import logging
from rapthor import _logging
from rapthor.lib.parset import parset_read
from rapthor.lib.strategy import set_strategy
from rapthor.operations.calibrate import Calibrate
from rapthor.operations.image import Image
from rapthor.operations.mosaic import Mosaic
from rapthor.operations.predict import Predict
from rapthor.lib.field import Field
import numpy as np

log = logging.getLogger('rapthor')


def run(parset_file, logging_level='info'):
    """
    Processes a dataset using DDE calibration and screens

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

    # Initialize field object
    field = Field(parset)

    # Set the processing strategy
    strategy_steps = set_strategy(field)

    # Run the strategy
    for index, step in enumerate(strategy_steps):

        # Update the field object for the current step
        field.update(step, index+1)

        # Calibrate
        if field.do_calibrate:
            op = Calibrate(field, index+1)
            op.run()

        # Predict and subtract the sector models
        if field.do_predict:
            op = Predict(field, index+1)
            op.run()

        # Image the sectors
        # Since we're doing selfcal, ensure that only Stokes I is imaged
        field.image_pol = 'I'
        if field.do_image:
            op = Image(field, index+1)
            op.run()

            # Mosaic the sectors
            op = Mosaic(field, index+1)
            op.run()

        # Check for selfcal convergence/divergence
        if field.do_check:
            log.info("Checking selfcal convergence...")
            has_converged, has_diverged = field.check_selfcal_progress()
            if not has_converged and not has_diverged:
                # Continue selfcal
                log.info("Improvement in image noise, dynamic range, and/or number of "
                         "sources exceeds that set by the convergence ratio of "
                         "{0}.".format(field.convergence_ratio))
                log.info("Continuing selfcal...")
            else:
                # Stop selfcal
                if has_converged:
                    log.info("Selfcal has converged (improvement in image noise, dynamic "
                             "range, and number of sources does not exceed that set by the "
                             "convergence ratio of {0})".format(field.convergence_ratio))
                if has_diverged:
                    log.warning("Selfcal has diverged (ratio of current image noise "
                                "to previous value is > {})".format(field.divergence_ratio))
                log.info("Stopping selfcal at iteration {0} of {1}".format(index+1, len(strategy_steps)))
                break

    # Run a final pass if needed
    if (not np.isclose(parset['final_data_fraction'], parset['selfcal_data_fraction']) or
            field.make_quv_images):
        log.info("Starting final iteration with a data fraction of "
                 "{0:.2f}".format(parset['final_data_fraction']))
        if field.make_quv_images:
            log.info("Stokes I, Q, U, and V images will be made")

        # Set peel_outliers to that of initial iteration, since the observations
        # will be regenerated and outliers may need to be peeled
        step['peel_outliers'] = strategy_steps[0]['peel_outliers']

        # Now start the final processing pass, incrementing the iteration index
        # from that of the last selfcal iteration (so to index+2)
        field.update(step, index+2, final=True)

        # Calibrate
        if field.do_calibrate:
            op = Calibrate(field, index+2)
            op.run()

        # Predict and subtract the sector models
        if field.do_predict:
            op = Predict(field, index+2)
            op.run()

        # Image the sectors
        # Set whether all Stokes parameters should be imaged or just I
        if field.make_quv_images:
            field.image_pol = 'IQUV'
        else:
            field.image_pol = 'I'
        if field.do_image:
            op = Image(field, index+2)
            op.run()

            # Mosaic the sectors
            op = Mosaic(field, index+2)
            op.run()

    log.info("Rapthor has finished :)")

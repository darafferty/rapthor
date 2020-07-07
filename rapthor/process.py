"""
Module that preforms the processing
"""
import logging
from rapthor import _logging
from rapthor.parset import parset_read
from rapthor.strategy import set_strategy
from rapthor.operations.calibrate import Calibrate
from rapthor.operations.image import Image
from rapthor.operations.mosaic import Mosaic
from rapthor.operations.predict import Predict
from rapthor.lib.field import Field

log = logging.getLogger('rapthor')


def run(parset_file, logging_level='info', sectors_to_export=[], export_corrected_data=False):
    """
    Processes a dataset using facet calibration

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
    for iter, step in enumerate(strategy_steps):
        # Update the field object for the current step
        field.update(step, iter+1)

        # Calibrate
        if step['do_calibrate']:
            op = Calibrate(field, iter+1)
            op.run()

        # Predict and subtract the sector models
        if step['do_predict']:
            op = Predict(field, iter+1)
            op.run()

        # Image the sectors
        if step['do_image']:
            op = Image(field, iter+1)
            op.run()

            # Mosaic the sectors, for now just Stokes I
            # TODO: run mosaic ops for IQUV+residuals
            op = Mosaic(field, iter+1)
            op.run()

        # Check for selfcal convergence
        if step['do_check']:
            has_converged = field.check_selfcal_convergence()
            if has_converged:
                break

    log.info("rapthor has finished :)")

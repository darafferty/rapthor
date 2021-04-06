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
import os

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

    # Initialize field and cal_field objects
    field = Field(parset)
    cal_parset = parset.copy()
    cal_parset['dir_working'] = os.path.join(parset['dir_working'], 'calibrators')
    cal_parset['imaging_specific']['use_screens'] = False
    cal_field = Field(cal_parset)

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
        if field.do_image:
            op = Image(field, index+1)
            op.run()

            # Mosaic the sectors, for now just Stokes I
            # TODO: run mosaic ops for IQUV+residuals
            op = Mosaic(field, index+1)
            op.run()

        # Check for selfcal convergence/divergence
        if field.do_check:
            has_converged, has_diverged = field.check_selfcal_progress()
            if has_converged or has_diverged:
                # Stop the cycle
                if has_converged:
                    log.info("Selfcal has converged (ratio of current image noise "
                             "to previous value is > {})".format(field.convergence_ratio))
                if has_diverged:
                    log.warning("Selfcal has diverged (ratio of current image noise "
                                "to previous value is > {})".format(field.divergence_ratio))
                log.info("Stopping at iteration {0} of {1}".format(index+1, len(strategy_steps)))
                break

        # Update the calibrator field object
        cal_field.h5parm_filename = field.h5parm_filename
        cal_field.define_cal_sectors()
        cal_field.__dict__.update(step)
        for sector in cal_field.imaging_sectors:
            sector.__dict__.update(step)
        cal_field.peel_bright_sources = False
        cal_field.peel_outliers = True
        cal_field.do_predict = True
        cal_field.do_image = True

        # Predict and subtract the calibrator sector models
        if cal_field.do_predict:
            op = Predict(cal_field, index+1)
            op.run()

        # Image the calibrator sectors
        if cal_field.do_image:
            op = Image(cal_field, index+1)
            op.run()

            # Mosaic the sectors, for now just Stokes I
            # TODO: run mosaic ops for IQUV+residuals
            op = Mosaic(cal_field, index+1)
            op.run()

            field.cal_sectors = cal_field.imaging_sectors

    log.info("Rapthor has finished :)")

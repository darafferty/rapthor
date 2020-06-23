"""
Module that modifies the state of the pipelines
"""
from rapthor.parset import parset_read
from rapthor.strategy import set_strategy
from rapthor.lib.field import Field
import logging
import os
import sys
import numpy as np

log = logging.getLogger('rapthor:state')
logging.getLogger('rapthor:parset').setLevel(logging.CRITICAL)
logging.getLogger('rapthor:strategy').setLevel(logging.CRITICAL)


def check_operation(operation):
    """
    Returns list of started/completed pipeline names for given operation path

    Parameters
    ----------
    operation : str
        Path of operation output
    """
    pipelines = []
    jobstore = os.path.join(operation, 'jobstore')
    if os.path.exists(jobstore):
        pipelines.append(os.path.basename(operation))

    return pipelines


def get_number_of_sectors(field):
    """
    Returns number of imaging sectors

    Parameters
    ----------
    field : Field object
        Input Field object
    """
    # Determine whether we use a user-supplied list of sectors or a grid
    if len(field.parset['imaging_specific']['sector_center_ra_list']) > 0:
        # Use user-supplied list
        sector_center_ra_list = field.parset['imaging_specific']['sector_center_ra_list']
        sector_center_dec_list = field.parset['imaging_specific']['sector_center_dec_list']
        sector_width_ra_deg_list = field.parset['imaging_specific']['sector_width_ra_deg_list']
        sector_width_dec_deg_list = field.parset['imaging_specific']['sector_width_dec_deg_list']
        n = 1
        for ra, dec, width_ra, width_dec in zip(sector_center_ra_list, sector_center_dec_list,
                                                sector_width_ra_deg_list, sector_width_dec_deg_list):
            n += 1
    else:
        # Make a regular grid of sectors
        if field.parset['imaging_specific']['grid_width_ra_deg'] is None:
            image_width_ra = field.fwhm_ra_deg
        else:
            image_width_ra = field.parset['imaging_specific']['grid_width_ra_deg']
        if field.parset['imaging_specific']['grid_width_dec_deg'] is None:
            image_width_dec = field.fwhm_dec_deg
        else:
            image_width_dec = field.parset['imaging_specific']['grid_width_dec_deg']

        nsectors_ra = field.parset['imaging_specific']['grid_nsectors_ra']
        if nsectors_ra == 0:
            # Force a single sector
            nsectors_ra = 1
            nsectors_dec = 1
        else:
            nsectors_dec = int(np.ceil(image_width_dec / (image_width_ra / nsectors_ra)))

        n = nsectors_ra*nsectors_dec

    return n


def run(parset_file):
    """
    Modifies the state of one or more pipelines

    Parameters
    ----------
    parset_file : str
        Filename of parset containing processing parameters
    """
    # Read parset
    log.info('Reading parset and checking state...')
    parset = parset_read(parset_file, use_log_file=False, skip_cluster=True)

    # Initialize field object
    field = Field(parset, mininmal=True)
    field.outlier_sectors = [None]
    field.imaging_sectors = [None] #* get_number_of_sectors(field)

    # Get the processing strategy
    strategy_steps = set_strategy(field)

    # Check each operation for started pipelines
    while True:
        pipelines = []
        for iter, step in enumerate(strategy_steps):
            if step['do_calibrate']:
                operation = os.path.join(parset['dir_working'], 'pipelines', 'calibrate_{}'.format(iter+1))
                pipelines.extend(check_operation(operation))
            if step['do_predict']:
                operation = os.path.join(parset['dir_working'], 'pipelines', 'predict_{}'.format(iter+1))
                pipelines.extend(check_operation(operation))
            if step['do_image']:
                operation = os.path.join(parset['dir_working'], 'pipelines', 'image_{}'.format(iter+1))
                pipelines.extend(check_operation(operation))
                operation = os.path.join(parset['dir_working'], 'pipelines', 'mosaic_{}'.format(iter+1))
                pipelines.extend(check_operation(operation))

        # List pipelines and query user
        print('\nCurrent strategy: {}'.format(field.parset['strategy']))
        print('\nPipelines:')
        i = 0
        if len(pipelines) == 0:
            print('    None')
        else:
            for p in pipelines:
                i += 1
                print('    {0}) {1}'.format(i, p))
        try:
            while(True):
                p_number_raw = input('Enter number of pipeline to reset or press "q" to quit: ')
                try:
                    if p_number_raw.lower() == "q":
                        sys.exit(0)
                    elif int(p_number_raw) > 0 and int(p_number_raw) <= i:
                        break
                    else:
                        print("Please enter a number between 1 and {}".format(i))
                except ValueError:
                    pass
        except KeyboardInterrupt:
            sys.exit(0)
        pipeline = pipelines[int(p_number_raw)-1]

        # Ask for confirmation
        try:
            while(True):
                answer = input('Reset all pipelines from {} onwards (y/n)? '.format(pipeline))
                if (answer.lower() == "n" or answer.lower() == "no" or
                    answer.lower() == "y" or answer.lower() == "yes"):
                    break
                else:
                    print('Please enter "y" or "n"')
        except KeyboardInterrupt:
            sys.exit(0)

        # Reset pipeline states as requested
        if answer.lower() == "y" or answer.lower() == "yes":
            print('Reseting state...')
            for pipeline in pipelines[int(p_number_raw)-1:]:
                jobstore = os.path.join(parset['dir_working'], 'pipelines', pipeline, 'jobstore')
                os.system('rm -rf {}'.format(jobstore))
            print('Reset complete.')

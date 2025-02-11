"""
Module that modifies the state of the pipeline
"""
from rapthor.lib.parset import parset_read
from rapthor.lib.strategy import set_strategy
from rapthor.lib.field import Field
import filecmp
import logging
import os
import shutil
import sys

log = logging.getLogger('rapthor:state')


def run(parset_file):
    """
    Modifies the state of one or more operations

    Parameters
    ----------
    parset_file : str
        Filename of parset containing processing parameters
    """
    # Set logging level to suppress unnecessary messages
    logging.getLogger('rapthor:parset').setLevel(logging.ERROR)
    logging.getLogger('rapthor:strategy').setLevel(logging.ERROR)

    # Read parset
    log.info('Reading parset and checking state...')
    parset = parset_read(parset_file, use_log_file=False)

    # Initialize minimal field object
    field = Field(parset, minimal=True)
    field.outlier_sectors = [None]
    field.imaging_sectors = [None]

    # Get the processing strategy
    strategy_steps = set_strategy(field)

    # Check each operation for started pipelines (workflows)
    # Note: the order here should match the order in which the operations were run
    operation_list = ['concatenate', 'initial_image', 'predict_nc', 'calibrate',
                      'predict_di', 'calibrate_di', 'predict', 'normalize',  'image',
                      'mosaic']
    while True:
        pipelines = []
        for index, step in enumerate(strategy_steps):
            for opname in operation_list:
                if index == 0 and opname == 'initial_image':
                    # Handle the initial sky model image operation separately, as it only
                    # occurs in the first cycle and does not include an index in its paths
                    operation = os.path.join(parset['dir_working'], 'pipelines', '{0}'.format(opname))
                else:
                    operation = os.path.join(parset['dir_working'], 'pipelines', '{0}_{1}'.format(opname, index+1))
                if os.path.exists(operation):
                    pipelines.append(os.path.basename(operation))

        # List operations and query user
        print('\nCurrent strategy: {}'.format(field.parset['strategy']))
        print('\nOperations:')
        i = 0
        if len(pipelines) == 0:
            print('    None')
            print('No reset can be done.')
            sys.exit(0)
        else:
            for p in pipelines:
                i += 1
                print('    {0}) {1}'.format(i, p))
        try:
            while(True):
                p_number_raw = input('Enter number of operation to reset or "q" to quit: ')
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
                answer = input('Reset all operations from {} onwards (y/n)? '.format(pipeline))
                if (answer.lower() == "n" or answer.lower() == "no" or
                    answer.lower() == "y" or answer.lower() == "yes"):
                    break
                else:
                    print('Please enter "y" or "n"')
        except KeyboardInterrupt:
            sys.exit(0)

        # Reset operation states as requested
        if answer.lower() == "y" or answer.lower() == "yes":
            print('Reseting state...')
            for pipeline in pipelines[int(p_number_raw)-1:]:
                # Remove the operation working directory to ensure files from previous
                # runs are not kept and used in subsequent ones (e.g., Toil does not
                # seem to always overwrite existing files from previous runs). This
                # also removes Toil's jobstore when present (where the state is tracked).
                # Other associated files are removed as well
                path = os.path.join(parset['dir_working'], 'pipelines', pipeline)
                shutil.rmtree(path, ignore_errors=True)

            # Now remove any sub-directories in the other output directories, that
            # are _not_ present in the 'pipelines' directory.
            for dirname in ('skymodels', 'solutions', 'logs', 'plots', 'regions', 'images'):
                dcmp = filecmp.dircmp(
                    os.path.join(parset['dir_working'], 'pipelines'),
                    os.path.join(parset['dir_working'], dirname)
                )
                for path in (os.path.join(dcmp.right, item) for item in dcmp.right_only):
                    shutil.rmtree(path, ignore_errors=True)

            print('Reset complete.')

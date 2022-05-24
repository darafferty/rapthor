"""
Module that modifies the state of the pipelines
"""
from rapthor.lib.parset import parset_read
from rapthor.lib.strategy import set_strategy
from rapthor.lib.field import Field
import logging
import os
import sys

log = logging.getLogger('rapthor:state')


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


def run(parset_file):
    """
    Modifies the state of one or more pipelines

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
    parset = parset_read(parset_file, use_log_file=False, skip_cluster=True)
    if parset['cwl_runner'] != 'toil':
        log.error("Modifying state is only supported when using Toil as CWL runner")
        sys.exit(1)

    # Initialize minimal field object
    field = Field(parset, mininmal=True)
    field.outlier_sectors = [None]
    field.imaging_sectors = [None]

    # Get the processing strategy
    strategy_steps = set_strategy(field)

    # Check each operation for started pipelines
    operation_list = ['calibrate', 'predict', 'image', 'mosaic']  # in order of execution
    while True:
        pipelines = []
        for index, step in enumerate(strategy_steps):
            for opname in operation_list:
                operation = os.path.join(parset['dir_working'], 'pipelines', '{0}_{1}'.format(opname, index+1))
                pipelines.extend(check_operation(operation))

        # List pipelines and query user
        print('\nCurrent strategy: {}'.format(field.parset['strategy']))
        print('\nPipelines:')
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

                # Remove associated sky models as well
                skymodel = os.path.join(parset['dir_working'], 'skymodels', pipeline)
                os.system('rm -rf {}'.format(skymodel))

            print('Reset complete.')

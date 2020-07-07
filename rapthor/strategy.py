"""
Module that holds all strategy-related functions
"""
import os
import logging
import sys
import runpy

log = logging.getLogger('rapthor:strategy')


def set_strategy(field):
    """
    Sets up the processing strategy

    Parameters
    ----------
    field : Field object
        Field object

    Returns
    -------
    strategy_steps : list
        List of strategy parameter dicts (one per step)
    """
    strategy_steps = []
    always_do_slowgain = True

    if field.parset['strategy'] == 'fullfieldselfcal':
        # Selfcal without peeling of non-imaged sources:
        #     - calibration on all sources
        #     - imaging of sectors
        #     - regrouping of resulting sky model to meet flux criteria
        #     - calibration on regrouped sources (calibration groups may differ from sectors)
        max_selfcal_loops = field.parset['calibration_specific']['max_selfcal_loops']
        for i in range(max_selfcal_loops):
            strategy_steps.append({})

            strategy_steps[i]['do_calibrate'] = True
            if field.input_h5parm is not None and i == 0:
                strategy_steps[i]['do_calibrate'] = False
            if i < 3 and not always_do_slowgain:
                strategy_steps[i]['do_slowgain_solve'] = False
            else:
                strategy_steps[i]['do_slowgain_solve'] = True

            strategy_steps[i]['peel_outliers'] = False

            strategy_steps[i]['do_image'] = True
            if i == 0:
                strategy_steps[i]['auto_mask'] = 3.6
                strategy_steps[i]['threshisl'] = 5.0
                strategy_steps[i]['threshpix'] = 7.5
            elif i == 1:
                strategy_steps[i]['auto_mask'] = 3.3
                strategy_steps[i]['threshisl'] = 5.0
                strategy_steps[i]['threshpix'] = 6.0
            else:
                strategy_steps[i]['auto_mask'] = 3.0
                strategy_steps[i]['threshisl'] = 4.0
                strategy_steps[i]['threshpix'] = 5.0

            if i == max_selfcal_loops - 1:
                strategy_steps[i]['do_update'] = False
            else:
                strategy_steps[i]['do_update'] = True
            strategy_steps[i]['regroup_model'] = True
            strategy_steps[i]['imaged_sources_only'] = False
            strategy_steps[i]['target_flux'] = None

            if i < 1 or i == max_selfcal_loops - 1:
                strategy_steps[i]['do_check'] = False
            else:
                strategy_steps[i]['do_check'] = True

    elif field.parset['strategy'] == 'sectorselfcal':
        # Selfcal with peeling of non-imaged sources (intended to be run on separated
        # sectors):
        #     - calibration on all sources
        #     - peeling of non-sector sources
        #     - imaging of sectors
        #     - no regrouping of resulting sky models
        #     - calibration on sector sources only (calibration groups are defined by
        #       sectors, one per sector)
        # TODO: allow regrouping on sector-by-sector
        #       basis so that large sectors can have multiple groups
        max_selfcal_loops = field.parset['calibration_specific']['max_selfcal_loops']
        for i in range(max_selfcal_loops):
            strategy_steps.append({})

            strategy_steps[i]['do_calibrate'] = True
            if field.input_h5parm is not None and i == 0:
                strategy_steps[i]['do_calibrate'] = False
            if i < 3 and not always_do_slowgain:
                strategy_steps[i]['do_slowgain_solve'] = False
            else:
                strategy_steps[i]['do_slowgain_solve'] = True

            if i < 1:
                strategy_steps[i]['peel_outliers'] = True
            else:
                strategy_steps[i]['peel_outliers'] = False

            strategy_steps[i]['do_image'] = True
            if i == 0:
                strategy_steps[i]['auto_mask'] = 3.6
                strategy_steps[i]['threshisl'] = 5.0
                strategy_steps[i]['threshpix'] = 7.5
            elif i == 1:
                strategy_steps[i]['auto_mask'] = 3.3
                strategy_steps[i]['threshisl'] = 5.0
                strategy_steps[i]['threshpix'] = 6.0
            else:
                strategy_steps[i]['auto_mask'] = 3.0
                strategy_steps[i]['threshisl'] = 4.0
                strategy_steps[i]['threshpix'] = 5.0

            if i == max_selfcal_loops - 1:
                strategy_steps[i]['do_update'] = False
            else:
                strategy_steps[i]['do_update'] = True
            strategy_steps[i]['regroup_model'] = True
            strategy_steps[i]['imaged_sources_only'] = True
            strategy_steps[i]['target_flux'] = None

            if i < 1 or i == max_selfcal_loops - 1:
                strategy_steps[i]['do_check'] = False
            else:
                strategy_steps[i]['do_check'] = True

    elif field.parset['strategy'] == 'image':
        # Image one or more sectors:
        #     - no calibration
        strategy_steps.append({})

        strategy_steps[0]['do_calibrate'] = False

        strategy_steps[0]['peel_outliers'] = False

        strategy_steps[0]['do_image'] = True
        strategy_steps[0]['auto_mask'] = 3.0
        strategy_steps[0]['threshisl'] = 4.0
        strategy_steps[0]['threshpix'] = 5.0

        strategy_steps[0]['do_update'] = False

        strategy_steps[0]['do_check'] = False

    elif os.path.exists(field.parset['strategy']):
        # Load user-defined strategy
        try:
            strategy_steps = runpy.run_path(field.parset['strategy'],
                                            init_globals={'field': field})['strategy_steps']
        except KeyError:
            log.error('Strategy "{}" does not define strategy_steps. '
                      'Exiting...'.format(field.parset['strategy']))
            sys.exit(1)

    else:
        log.error('Strategy "{}" not understood. Exiting...'.format(field.parset['strategy']))
        sys.exit(1)

    log.info('Using "{}" processing strategy'.format(field.parset['strategy']))
    return strategy_steps

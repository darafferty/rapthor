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

    if field.parset['strategy'] == 'selfcal':
        # Standard selfcal:
        #     - calibration on all sources
        #     - peeling of non-sector sources
        #     - peeling of bright sources (after 2 cycles)
        #     - imaging of sectors
        #     - regrouping of resulting sky model to meet flux criteria
        #     - calibration on regrouped sources (calibration groups may differ from sectors)
        min_selfcal_loops = 2
        max_selfcal_loops = 5
        for i in range(max_selfcal_loops):
            strategy_steps.append({})

            strategy_steps[i]['do_calibrate'] = True
            if i == 0:
                strategy_steps[i]['do_slowgain_solve'] = False
            else:
                strategy_steps[i]['do_slowgain_solve'] = True

            if i == 0:
                strategy_steps[i]['peel_outliers'] = True
                strategy_steps[i]['peel_bright_sources'] = False
            else:
                strategy_steps[i]['peel_outliers'] = False
                strategy_steps[i]['peel_bright_sources'] = True

            strategy_steps[i]['do_image'] = True
            if i == 0:
                strategy_steps[i]['auto_mask'] = 5.0
                strategy_steps[i]['threshisl'] = 4.0
                strategy_steps[i]['threshpix'] = 5.0
            elif i == 1:
                strategy_steps[i]['auto_mask'] = 4.0
                strategy_steps[i]['threshisl'] = 3.0
                strategy_steps[i]['threshpix'] = 5.0
            else:
                strategy_steps[i]['auto_mask'] = 3.0
                strategy_steps[i]['threshisl'] = 3.0
                strategy_steps[i]['threshpix'] = 5.0

            if i == 0:
                strategy_steps[i]['target_flux'] = 1.0
                strategy_steps[i]['max_nmiter'] = 10
            elif i == 1:
                strategy_steps[i]['target_flux'] = 0.8
                strategy_steps[i]['max_nmiter'] = 12
            elif i == 2:
                strategy_steps[i]['target_flux'] = 0.6
                strategy_steps[i]['max_nmiter'] = 14
            elif i == 3:
                strategy_steps[i]['target_flux'] = 0.5
                strategy_steps[i]['max_nmiter'] = 16
            else:
                strategy_steps[i]['target_flux'] = 0.45
                strategy_steps[i]['max_nmiter'] = 16
            strategy_steps[i]['regroup_model'] = True

            if i < min_selfcal_loops or i == max_selfcal_loops - 1:
                strategy_steps[i]['do_check'] = False
            else:
                strategy_steps[i]['do_check'] = True
                strategy_steps[i]['convergence_ratio'] = 0.95
                strategy_steps[i]['divergence_ratio'] = 1.1

    elif field.parset['strategy'] == 'image':
        # Image one or more sectors:
        #     - no calibration
        strategy_steps.append({})

        strategy_steps[0]['do_calibrate'] = False

        strategy_steps[0]['peel_outliers'] = False

        strategy_steps[0]['do_image'] = True
        strategy_steps[0]['auto_mask'] = 5.0
        strategy_steps[0]['threshisl'] = 4.0
        strategy_steps[0]['threshpix'] = 5.0
        strategy_steps[0]['max_nmiter'] = 12

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

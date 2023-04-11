"""
Module that holds all strategy-related functions
"""
import os
import logging
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
        #     - imaging of sectors
        #     - regrouping of resulting sky model to meet flux criteria
        #     - calibration on regrouped sources (calibration groups may span multiple sectors)
        min_selfcal_loops = 4
        max_selfcal_loops = 8
        for i in range(max_selfcal_loops):
            strategy_steps.append({})

            strategy_steps[i]['do_calibrate'] = True
            if i == 0:
                strategy_steps[i]['do_slowgain_solve'] = False
                strategy_steps[i]['peel_outliers'] = True
            elif i == 1:
                strategy_steps[i]['do_slowgain_solve'] = False
                strategy_steps[i]['peel_outliers'] = False
            else:
                strategy_steps[i]['do_slowgain_solve'] = True
                strategy_steps[i]['peel_outliers'] = False
            if i <= 2:
                strategy_steps[i]['solve_min_uv_lambda'] = 2000
            else:
                strategy_steps[i]['solve_min_uv_lambda'] = 350
            strategy_steps[i]['peel_bright_sources'] = False

            strategy_steps[i]['do_image'] = True
            if i < 2:
                strategy_steps[i]['auto_mask'] = 5.0
                strategy_steps[i]['threshisl'] = 4.0
                strategy_steps[i]['threshpix'] = 5.0
            elif i == 2:
                strategy_steps[i]['auto_mask'] = 4.0
                strategy_steps[i]['threshisl'] = 3.0
                strategy_steps[i]['threshpix'] = 5.0
            else:
                strategy_steps[i]['auto_mask'] = 3.0
                strategy_steps[i]['threshisl'] = 3.0
                strategy_steps[i]['threshpix'] = 5.0

            if i == 0:
                strategy_steps[i]['target_flux'] = 0.6
                strategy_steps[i]['max_nmiter'] = 8
                strategy_steps[i]['max_directions'] = 20
            elif i == 1:
                strategy_steps[i]['target_flux'] = 0.4
                strategy_steps[i]['max_nmiter'] = 9
                strategy_steps[i]['max_directions'] = 30
            elif i == 2:
                strategy_steps[i]['target_flux'] = 0.3
                strategy_steps[i]['max_nmiter'] = 10
                strategy_steps[i]['max_directions'] = 40
            else:
                strategy_steps[i]['target_flux'] = 0.25
                strategy_steps[i]['max_nmiter'] = 12
                strategy_steps[i]['max_directions'] = 50
            strategy_steps[i]['regroup_model'] = True

            if i < min_selfcal_loops - 1 or i == max_selfcal_loops - 1:
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
            raise ValueError('Strategy "{}" does not define '
                             'strategy_steps.'.format(field.parset['strategy']))

    else:
        raise ValueError('Strategy "{}" not understood.'.format(field.parset['strategy']))

    log.info('Using "{}" processing strategy'.format(field.parset['strategy']))

    # Check for required parameters. If any are missing, either print a warning if the
    # parameter has a default defined or raise an error if not
    primary_parameters = ['do_calibrate', 'do_image', 'do_check']
    secondary_parameters = {'do_calibrate': ['do_slowgain_solve', 'target_flux',
                                             'max_directions', 'regroup_model',
                                             'max_normalization_delta', 'solve_min_uv_lambda',
                                             'scale_normalization_delta'],
                            'do_image': ['auto_mask', 'threshisl', 'threshpix', 'max_nmiter',
                                         'peel_outliers', 'peel_bright_sources'],
                            'do_check': ['convergence_ratio', 'divergence_ratio']}
    for primary in primary_parameters:
        for i in range(len(strategy_steps)):
            if primary not in strategy_steps[i]:
                raise ValueError('Required parameter "{0}" not defined in the '
                                 'strategy for cycle {1}.'.format(primary, i+1))
            if strategy_steps[i][primary]:
                for secondary in secondary_parameters[primary]:
                    if secondary not in strategy_steps[i]:
                        if hasattr(field, secondary):
                            log.warn('Parameter "{0}" not defined in the strategy for '
                                     'cycle {1}. Using the default value of '
                                     '{2}'.format(secondary, i+1, getattr(field, secondary)))
                        else:
                            raise ValueError('Required parameter "{0}" not defined in the '
                                             'strategy for cycle {1}.'.format(secondary, i+1))
    return strategy_steps

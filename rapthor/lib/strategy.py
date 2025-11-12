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

    In general, the processing strategy must define a strategy_steps list as follows:
        - strategy_steps is a list of dicts, with one entry per processing cycle
        - each dict defines the values of the strategy parameters for its cycle
        - if only a single cycle is defined, no selfcal will be done, only the final
          cycle
        - if multiple cycles are defined, strategy_steps must be constructed so that
          self calibration is done first, followed by a final cycle:
          - if selfcal is to be done, the dicts for selfcal must be the first
            entries
          - the dict for the final pass must be the last entry

    Parameters
    ----------
    field : Field object
        Field object

    Returns
    -------
    strategy_steps : list
        List of strategy parameter dicts (one per processing cycle)

    Raises
    ------
    ValueError
        If the strategy could not be set successfully
    """
    if field.parset['strategy'] == 'selfcal':
        # Standard selfcal
        strategy_steps = set_selfcal_strategy(field)
    elif field.parset['strategy'] == 'image':
        # Standard imaging
        strategy_steps = set_image_strategy(field)
    elif os.path.exists(field.parset['strategy']):
        # User-defined
        strategy_steps = set_user_strategy(field)
    else:
        raise ValueError('Strategy "{}" not understood.'.format(field.parset['strategy']))

    log.info('Using "{}" processing strategy'.format(field.parset['strategy']))

    # Check the strategy for the presence of deprecated and/or missing
    # parameters
    strategy_steps = check_and_adjust_parameters(field, strategy_steps)

    return strategy_steps


def set_selfcal_strategy(field):
    """
    Sets up the standard selfcal strategy

    The standard selfcal strategy is tailored to a full-field reduction and includes:
        - calibration on all sources
        - peeling of the non-sector sources
        - imaging of the sectors
        - regrouping of the resulting sky model to meet the target flux criterion
        - calibration on the regrouped patches (calibration patches may span multiple
          sectors)
        - looping of the processing until convergence is obtained, after which a final
          cycle may be done (depending on the values of various parset parameters)

    The parameters of the final cycle (if done) are set to those of the last
    cycle of selfcal.

    Parameters
    ----------
    field : Field object
        Field object

    Returns
    -------
    strategy_steps : list of dicts
        List of strategy parameter dicts, with selfcal cycles first and the final cycle as
        the last entry
    """
    # Set flag that determines if the phase-only solve cycles will be done
    # Note: currently, this flag depends only on whether an initial sky model was
    # generated from the input data or not, as other starting models (such as those from
    # the TGSS or LoTSS) have been found to be too poor to start amplitude calibration
    # immediately. Other criteria can be added in future if deemed useful
    do_phase_only_solves = not field.parset['generate_initial_skymodel']

    strategy_steps = []

    min_selfcal_loops = 4 if do_phase_only_solves else 2
    max_selfcal_loops = 8 if do_phase_only_solves else 6
    for i in range(max_selfcal_loops):
        strategy_steps.append({})

        strategy_steps[i]['do_calibrate'] = True
        if i == 0:
            strategy_steps[i]['do_slowgain_solve'] = not do_phase_only_solves
            strategy_steps[i]['peel_outliers'] = True
        elif i == 1:
            strategy_steps[i]['do_slowgain_solve'] = not do_phase_only_solves
            strategy_steps[i]['peel_outliers'] = False
        else:
            strategy_steps[i]['do_slowgain_solve'] = True
            strategy_steps[i]['peel_outliers'] = False
        if i == 2 and field.antenna == 'HBA' and do_phase_only_solves:
            strategy_steps[i]['solve_min_uv_lambda'] = 2000
        else:
            strategy_steps[i]['solve_min_uv_lambda'] = 750
        strategy_steps[i]['do_fulljones_solve'] = False
        strategy_steps[i]['peel_bright_sources'] = False
        strategy_steps[i]['max_normalization_delta'] = 0.3
        strategy_steps[i]['scale_normalization_delta'] = True
        strategy_steps[i]['fast_timestep_sec'] = 32.0
        strategy_steps[i]['medium_timestep_sec'] = 120.0
        strategy_steps[i]['slow_timestep_sec'] = 600.0

        if i == 0:
            strategy_steps[i]['do_normalize'] = True
        else:
            strategy_steps[i]['do_normalize'] = False

        strategy_steps[i]['do_image'] = True
        if i < 2 and do_phase_only_solves:
            strategy_steps[i]['auto_mask'] = 5.0
            strategy_steps[i]['threshisl'] = 4.0
            strategy_steps[i]['threshpix'] = 5.0
            strategy_steps[i]['max_nmiter'] = 8
        elif (i == 2 and do_phase_only_solves) or (i == 0 and not do_phase_only_solves):
            strategy_steps[i]['auto_mask'] = 4.0
            strategy_steps[i]['threshisl'] = 3.0
            strategy_steps[i]['threshpix'] = 5.0
            strategy_steps[i]['max_nmiter'] = 10
        else:
            strategy_steps[i]['auto_mask'] = 3.0
            strategy_steps[i]['threshisl'] = 3.0
            strategy_steps[i]['threshpix'] = 5.0
            strategy_steps[i]['max_nmiter'] = 12
        strategy_steps[i]['auto_mask_nmiter'] = 2
        strategy_steps[i]['channel_width_hz'] = 8e6

        if i == 0:
            if do_phase_only_solves:
                strategy_steps[i]['target_flux'] = 0.6
                strategy_steps[i]['max_directions'] = 20
                strategy_steps[i]['max_distance'] = 3.0
            else:
                strategy_steps[i]['target_flux'] = 0.3
                strategy_steps[i]['max_directions'] = 40
                strategy_steps[i]['max_distance'] = 3.0
        elif i == 1:
            if do_phase_only_solves:
                strategy_steps[i]['target_flux'] = 0.4
                strategy_steps[i]['max_directions'] = 30
                strategy_steps[i]['max_distance'] = 3.0
            else:
                strategy_steps[i]['target_flux'] = 0.25
                strategy_steps[i]['max_directions'] = 40
                strategy_steps[i]['max_distance'] = 3.5
        elif i == 2 and do_phase_only_solves:
            strategy_steps[i]['target_flux'] = 0.3
            strategy_steps[i]['max_directions'] = 40
            strategy_steps[i]['max_distance'] = 3.5
        else:
            strategy_steps[i]['target_flux'] = 0.25
            strategy_steps[i]['max_directions'] = 50
            strategy_steps[i]['max_distance'] = 4.0
        strategy_steps[i]['regroup_model'] = True

        if i < min_selfcal_loops - 1:
            strategy_steps[i]['do_check'] = False
        else:
            strategy_steps[i]['do_check'] = True
            strategy_steps[i]['convergence_ratio'] = 0.95
            strategy_steps[i]['divergence_ratio'] = 1.1
            strategy_steps[i]['failure_ratio'] = 10.0

    # Set a final step as a duplicate of the last selfcal one, except that
    # the target output imaging channel width is smaller
    if strategy_steps:
        strategy_steps.append(strategy_steps[-1])
        strategy_steps[-1]['channel_width_hz'] = 4e6

    return strategy_steps


def set_image_strategy(field):
    """
    Sets up the standard imaging strategy

    The standard imaging strategy is a single cycle that includes:
        - no selfcal cycles
        - no calibration
        - no peeling
        - imaging of the sectors

    Returns
    -------
    strategy_steps : list of dicts
        List of strategy parameter dicts. A single cycle only is defined (i.e., no
        selfcal cycles)
    """
    strategy_steps = [{}]

    strategy_steps[0]['do_calibrate'] = False
    strategy_steps[0]['do_normalize'] = False
    strategy_steps[0]['peel_outliers'] = True
    strategy_steps[0]['peel_bright_sources'] = False
    strategy_steps[0]['do_image'] = True
    strategy_steps[0]['auto_mask'] = 3.0
    strategy_steps[0]['auto_mask_nmiter'] = 2
    strategy_steps[0]['threshisl'] = 3.0
    strategy_steps[0]['threshpix'] = 5.0
    strategy_steps[0]['max_nmiter'] = 12
    strategy_steps[0]['do_check'] = False
    strategy_steps[0]['regroup_model'] = False

    return strategy_steps


def set_user_strategy(field):
    """
    Sets up a user-defined strategy

    The user-defined strategy is read from a Python file, the contents of which
    must define the strategy_steps list as follows:
        - strategy_steps is a list of dicts, with one entry per processing cycle
        - if selfcal is to be done, the dicts for selfcal must be the first
          entries
        - the dict for the final cycle must be the last entry

    Parameters
    ----------
    field : Field object
        Field object

    Returns
    -------
    strategy_steps : list
        List of strategy parameter dicts

    Raises
    ------
    ValueError
        If the strategy file did not define strategy_steps
    """
    try:
        strategy_steps = runpy.run_path(field.parset['strategy'],
                                        init_globals={'field': field})['strategy_steps']
    except KeyError:
        raise ValueError('Strategy "{}" does not define '
                         'strategy_steps.'.format(field.parset['strategy']))

    return strategy_steps


def check_and_adjust_parameters(field, strategy_steps):
    """
    Checks the strategy for deprecated or missing parameters

    A deprecated parameter is replaced with its corresponding new parameter
    when possible; if there is no replacement, it is removed entirely.

    A missing required parameter is set to its default when possible; if there
    is no default defined, a ValueError error is raised.

    Parameters
    ----------
    field : Field object
        Field object
    strategy_steps : list of dicts
        List of strategy steps to check

    Returns
    -------
    strategy_steps : list of dicts
        Adjusted version of the input strategy_steps

    Raises
    ------
    ValueError
        If a required parameter is not defined and no suitable default
        is available for it
    """
    # Define the deprecated parameters and their replacements (if any)
    deprecated_parameters = {'slow_timestep_joint_sec': None,
                             'slow_timestep_separate_sec': 'slow_timestep_sec'}

    # Define the required parameters for each of the main strategy parts
    required_parameters = {'do_calibrate': ['do_slowgain_solve', 'do_fulljones_solve',
                                            'target_flux', 'max_directions', 'regroup_model',
                                            'max_normalization_delta', 'solve_min_uv_lambda',
                                            'fast_timestep_sec', 'slow_timestep_sec',
                                            'scale_normalization_delta', 'max_directions'],
                           'do_normalize': [],
                           'do_image': ['auto_mask', 'auto_mask_nmiter', 'threshisl',
                                        'threshpix', 'max_nmiter', 'peel_outliers',
                                        'peel_bright_sources', 'channel_width_hz'],
                           'do_check': ['convergence_ratio', 'divergence_ratio',
                                        'failure_ratio']}

    # Check for deprectaed parameters, updating the steps to use the new
    # names when defined
    for deprecated, replacement in deprecated_parameters.items():
        for i in range(len(strategy_steps)):
            if deprecated in strategy_steps[i]:
                if replacement is not None:
                    log.warn(f'Parameter "{deprecated}" is defined in the strategy for '
                             f'cycle {i+1} but is deprecated. Please use "{replacement}" '
                             'instead.')
                    strategy_steps[i][replacement] = strategy_steps[i][deprecated]
                else:
                    log.warn(f'Parameter "{deprecated}" is defined in the strategy for '
                             f'cycle {i+1} but is no longer used.')
                strategy_steps[i].pop(deprecated)

    # Check for required parameters. If any are missing, either print a warning if the
    # parameter has a default defined or raise an error if not
    for primary, secondary_parameters in required_parameters.items():
        for i in range(len(strategy_steps)):
            if primary not in strategy_steps[i]:
                raise ValueError(f'Required parameter "{primary}" is not defined in the '
                                 f'strategy for cycle {i+1}.')
            if strategy_steps[i][primary]:
                for secondary in secondary_parameters:
                    if secondary not in strategy_steps[i]:
                        if hasattr(field, secondary):
                            log.warn(f'Parameter "{secondary}" is not defined in the '
                                     f'strategy for cycle {i+1}. Using the default value '
                                     f'of {getattr(field, secondary)}.')
                        else:
                            raise ValueError(f'Required parameter "{secondary}" is not '
                                             f'defined in the strategy for cycle {i+1}.')

    return strategy_steps

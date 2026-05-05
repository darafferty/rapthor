"""
Script that defines a custom user processing strategy.

This script demonstrates a simple alternative processing strategy. When this
file is specified as the strategy in the parset file, Rapthor will perform
self-calibration differently from the default approach (see below for details).

If you need to tweak the default calibration strategy slightly, it might be
easier to start from `default_calibration_strategy.py`.
"""
strategy_steps = []
max_selfcal_loops = 8

# Set a default calibration strategy, which is overridden by the cycle-specific
# strategies defined in the loop below.  When setting `calibration_strategy`
# in the strategy file, the legacy parameters `do_fulljones_solve` and 
# `do_slowgain_solve` are ignored. The order of the solves specified is respected.
default_calibration_strategy = {
    "di": { # No direction-independent calibration by default
        "fast_phase": False,
        "medium_phase": False,
        "slow_gain": False,
        "full_jones": False,
    },
    "dd": { # Do fast and medium phase and slow gain DD calibration by default
        "fast_phase": True,
        "medium_phase": True,
        "slow_gain": True,
        "full_jones": False
    }
}
for i in range(max_selfcal_loops):
    strategy_steps.append({})

    # Here we adjust the calibration strategy to specify a different order 
    # of solves and add direction independent calibration.
    strategy_steps[i]['do_calibrate'] = True
    if i == 0: # Run fast phase only DI calibration in the first cycle
        strategy_steps[i]['calibration_strategy'] = {
            "di": {
                "fast_phase": True,
                "medium_phase": False,
                "slow_gain": False,
                "full_jones": False,
            },
            "dd": {
                "fast_phase": False,
                "medium_phase": False,
                "slow_gain": False,
                "full_jones": False
            }
        }
    elif i == 1: # Run full jones DI calibration in the second cycle
        strategy_steps[i]['calibration_strategy'] = {
            "di": {
                "fast_phase": False,
                "medium_phase": False,
                "slow_gain": False,
                "full_jones": True,
            },
            "dd": {
                "fast_phase": False,
                "medium_phase": False,
                "slow_gain": False,
                "full_jones": False
            }
        }
    elif i == 2: # Run fast and medium phase only DD calibration in the third cycle
        strategy_steps[i]['calibration_strategy'] = {
            "di": {
                "fast_phase": False,
                "medium_phase": False,
                "slow_gain": False,
                "full_jones": False,
            },
            "dd": {
                "fast_phase": True,
                "medium_phase": True,
                "slow_gain": False,
                "full_jones": False
             }
        }
    else: # Run the direction-dependent default strategy for the later cycles (includes slow gain)
        strategy_steps[i]['calibration_strategy'] = default_calibration_strategy

    if i == 0:
        strategy_steps[i]['peel_outliers'] = True
    else:
        strategy_steps[i]['peel_outliers'] = False
    strategy_steps[i]['peel_bright_sources'] = False
    strategy_steps[i]['max_normalization_delta'] = 0.3
    strategy_steps[i]['scale_normalization_delta'] = True
    strategy_steps[i]['solve_min_uv_lambda'] = 150
    strategy_steps[i]['fast_timestep_sec'] = 32.0
    strategy_steps[i]['medium_timestep_sec'] = 120.0
    strategy_steps[i]['slow_timestep_sec'] = 600.0
    strategy_steps[i]['fulljones_timestep_sec'] = 600.0

    # Here we set adjust the imaging strategy from the default, using fixed
    # thresholds for source finding and higher thresholds for calibrator
    # selection. This approach could be useful if higher SNRs are needed during
    # calibrations
    strategy_steps[i]['do_normalize'] = True
    strategy_steps[i]['do_image'] = True
    strategy_steps[i]['auto_mask'] = 3.0
    strategy_steps[i]['auto_mask_nmiter'] = 2
    strategy_steps[i]['channel_width_hz'] = 4e6
    strategy_steps[i]['threshisl'] = 4.0
    strategy_steps[i]['threshpix'] = 5.0
    if i < 1:
        strategy_steps[i]['target_flux'] = 1.5
        strategy_steps[i]['max_nmiter'] = 6
    elif i < 5:
        strategy_steps[i]['target_flux'] = 1.0
        strategy_steps[i]['max_nmiter'] = 8
    elif i < 7:
        strategy_steps[i]['target_flux'] = 0.75
        strategy_steps[i]['max_nmiter'] = 10
    else:
        strategy_steps[i]['target_flux'] = 0.5
        strategy_steps[i]['max_nmiter'] = 12

    strategy_steps[i]['regroup_model'] = True
    strategy_steps[i]['max_directions'] = 30
    strategy_steps[i]['max_distance'] = 3.0

    # Here we use the same settings as the default strategy
    if i == 0 or i == max_selfcal_loops - 1:
        strategy_steps[i]['do_check'] = False
    else:
        strategy_steps[i]['do_check'] = True
        strategy_steps[i]['convergence_ratio'] = 0.95
        strategy_steps[i]['divergence_ratio'] = 1.1
        strategy_steps[i]['failure_ratio'] = 10.0

# Set the parameters for the final pass as duplicates of the last selfcal step
strategy_steps.append(strategy_steps[-1])

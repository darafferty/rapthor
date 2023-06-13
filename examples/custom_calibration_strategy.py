"""
Script that defines a custom user processing strategy.

This script demonstrates a simple alternative calibration strategy. When this
file is specified as calibration strategy in the parset file, Rapthor will
perform calibration different to the default approach, e.g. it will perform a
slowgain solve already in the first iteration.

If you need to tweak the default calibration strategy slightly, it might be
easier to start from `default_calibration_strategy.py`.
"""
strategy_steps = []
max_selfcal_loops = 8
for i in range(max_selfcal_loops):
    strategy_steps.append({})

    # Here we adjust the calibration strategy from the default so that slow-gain
    # calibration is done from the start and the minimum uv distance used in the
    # all solves is set to 350 lambda. This approach could be useful if the
    # starting sky model is of high quality.
    strategy_steps[i]['do_calibrate'] = True
    strategy_steps[i]['do_slowgain_solve'] = True
    if i == 0:
        strategy_steps[i]['peel_outliers'] = True
    else:
        strategy_steps[i]['peel_outliers'] = False
    strategy_steps[i]['peel_bright_sources'] = False
    strategy_steps[i]['max_normalization_delta'] = 0.3
    strategy_steps[i]['scale_normalization_delta'] = True
    strategy_steps[i]['solve_min_uv_lambda'] = 350

    # Here we set adjust the imaging strategy from the default, using fixed
    # thresholds for source finding and higher thresholds for calibrator
    # selection. This approach could be useful if higher SNRs are needed during
    # calibrations
    strategy_steps[i]['do_image'] = True
    strategy_steps[i]['auto_mask'] = 3.0
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

    # Here we use the same settings as the default strategy
    if i == 0 or i == max_selfcal_loops - 1:
        strategy_steps[i]['do_check'] = False
    else:
        strategy_steps[i]['do_check'] = True
        strategy_steps[i]['convergence_ratio'] = 0.95
        strategy_steps[i]['divergence_ratio'] = 1.1

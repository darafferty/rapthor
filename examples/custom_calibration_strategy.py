"""
Script that defines a custom user processing strategy.

This script demonstrates a simple alternative calibration
strategy. When this file is specified as calibration strategy
in the parset file, Rapthor will perform calibration different
to the default approach, e.g. it will perform a slowgain solve
already in the first iteration.

If you need to tweak the default calibration startegy slightly,
it might be easier to start from `default_calibration_strategy.py`.
"""
strategy_steps = []
max_selfcal_loops = 8
for i in range(max_selfcal_loops):
    strategy_steps.append({})

    strategy_steps[i]['do_calibrate'] = True
    strategy_steps[i]['do_slowgain_solve'] = True
    strategy_steps[i]['max_normalization_delta'] = 0.0

    if i == 0:
        strategy_steps[i]['peel_outliers'] = True
    else:
        strategy_steps[i]['peel_outliers'] = False
    strategy_steps[i]['peel_bright_sources'] = True

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

    if i == 0 or i == max_selfcal_loops - 1:
        strategy_steps[i]['do_check'] = False
    else:
        strategy_steps[i]['do_check'] = True

"""
Script that defines the default user processing strategy. Specifying
this file as the calibration strategy in the Rapthor parset causes
default Rapthor to use the default calibration behaviour, which is
equal to specifying no specific calibration strategy.

This file is provided to base custom strategies from.
"""
strategy_steps = []
min_selfcal_loops = 4
max_selfcal_loops = 8
for i in range(max_selfcal_loops):
    strategy_steps.append({})

    strategy_steps[i]['do_calibrate'] = True
    if i == 0:
        strategy_steps[i]['do_slowgain_solve'] = False
        strategy_steps[i]['peel_outliers'] = True
        strategy_steps[i]['peel_bright_sources'] = False
    elif i == 1:
        strategy_steps[i]['do_slowgain_solve'] = False
        strategy_steps[i]['peel_outliers'] = False
        strategy_steps[i]['peel_bright_sources'] = False
    else:
        strategy_steps[i]['do_slowgain_solve'] = True
        strategy_steps[i]['peel_outliers'] = False
        strategy_steps[i]['peel_bright_sources'] = True

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
        strategy_steps[i]['target_flux'] = 1.5
        strategy_steps[i]['max_nmiter'] = 8
    elif i == 1:
        strategy_steps[i]['target_flux'] = 1.0
        strategy_steps[i]['max_nmiter'] = 9
    elif i == 2:
        strategy_steps[i]['target_flux'] = 0.8
        strategy_steps[i]['max_nmiter'] = 10
    elif i == 3:
        strategy_steps[i]['target_flux'] = 0.6
        strategy_steps[i]['max_nmiter'] = 11
    else:
        strategy_steps[i]['target_flux'] = 0.5
        strategy_steps[i]['max_nmiter'] = 12
    strategy_steps[i]['regroup_model'] = True

    if i < min_selfcal_loops - 1 or i == max_selfcal_loops - 1:
        strategy_steps[i]['do_check'] = False
    else:
        strategy_steps[i]['do_check'] = True
        strategy_steps[i]['convergence_ratio'] = 0.95
        strategy_steps[i]['divergence_ratio'] = 1.1

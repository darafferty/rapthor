"""
Script that defines a custom user processing strategy
"""
strategy_steps = []
max_selfcal_loops = 8
for i in range(max_selfcal_loops):
    strategy_steps.append({})

    strategy_steps[i]['do_calibrate'] = True
    strategy_steps[i]['do_slowgain_solve'] = True

    if i < 1:
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
    elif i < 5:
        strategy_steps[i]['target_flux'] = 1.0
    elif i < 7:
        strategy_steps[i]['target_flux'] = 0.75
    else:
        strategy_steps[i]['target_flux'] = 0.5
    strategy_steps[i]['regroup_model'] = True
    strategy_steps[i]['imaged_sources_only'] = True

    if i < 1 or i == max_selfcal_loops - 1:
        strategy_steps[i]['do_check'] = False
    else:
        strategy_steps[i]['do_check'] = True

strategy_steps = []
max_selfcal_loops = 2
for i in range(max_selfcal_loops):
    strategy_steps.append({})
 
    strategy_steps[i]['do_calibrate'] = True
    strategy_steps[i]['do_slowgain_solve'] = False
    strategy_steps[i]['do_fulljones_solve'] = False
 
    strategy_steps[i]['peel_outliers'] = False
    strategy_steps[i]['peel_bright_sources'] = False
 
    strategy_steps[i]['fast_timestep_sec'] = 10.19349
    strategy_steps[i]['slow_timestep_separate_sec'] = 1800
 
 
    strategy_steps[i]['do_normalize'] = False
    strategy_steps[i]['do_image'] = True
    strategy_steps[i]['auto_mask'] = 5.0
    strategy_steps[i]['auto_mask_nmiter'] = 2
    strategy_steps[i]['threshisl'] = 4.0
    strategy_steps[i]['threshpix'] = 5.0
 
    if i == 0:
        strategy_steps[i]['target_flux'] = 0.6
        strategy_steps[i]['max_directions'] = 20
    else:
        strategy_steps[i]['target_flux'] = 0.4
        strategy_steps[i]['max_directions'] = 30
 
    strategy_steps[i]['max_nmiter'] = 6666666
    strategy_steps[i]['regroup_model'] = True
     
    strategy_steps[i]['max_distance'] = 3.0
 
    strategy_steps[i]['do_check'] = False

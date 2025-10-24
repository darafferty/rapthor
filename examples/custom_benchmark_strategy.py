# Strategy – doing at maximum 4
# rounds/cycles, starting with DI phase-only, then DD phase-only, then DD
# complexgain (amplitude and phase)

strategy_steps = []

min_selfcal_loops = 2
max_selfcal_loops = 4

for i in range(max_selfcal_loops):
    # General settings
    strategy_steps.append(step := {})

    step['do_calibrate'] = True
    step['do_slowgain_solve'] = False
    step['do_fulljones_solve'] = False

    step['peel_outliers'] = False
    step['peel_bright_sources'] = False

    step['fast_timestep_sec'] = 60
    step['slow_timestep_separate_sec'] = 3600


    step['do_normalize'] = False
    step['do_image'] = True
    step['auto_mask'] = 5.0
    step['auto_mask_nmiter'] = 2
    step['threshisl'] = 4.0
    step['threshpix'] = 5.0
    step['max_nmiter'] = 50
    step['regroup_model'] = True
    step['max_distance'] = 3.0
    step['do_check'] = False
    
    # Cycle-specific settings
    # Do phase-only DI cycle
    if i < 2:
        step['auto_mask'] = 10.0
        step['threshisl'] = 10.0
        step['threshpix'] = 10.0
        step['target_flux'] = 1.0
        step['max_directions'] = 1
    # Do phase-only DD cycle
    elif i == 2:
        step['target_flux'] = 1.0
        step['max_directions'] = 5
    # Do complex gain DD cycle
    else:
        step['do_slowgain_solve'] = True
        step['target_flux'] = 0.5
        step['max_directions'] = 5
        step['auto_mask'] = 3.0
        step['auto_mask_nmiter'] = 5
        step['threshisl'] = 3.0
        step['threshpix'] = 3.0


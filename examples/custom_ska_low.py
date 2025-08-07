strategy_steps = []

min_selfcal_loops = 2
max_selfcal_loops = 4

for i in range(max_selfcal_loops):
    # General settings
    strategy_steps.append({})

    # General calibration
    strategy_steps[i]['do_calibrate'] = True
    strategy_steps[i]['do_fulljones_solve'] = False

    # Bright source removal
    strategy_steps[i]['peel_outliers'] = False
    strategy_steps[i]['peel_bright_sources'] = False

    # Solution intervals (maxmimum for slow gains)
    strategy_steps[i]['fast_timestep_sec'] = 10.19349
    strategy_steps[i]['slow_timestep_separate_sec'] = 1800

    # Don't bootstrap fluxes from LoTSS/TGSS
    strategy_steps[i]['do_normalize'] = False

    # Make images
    strategy_steps[i]['do_image'] = True

    # Arbitarily large max number of major cycles
    strategy_steps[i]['max_nmiter'] = 6666666

    # Allow changing of model source grouping to satisfy direction constraints (below)
    strategy_steps[i]['regroup_model'] = True

    # Max distance to DDE calibrators (default)
    strategy_steps[i]['max_distance'] = 3.0

    # Don't check for self-calibration convergence
    strategy_steps[i]['do_check'] = False
    
    # Cycle-specific settings
    # Do phase-only DI cycle
    if i == 0:
        strategy_steps[i]['do_slowgain_solve'] = False
        strategy_steps[i]['target_flux'] = 1.0
        strategy_steps[i]['max_directions'] = 1
        strategy_steps[i]['auto_mask'] = 5.0
        strategy_steps[i]['auto_mask_nmiter'] = 2
        strategy_steps[i]['threshisl'] = 4.0
        strategy_steps[i]['threshpix'] = 5.0
    # Do phase-only DD cycle
    elif i == 1:
        strategy_steps[i]['do_slowgain_solve'] = False
        strategy_steps[i]['target_flux'] = 0.5
        strategy_steps[i]['max_directions'] = 5
        strategy_steps[i]['auto_mask'] = 5.0
        strategy_steps[i]['auto_mask_nmiter'] = 2
        strategy_steps[i]['threshisl'] = 4.0
        strategy_steps[i]['threshpix'] = 5.0
    # Do complex gain DD cycle
    else:
        strategy_steps[i]['do_slowgain_solve'] = True
        strategy_steps[i]['target_flux'] = 0.5
        strategy_steps[i]['max_directions'] = 5
        strategy_steps[i]['auto_mask'] = 3.0
        strategy_steps[i]['auto_mask_nmiter'] = 5
        strategy_steps[i]['threshisl'] = 3.0
        strategy_steps[i]['threshpix'] = 3.0


"""Custom strategy for testing."""

strategy_steps = []

for i in range(3):
    strategy_steps.append({})
    # Primary parameters
    strategy_steps[i]['do_calibrate'] = True
    strategy_steps[i]['do_check'] = False
    strategy_steps[i]['do_image'] = True
    strategy_steps[i]['do_normalize'] = True

    # Secondary parameters
    # Calibration parameters
    strategy_steps[i]['do_slowgain_solve'] = True
    strategy_steps[i]['do_fulljones_solve'] = False
    strategy_steps[i]['target_flux'] = 3.0
    strategy_steps[i]['max_directions'] = 3.0
    strategy_steps[i]['regroup_model'] = False
    strategy_steps[i]['max_normalization_delta'] = 0.3
    strategy_steps[i]['scale_normalization_delta'] = True
    strategy_steps[i]['solve_min_uv_lambda'] = 150
    strategy_steps[i]['fast_timestep_sec'] = 32.0
    strategy_steps[i]['medium_timestep_sec'] = 120.0
    strategy_steps[i]['slow_timestep_sec'] = 600.0
    strategy_steps[i]['fulljones_timestep_sec'] = 600.0

    # Imaging parameters
    strategy_steps[i]['peel_outliers'] = False
    strategy_steps[i]['peel_bright_sources'] = False
    strategy_steps[i]['auto_mask'] = 3.0
    strategy_steps[i]['auto_mask_nmiter'] = 2
    strategy_steps[i]['channel_width_hz'] = 4e6
    strategy_steps[i]['threshisl'] = 3.0
    strategy_steps[i]['threshpix'] = 5.0
    strategy_steps[i]['max_nmiter'] = 12

"""Two-cycle DI/DD strategy used by the multiple-cycle integration test."""

COMMON_SETTINGS = {
    "channel_width_hz": 195312.5,
    "do_slowgain_solve": False,
    "do_fulljones_solve": False,
    "peel_outliers": False,
    "peel_bright_sources": False,
    "fast_timestep_sec": 32.0,
    "medium_timestep_sec": 120.0,
    "slow_timestep_sec": 600.0,
    "do_normalize": False,
    "auto_mask": 3.0,
    "auto_mask_nmiter": 2,
    "threshisl": 2.0,
    "threshpix": 3.0,
    "max_nmiter": 12,
    "regroup_model": True,
    "max_distance": None,
    "do_check": False,
    "target_flux": 0.1,
    "max_directions": 4,
    "max_wsclean_nchannels": 2,
}

strategy_steps = [
    {
        **COMMON_SETTINGS,
        "do_calibrate": True,
        "do_image": True,
        "calibration_strategy": {"di": ["full_jones"], "dd": []},
    },
    {
        **COMMON_SETTINGS,
        "do_calibrate": True,
        "do_image": True,
        "calibration_strategy": {"di": [], "dd": ["fast_phase"]},
    },
]

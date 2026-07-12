"""Current-branch DD slow-gain/default-like calibration-only scenario."""

COMMON_SETTINGS = {
    "do_calibrate": True,
    "do_image": False,
    "do_normalize": False,
    "do_check": False,
    "peel_outliers": False,
    "peel_bright_sources": False,
    "fast_timestep_sec": 20.0,
    "medium_timestep_sec": 40.0,
    "slow_timestep_sec": 80.0,
    "fulljones_timestep_sec": 80.0,
    "max_normalization_delta": 0.3,
    "scale_normalization_delta": True,
    "solve_min_uv_lambda": 80,
    "target_flux": 0.6,
    "max_directions": 5,
    "max_distance": None,
    "regroup_model": True,
    "auto_mask": 5.0,
    "auto_mask_nmiter": 1,
    "channel_width_hz": 48828.125,
    "threshisl": 3.0,
    "threshpix": 5.0,
    "max_nmiter": 6,
    "calibration_strategy": {
        "dd": ["fast_phase", "medium_phase", "slow_gains", "medium_phase"],
        "di": [],
    },
}


strategy_steps = [COMMON_SETTINGS]

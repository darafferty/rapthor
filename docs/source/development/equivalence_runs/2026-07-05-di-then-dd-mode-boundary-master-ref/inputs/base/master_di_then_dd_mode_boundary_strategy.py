"""Master-compatible DI-to-DD mode-boundary scenario.

The first selfcal cycle runs the legacy master order: DD fast+medium phase-only
calibration followed by DI full-Jones. The second selfcal cycle disables the DI
full-Jones solve and performs DD phase-only calibration only. A fixed facet
layout keeps DD directions compatible, so differences are focused on how the
previous DI product is treated when the later cycle returns to DD calibration.
"""

COMMON_SETTINGS = {
    "do_calibrate": True,
    "do_image": True,
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
    "do_slowgain_solve": False,
}


def _step(**overrides):
    step = dict(COMMON_SETTINGS)
    step.update(overrides)
    return step


strategy_steps = [
    _step(do_fulljones_solve=True),
    _step(do_fulljones_solve=False),
    _step(do_fulljones_solve=False),
]

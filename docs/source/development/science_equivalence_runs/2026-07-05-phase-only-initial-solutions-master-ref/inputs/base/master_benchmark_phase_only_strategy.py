"""Master-compatible DD fast+medium phase-only benchmark strategy.

This is the branch-equivalence control scenario for checking the current branch
against master without exercising the legacy slow-gain amplitude path.
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
    "auto_mask": 5.0,
    "auto_mask_nmiter": 1,
    "channel_width_hz": 48828.125,
    "threshisl": 3.0,
    "threshpix": 5.0,
    "max_nmiter": 6,
    "do_fulljones_solve": False,
    "do_slowgain_solve": False,
}


def _step(*, regroup_model):
    return {
        **COMMON_SETTINGS,
        "regroup_model": regroup_model,
    }


strategy_steps = [
    _step(regroup_model=False),
    _step(regroup_model=True),
    _step(regroup_model=True),
    _step(regroup_model=False),
]

"""Short multi-cycle strategy for the local Prefect demo parset."""

COMMON_SETTINGS = {
    "do_calibrate": True,
    "do_image": True,
    "do_normalize": False,
    "do_check": False,
    "peel_outliers": False,
    "peel_bright_sources": False,
    "fast_timestep_sec": 16.0,
    "medium_timestep_sec": 32.0,
    "slow_timestep_sec": 48.0,
    "fulljones_timestep_sec": 48.0,
    "max_normalization_delta": 0.3,
    "scale_normalization_delta": True,
    "solve_min_uv_lambda": 80,
    # The quick demo uses a tiny fixture MS, so later source lists can contain
    # only a very faint placeholder source. Avoid DD slow-gain calibration here;
    # use the generated rich demo for slow-gain amplitude calibration.
    "target_flux": None,
    "max_directions": 4,
    "max_distance": None,
    "regroup_model": True,
    "auto_mask": 5.0,
    "auto_mask_nmiter": 2,
    "channel_width_hz": 195312.5,
    "threshisl": 3.0,
    "threshpix": 5.0,
    "max_nmiter": 12,
}


def _step(calibration_strategy, **overrides):
    step = {
        **COMMON_SETTINGS,
        "calibration_strategy": calibration_strategy,
    }
    step.update(overrides)
    return step


strategy_steps = [
    _step(
        {"di": ["fast_phase"]},
        regroup_model=False,
    ),
    _step(
        {"di": ["full_jones"], "dd": []},
        regroup_model=False,
    ),
    _step(
        {"di": [], "dd": ["fast_phase", "medium_phase"]},
    ),
]

# Duplicate the final self-cal settings as the final-pass definition. With the
# demo parset's equal selfcal/final data fractions, Rapthor will usually skip an
# extra final pass, but keeping the final step explicit exercises the normal
# strategy shape.
strategy_steps.append(
    _step(
        {"di": [], "dd": ["fast_phase", "medium_phase"]},
        max_nmiter=12,
        regroup_model=False,
    )
)

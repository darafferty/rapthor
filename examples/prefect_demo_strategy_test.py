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
    # Select demo calibrators by count rather than a fixed flux threshold. The
    # tiny fixture data can produce a low-flux placeholder source between cycles,
    # so this quick strategy avoids slow-gain amplitude calibration.
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
        "do_slowgain_solve": "slow_gains" in calibration_strategy.get("dd", []),
        "do_fulljones_solve": "full_jones" in calibration_strategy.get("di", []),
        "calibration_strategy": calibration_strategy,
    }
    step.update(overrides)
    return step


strategy_steps = [
    _step(
        {"di": [], "dd": ["fast_phase", "medium_phase"]},
        max_nmiter=8,
    ),
    _step(
        {"di": [], "dd": ["fast_phase", "medium_phase"]},
        max_nmiter=10,
    ),
    _step(
        {"di": ["full_jones"], "dd": []},
        max_nmiter=12,
        regroup_model=False,
    ),
]

# Duplicate the final self-cal settings as the final-pass definition. With the
# demo parset's equal selfcal/final data fractions, Rapthor will usually skip an
# extra final pass, but keeping the final step explicit exercises the normal
# strategy shape.
strategy_steps.append(
    _step(
        {"di": ["full_jones"], "dd": []},
        max_nmiter=12,
        regroup_model=False,
    )
)

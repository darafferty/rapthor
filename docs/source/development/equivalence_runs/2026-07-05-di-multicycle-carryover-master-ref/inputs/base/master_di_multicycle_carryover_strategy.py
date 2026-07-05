"""Master-compatible two-cycle DI full-Jones carry-over scenario.

Legacy master cannot express a pure DI-only calibration cycle through the
strategy interface: ``do_fulljones_solve`` runs after the DD calibration branch.
This scenario therefore keeps DD calibration phase-only in both cycles and uses
the repeated DI full-Jones solve to expose the master/current carry-over
contract.
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
    "regroup_model": False,
    "auto_mask": 5.0,
    "auto_mask_nmiter": 1,
    "channel_width_hz": 48828.125,
    "threshisl": 3.0,
    "threshpix": 5.0,
    "max_nmiter": 6,
    "do_fulljones_solve": True,
    "do_slowgain_solve": False,
}


def _step(**overrides):
    step = dict(COMMON_SETTINGS)
    step.update(overrides)
    return step


strategy_steps = [
    _step(),
    _step(),
    _step(),
]

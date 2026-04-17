"""Module for pytest fixtures."""

import pytest

COMMON_STRATEGY_SETTINGS = {
    "channel_width_hz": 195312.5,
    # Set slow-gain and fulljones solves to False except when required
    "do_slowgain_solve": False,
    "do_fulljones_solve": False,
    # Don't remove bright outliers or in-field sources -- image full field
    "peel_outliers": False,
    "peel_bright_sources": False,
    # Fast phase (ionosphere) and slow gain (beam) time intervals (s)
    "fast_timestep_sec": 32.0,
    "medium_timestep_sec": 120.0,
    "slow_timestep_sec": 600.0,
    # Turn off flux-scale bootstrapping
    "do_normalize": False,
    # PyBDSF settings
    "auto_mask": 5.0,
    "auto_mask_nmiter": 2,
    "threshisl": 3.0,
    "threshpix": 5.0,
    # Constrain max nr of imaging major cycles
    "max_nmiter": 12,
    # Disable regrouping of sky model
    "regroup_model": True,
    # Max distance allowed between selected DDE calibrators
    "max_distance": None,  # no distance constraint
    # Don't check for self-cal convergence
    "do_check": False,
    "target_flux": 0.3,
    "max_directions": 4,
}


def make_strategy_step(**overrides):
    """Helper to create a strategy step with settings and overrides."""
    return {**COMMON_STRATEGY_SETTINGS, **overrides}


@pytest.fixture
def single_loop_strategy_path(tmp_path):
    """Fixture to generate a strategy file for a single self-calibration loop."""
    strategy_steps = [make_strategy_step(do_calibrate=True, do_image=True)]
    strategy_content = f"strategy_steps = {strategy_steps}"
    strategy_path = tmp_path / "single_loop_strategy.py"
    strategy_path.write_text(strategy_content)
    return strategy_path

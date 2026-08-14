"""Shared scientific metadata for supported calibration solve types."""

SOLUTION_INTERVAL_BY_SOLVE_TYPE = {
    "fast_phase": "fast",
    "medium_phase": "medium",
    "slow_gains": "slow",
    "full_jones": "fulljones",
}

MODE_BY_SOLVE = {
    "fast_phase": "scalarphase",
    "medium_phase": "scalarphase",
    "slow_gains": "diagonal",
    "full_jones": "fulljones",
}

INTERVAL_KEYS_BY_SOLVE = {
    "fast_phase": ("solint_fast_timestep", "solint_fast_freqstep"),
    "medium_phase": ("solint_medium_timestep", "solint_medium_freqstep"),
    "slow_gains": ("solint_slow_timestep", "solint_slow_freqstep"),
    "full_jones": ("solint_fulljones_timestep", "solint_fulljones_freqstep"),
}

TARGET_TIMESTEP_BY_SOLVE = {
    "fast_phase": "fast_timestep_sec",
    "medium_phase": "medium_timestep_sec",
    "slow_gains": "slow_timestep_sec",
    "full_jones": "fulljones_timestep_sec",
}

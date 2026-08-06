"""Shared calibration strategy definitions and resolution helpers."""

FIELD_PREFIX_BY_SOLVE = {
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


def resolve_calibration_strategy(
    *,
    calibration_strategy,
    do_slowgain_solve,
    do_fulljones_solve,
):
    """Return an explicit calibration strategy and whether legacy defaults were used."""
    if calibration_strategy:
        return {mode: list(solves or []) for mode, solves in calibration_strategy.items()}, False

    dd_solves = ["fast_phase", "medium_phase"]
    if do_slowgain_solve:
        dd_solves.append("slow_gains")
    di_solves = ["full_jones"] if do_fulljones_solve else []
    return {"dd": dd_solves, "di": di_solves}, True


def resolve_calibration_solves(
    mode,
    *,
    calibration_strategy,
    do_slowgain_solve,
    do_fulljones_solve,
):
    """Return the ordered solve list for one calibration mode."""
    strategy, defaulted = resolve_calibration_strategy(
        calibration_strategy=calibration_strategy,
        do_slowgain_solve=do_slowgain_solve,
        do_fulljones_solve=do_fulljones_solve,
    )
    return strategy.get(mode, []), defaulted

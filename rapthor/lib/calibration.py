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
    solve: (f"solint_{prefix}_timestep", f"solint_{prefix}_freqstep")
    for solve, prefix in FIELD_PREFIX_BY_SOLVE.items()
}

TARGET_TIMESTEP_BY_SOLVE = {
    solve: f"{prefix}_timestep_sec" for solve, prefix in FIELD_PREFIX_BY_SOLVE.items()
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

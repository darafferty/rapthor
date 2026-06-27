"""Pure calibration solve-plan helpers for the Calibrate operation."""

from dataclasses import dataclass
from typing import Mapping, Optional

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


@dataclass(frozen=True)
class CalibrationSolve:
    """Resolved mapping from a strategy solve to a DP3 solve slot."""

    solve_type: str
    slot: int
    mode: str
    output_prefix: str
    collected_h5parm: str
    timestep_key: str
    freqstep_key: str
    field_prefix: str

    @property
    def step(self):
        return f"solve{self.slot}"

    def output_h5parms(self, ntimechunks):
        return [f"{self.output_prefix}_{index}.h5parm" for index in range(ntimechunks)]


def requested_calibration_solves(
    mode: str,
    calibration_strategy: Optional[Mapping[str, list[str]]],
    do_slowgain_solve: bool,
    *,
    strategy_defaulted: bool = False,
) -> tuple[list[str], bool]:
    """Return requested solve types and whether they came from default strategy."""
    if calibration_strategy is not None and mode in calibration_strategy:
        return list(calibration_strategy.get(mode) or []), strategy_defaulted

    if mode == "dd":
        solves = ["fast_phase", "medium_phase"]
        if do_slowgain_solve:
            solves.append("slow_gains")
        return solves, True

    if mode == "di":
        return ["full_jones"], True

    raise ValueError(f"Unsupported calibration mode: {mode}")


def build_calibration_solve_plan(
    mode: str,
    requested_solves: list[str],
    *,
    defaulted_strategy: bool = False,
) -> list[CalibrationSolve]:
    """Build the ordered DP3 solve-slot plan for a calibration cycle."""
    expanded_solves = list(requested_solves)

    if (
        mode == "dd"
        and defaulted_strategy
        and expanded_solves == ["fast_phase", "medium_phase", "slow_gains"]
    ):
        expanded_solves.append("medium_phase")

    if len(expanded_solves) > 4:
        raise ValueError("A calibration cycle can contain at most four solve slots")

    medium_count = 0
    solve_plan = []
    for slot, solve_type in enumerate(expanded_solves, start=1):
        if solve_type not in MODE_BY_SOLVE:
            raise ValueError(f"Unsupported solve type: {solve_type}")
        if solve_type == "medium_phase":
            medium_count += 1
        solve_plan.append(build_calibration_solve_slot(mode, solve_type, slot, medium_count))
    return solve_plan


def build_calibration_dp3_steps(
    bda_timebase: float,
    bda_frequencybase: float,
    *,
    all_channels_regular: bool,
    use_image_based_predict: bool,
    do_slowgain_solve: bool,
    solve_steps: Optional[list[str]] = None,
    preapply_solutions: bool = False,
) -> list[str]:
    """Build the DP3 step chain for calibration solves."""
    if solve_steps is None:
        if do_slowgain_solve:
            common_steps = ["solve1", "solve2", "solve3", "solve4"]
        else:
            common_steps = ["solve1", "solve2"]
    else:
        common_steps = list(solve_steps)

    if (
        (bda_timebase > 0 or bda_frequencybase > 0)
        and all_channels_regular
        and not use_image_based_predict
    ):
        common_steps = ["avg", *common_steps, "null"]

    if preapply_solutions and not use_image_based_predict:
        common_steps = ["applycal", *common_steps]

    if use_image_based_predict:
        preprocessing_steps = (
            ["predict", "applybeam", "applycal"] if preapply_solutions else ["predict", "applybeam"]
        )
        return preprocessing_steps + common_steps

    return common_steps


def build_calibration_preapply_steps(
    mode: str,
    *,
    has_di_h5parm: bool,
    has_fulljones_h5parm: bool,
    apply_amplitudes: bool,
    apply_normalizations: bool,
    calibration_strategy: Optional[Mapping[str, list[str]]] = None,
) -> list[str]:
    """
    Build the ordered DP3 applycal step names used before calibration solves.

    File existence checks and FileRecord conversion stay with the Calibrate
    adapter; this helper only decides which steps are needed from resolved
    inputs.
    """
    steps = []
    if mode == "dd" and has_di_h5parm:
        steps.append("fastphase")
        di_strategy = (calibration_strategy or {}).get("di", [])
        di_has_phase_solves = any(solve in {"fast_phase", "medium_phase"} for solve in di_strategy)
        if apply_amplitudes and not di_has_phase_solves:
            steps.append("slowgain")

    if mode == "dd" and has_fulljones_h5parm:
        steps.append("fulljones")

    if apply_normalizations:
        steps.append("normalization")

    return steps


def build_calibration_solve_slot(
    mode: str,
    solve_type: str,
    slot: int,
    medium_count: int,
) -> CalibrationSolve:
    """Build one resolved solve-slot entry."""
    output_prefix, collected_h5parm = solve_output_names(mode, solve_type, medium_count)
    timestep_key, freqstep_key = INTERVAL_KEYS_BY_SOLVE[solve_type]

    return CalibrationSolve(
        solve_type=solve_type,
        slot=slot,
        mode=MODE_BY_SOLVE[solve_type],
        output_prefix=output_prefix,
        collected_h5parm=collected_h5parm,
        timestep_key=timestep_key,
        freqstep_key=freqstep_key,
        field_prefix=FIELD_PREFIX_BY_SOLVE[solve_type],
    )


def solve_output_names(mode: str, solve_type: str, medium_count: int) -> tuple[str, str]:
    """Return per-chunk output prefix and collected h5parm filename."""
    if solve_type == "fast_phase":
        suffix = "_di" if mode == "di" else ""
        return f"fast_phase{suffix}", f"fast_phases{suffix}.h5parm"
    if solve_type == "medium_phase":
        medium_name = "medium2" if medium_count > 1 else "medium1"
        suffix = "_di" if mode == "di" else ""
        return f"{medium_name}_phase{suffix}", f"{medium_name}_phases{suffix}.h5parm"
    if solve_type == "slow_gains":
        if mode == "di":
            return "slow_gains_di", "slow_gains_di.h5parm"
        return "slow_gain", "slow_gains.h5parm"
    if solve_type == "full_jones":
        return "fulljones_gain", "fulljones_solutions.h5"

    raise ValueError(f"Unsupported solve type: {solve_type}")

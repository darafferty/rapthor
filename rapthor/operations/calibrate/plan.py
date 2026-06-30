"""Pure calibration solve-plan helpers for the Calibrate operation."""

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np

from rapthor.lib.strategy import default_calibration_strategy

SOLUTION_INTERVAL_BY_SOLVE_TYPE = {
    "fast_phase": "fast",
    "medium_phase": "medium",
    "slow_gains": "slow",
    "full_jones": "fulljones",
}


def solution_interval_for_solve_type(solve_type: str) -> str:
    """Return the Field solution-interval family used by a solve type."""
    return SOLUTION_INTERVAL_BY_SOLVE_TYPE[solve_type]

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

SUPERTERP_STATIONS_BY_ANTENNA = {
    "HBA": (
        "CS002HBA0",
        "CS003HBA0",
        "CS004HBA0",
        "CS005HBA0",
        "CS006HBA0",
        "CS007HBA0",
        "CS002HBA1",
        "CS003HBA1",
        "CS004HBA1",
        "CS005HBA1",
        "CS006HBA1",
        "CS007HBA1",
    ),
    "LBA": (
        "CS002LBA",
        "CS003LBA",
        "CS004LBA",
        "CS005LBA",
        "CS006LBA",
        "CS007LBA",
    ),
}

CORE_STATIONS_BY_ANTENNA = {
    "HBA": (
        "CS001HBA0",
        "CS002HBA0",
        "CS003HBA0",
        "CS004HBA0",
        "CS005HBA0",
        "CS006HBA0",
        "CS007HBA0",
        "CS011HBA0",
        "CS013HBA0",
        "CS017HBA0",
        "CS021HBA0",
        "CS024HBA0",
        "CS026HBA0",
        "CS028HBA0",
        "CS030HBA0",
        "CS031HBA0",
        "CS032HBA0",
        "CS101HBA0",
        "CS103HBA0",
        "CS201HBA0",
        "CS301HBA0",
        "CS302HBA0",
        "CS401HBA0",
        "CS501HBA0",
        "CS001HBA1",
        "CS002HBA1",
        "CS003HBA1",
        "CS004HBA1",
        "CS005HBA1",
        "CS006HBA1",
        "CS007HBA1",
        "CS011HBA1",
        "CS013HBA1",
        "CS017HBA1",
        "CS021HBA1",
        "CS024HBA1",
        "CS026HBA1",
        "CS028HBA1",
        "CS030HBA1",
        "CS031HBA1",
        "CS032HBA1",
        "CS101HBA1",
        "CS103HBA1",
        "CS201HBA1",
        "CS301HBA1",
        "CS302HBA1",
        "CS401HBA1",
        "CS501HBA1",
    ),
    "LBA": (
        "CS001LBA",
        "CS002LBA",
        "CS003LBA",
        "CS004LBA",
        "CS005LBA",
        "CS006LBA",
        "CS007LBA",
        "CS011LBA",
        "CS013LBA",
        "CS017LBA",
        "CS021LBA",
        "CS024LBA",
        "CS026LBA",
        "CS028LBA",
        "CS030LBA",
        "CS031LBA",
        "CS032LBA",
        "CS101LBA",
        "CS103LBA",
        "CS201LBA",
        "CS301LBA",
        "CS302LBA",
        "CS401LBA",
        "CS501LBA",
    ),
}

NEAREST_REMOTE_STATIONS_BY_ANTENNA = {
    "HBA": (
        "RS106HBA0",
        "RS205HBA0",
        "RS305HBA0",
        "RS306HBA0",
        "RS503HBA0",
        "RS106HBA1",
        "RS205HBA1",
        "RS305HBA1",
        "RS306HBA1",
        "RS503HBA1",
    ),
    "LBA": (
        "RS106LBA",
        "RS205LBA",
        "RS305LBA",
        "RS306LBA",
        "RS503LBA",
    ),
}


@dataclass(frozen=True)
class CalibrationSolve:
    """Resolved mapping from a strategy solve to a DP3 solve slot."""

    solve_type: str
    solution_label: str
    slot: int
    mode: str
    output_prefix: str
    collected_h5parm: str
    timestep_key: str
    freqstep_key: str
    medium_index: Optional[int] = None

    @property
    def step(self):
        return f"solve{self.slot}"

    def output_h5parms(self, ntimechunks):
        return [f"{self.output_prefix}_{index}.h5parm" for index in range(ntimechunks)]


def requested_calibration_solves(
    mode: str,
    calibration_strategy: Optional[Mapping[str, list[str]]],
    *,
    strategy_defaulted: bool = False,
) -> tuple[list[str], bool]:
    """Return requested solve types and whether they came from default strategy."""
    strategy = calibration_strategy or default_calibration_strategy()
    if mode in strategy:
        return list(strategy.get(mode) or []), strategy_defaulted or calibration_strategy is None

    raise ValueError(f"Unsupported calibration mode: {mode}")


def build_calibration_solve_plan(
    mode: str,
    requested_solves: list[str],
    *,
    defaulted_strategy: bool = False,
) -> list[CalibrationSolve]:
    """Build the ordered DP3 solve-slot plan for a calibration cycle."""
    expanded_solves = list(requested_solves)

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
    has_slow_gain_solve: bool = False,
    solve_steps: Optional[list[str]] = None,
    preapply_solutions: bool = False,
) -> list[str]:
    """Build the DP3 step chain for calibration solves."""
    if solve_steps is None:
        if has_slow_gain_solve:
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
    solution_label = solve_solution_label(solve_type, medium_count)
    output_prefix, collected_h5parm = solve_output_names(mode, solve_type, solution_label)
    timestep_key, freqstep_key = INTERVAL_KEYS_BY_SOLVE[solve_type]

    return CalibrationSolve(
        solve_type=solve_type,
        solution_label=solution_label,
        slot=slot,
        mode=MODE_BY_SOLVE[solve_type],
        output_prefix=output_prefix,
        collected_h5parm=collected_h5parm,
        timestep_key=timestep_key,
        freqstep_key=freqstep_key,
        medium_index=medium_count if solve_type == "medium_phase" else None,
    )


def build_calibration_solve_slot_inputs(
    slot: int,
    solve_type: str,
    *,
    ntimechunks: int,
    datause: object,
    solutions_per_direction: object,
    smoothness_dd_factors: object,
    smoothnessconstraint: float,
    antenna_constraint: str = "[]",
    include_smoothnessreffrequency: bool = False,
    smoothnessreffrequency: Optional[object] = None,
    include_smoothnessrefdistance: bool = False,
    smoothnessrefdistance: Optional[object] = None,
) -> dict[str, object]:
    """
    Build per-slot calibration inputs from resolved field/observation values.

    The operation adapter still reads values from Field. This helper keeps the
    slot naming, smoothness scaling, and optional fast/medium versus slow-gain
    defaults in one testable place.
    """
    solution_interval = solution_interval_for_solve_type(solve_type)
    inputs = {
        f"solve{slot}_datause": datause,
        f"solve{slot}_solutions_per_direction": solutions_per_direction,
        f"solve{slot}_smoothness_dd_factors": smoothness_dd_factors,
        f"solve{slot}_smoothnessconstraint": smoothnessconstraint / np.min(smoothness_dd_factors),
        f"solve{slot}_antennaconstraint": (
            antenna_constraint if solution_interval in {"fast", "medium"} else "[]"
        ),
    }

    if include_smoothnessreffrequency:
        inputs[f"solve{slot}_smoothnessreffrequency"] = (
            smoothnessreffrequency
            if solution_interval in {"fast", "medium"}
            else [0] * ntimechunks
        )

    if include_smoothnessrefdistance:
        inputs[f"solve{slot}_smoothnessrefdistance"] = (
            smoothnessrefdistance if solution_interval in {"fast", "medium"} else None
        )

    return inputs


def build_calibration_superterp_stations(antenna: str, stations: list[str]) -> list[str]:
    """Return superterp station names present in the observation station list."""
    superterp_stations = SUPERTERP_STATIONS_BY_ANTENNA.get(antenna, ())
    return [station for station in superterp_stations if station in stations]


def build_calibration_core_stations(
    antenna: str,
    stations: list[str],
    *,
    include_nearest_remote: bool = True,
) -> list[str]:
    """Return core calibration stations present in the observation station list."""
    core_stations = list(CORE_STATIONS_BY_ANTENNA.get(antenna, ()))
    if include_nearest_remote:
        core_stations.extend(NEAREST_REMOTE_STATIONS_BY_ANTENNA.get(antenna, ()))
    return [station for station in core_stations if station in stations]


def build_calibration_core_baseline_selection(antenna: str, stations: list[str]) -> str:
    """Return the DP3 baseline-selection string for core-station calibration."""
    core_stations = build_calibration_core_stations(antenna, stations)
    non_core_stations = [station for station in stations if station not in core_stations]
    return f"[CR]*&&;!{';!'.join(non_core_stations)}"


def solve_solution_label(solve_type: str, medium_count: int = 0) -> str:
    """Return the named solution product represented by a solve."""
    if solve_type == "fast_phase":
        return "fast"
    if solve_type == "medium_phase":
        if medium_count < 1:
            raise ValueError("medium_phase solves require a positive medium_count")
        return f"medium{medium_count}"
    if solve_type == "slow_gains":
        return "slow"
    if solve_type == "full_jones":
        return "fulljones"
    raise ValueError(f"Unsupported solve type: {solve_type}")


def solve_output_names(mode: str, solve_type: str, solution_label: str) -> tuple[str, str]:
    """Return per-chunk output prefix and collected h5parm filename."""
    if solve_type == "fast_phase":
        suffix = "_di" if mode == "di" else ""
        return f"fast_phase{suffix}", f"fast_phases{suffix}.h5parm"
    if solve_type == "medium_phase":
        suffix = "_di" if mode == "di" else ""
        return f"{solution_label}_phase{suffix}", f"{solution_label}_phases{suffix}.h5parm"
    if solve_type == "slow_gains":
        if mode == "di":
            return "slow_gains_di", "slow_gains_di.h5parm"
        return "slow_gain", "slow_gains.h5parm"
    if solve_type == "full_jones":
        return "fulljones_gain", "fulljones_solutions.h5"

    raise ValueError(f"Unsupported solve type: {solve_type}")

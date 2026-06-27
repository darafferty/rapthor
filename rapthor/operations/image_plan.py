"""Pure planning helpers for the Image operation."""

from collections.abc import Mapping
from typing import Optional

SOLVE_TYPE_TO_APPLYCAL_STEP = {
    "fast_phase": "fastphase",
    "medium_phase": "mediumphase",
    "slow_gains": "slowgain",
    "full_jones": "fulljones",
}


def build_image_applycal_steps(
    calibration_strategy: Optional[Mapping[str, list[str]]],
    *,
    dd_h5parm: Optional[str],
    di_h5parm: Optional[str],
    has_fulljones_h5parm: bool,
    use_facets: bool,
    apply_amplitudes: bool,
    apply_normalizations: bool,
    apply_none: bool,
) -> tuple[list[str], Optional[str]]:
    """
    Build prepare-imaging applycal steps and select the scalar h5parm to apply.

    The Image adapter resolves current-cycle h5parm filenames and converts
    selected files to FileRecord payload values. This helper only decides the
    ordered DP3 step names and which scalar h5parm the selected steps share.
    """
    if apply_none:
        return [], None

    strategy = calibration_strategy or {}
    di_phase_solves = {
        solve for solve in strategy.get("di", []) if solve in {"fast_phase", "medium_phase"}
    }
    prefer_dd_scalar = dd_h5parm is not None and any(
        solve != "full_jones" for solve in strategy.get("dd", [])
    )

    steps = []
    selected_scalar_h5parm = None
    for mode, solves in strategy.items():
        for solve in solves:
            if solve not in SOLVE_TYPE_TO_APPLYCAL_STEP:
                continue
            if solve == "full_jones":
                if has_fulljones_h5parm:
                    steps.append(SOLVE_TYPE_TO_APPLYCAL_STEP[solve])
                continue

            scalar_h5parm = dd_h5parm if mode == "dd" else di_h5parm
            if mode == "dd" and use_facets:
                if scalar_h5parm is not None and selected_scalar_h5parm is None:
                    selected_scalar_h5parm = scalar_h5parm
                continue
            if solve == "slow_gains" and not apply_amplitudes:
                continue
            if mode == "di" and solve == "slow_gains" and di_phase_solves:
                continue
            if prefer_dd_scalar and mode != "dd":
                continue
            if scalar_h5parm is None:
                continue

            if selected_scalar_h5parm is None:
                selected_scalar_h5parm = scalar_h5parm
            if scalar_h5parm == selected_scalar_h5parm:
                step = SOLVE_TYPE_TO_APPLYCAL_STEP[solve]
                if mode == "di" and solve in {"fast_phase", "medium_phase"}:
                    step = "fastphase"
                if step not in steps:
                    steps.append(step)

    if apply_normalizations:
        steps.append("normalization")

    return steps, selected_scalar_h5parm

"""Pure planning helpers for the Image operation."""

from collections.abc import Mapping
from typing import Optional, Union

SOLVE_TYPE_TO_APPLYCAL_STEP = {
    "fast_phase": "fastphase",
    "medium_phase": "mediumphase",
    "slow_gains": "slowgain",
    "full_jones": "fulljones",
}


def is_only_pol_I(image_pol: Union[list[str], str, None]) -> bool:
    """Return whether only Stokes I is being imaged."""
    if image_pol is None:
        return False
    if isinstance(image_pol, str):
        return image_pol.lower() == "i"
    if isinstance(image_pol, list):
        return len(image_pol) == 1 and image_pol[0].lower() == "i"
    return False


def build_image_wsclean_control_inputs(
    image_pol: Union[list[str], str, None],
    pol_combine_method: str,
    sector_niters: list[int],
    *,
    disable_clean: bool,
) -> tuple[object, bool, list[int]]:
    """Build WSClean polarization and clean-iteration controls."""
    link_polarizations = False
    join_polarizations = False
    if not is_only_pol_I(image_pol):
        if pol_combine_method == "link":
            # WSClean accepts a reference polarization string for linked cleaning.
            link_polarizations = "I"
        else:
            join_polarizations = True

    wsclean_niter = [0] * len(sector_niters) if disable_clean else list(sector_niters)
    return link_polarizations, join_polarizations, wsclean_niter


def build_image_facet_solution_controls(
    image_pol: Union[list[str], str, None],
    *,
    apply_amplitudes: bool,
    apply_diagonal_solutions: bool,
) -> dict[str, object]:
    """Build WSClean facet solution controls for scalar or diagonal visibilities."""
    controls = {
        "soltabs": "amplitude000,phase000" if apply_amplitudes else "phase000",
        "diagonal_visibilities": False,
        "scalar_visibilities": False,
    }
    if not is_only_pol_I(image_pol):
        return controls

    if apply_amplitudes and apply_diagonal_solutions:
        controls["diagonal_visibilities"] = True
    else:
        controls["scalar_visibilities"] = True
    return controls


def build_image_prepare_data_steps(
    *,
    preapply_solutions: bool,
    average_visibilities: bool,
    image_bda_timebase: float,
    all_channels_regular: bool,
    apply_screens: bool,
) -> list[str]:
    """
    Build the ordered DP3 steps for preparing imaging visibilities.

    The Image adapter determines whether pre-application has any concrete
    applycal steps and whether observations have regular channels. This helper
    only owns the step-order rules.
    """
    steps = ["applybeam", "shift"]
    if preapply_solutions:
        steps.append("applycal")
    if average_visibilities:
        steps.append("avg")
    if image_bda_timebase > 0 and all_channels_regular and not apply_screens:
        steps.append("bdaavg")
    return steps


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

"""Pure pipeline-planning helpers for top-level Rapthor orchestration."""

SUPPORTED_PIPELINE_FEATURES = frozenset(
    {
        "calibration",
        "calibration_dd",
        "calibration_di",
        "calibration_dd_then_di",
        "calibration_di_then_dd",
        "clean_disabled_full_stokes",
        "concatenate",
        "final_cycle",
        "full_stokes_imaging",
        "hybrid_screens",
        "image",
        "image_cube",
        "initial_skymodel",
        "mpi_wsclean",
        "normalize",
        "peel_bright_sources",
        "peel_outliers",
        "predict_dd",
        "repeat_final_cycle",
        "selfcal",
        "selfcal_check",
        "shared_facet_rw",
        "solve_dd_fast_phase",
        "solve_dd_full_jones",
        "solve_dd_medium_phase",
        "solve_dd_slow_gains",
        "solve_di_fast_phase",
        "solve_di_full_jones",
        "solve_di_medium_phase",
        "solve_di_slow_gains",
    }
)


def collect_pipeline_features(field: object, strategy_steps: list[dict], parset: dict) -> set[str]:
    """Return feature names requested by a pipeline run."""
    features = set()

    if any(len(obs) > 1 for obs in getattr(field, "epoch_observations", [])):
        features.add("concatenate")
    if parset.get("generate_initial_skymodel"):
        features.add("initial_skymodel")
    if len(strategy_steps) > 1:
        features.add("selfcal")
    if parset.get("ntimes_to_repeat_final_cycle", 0):
        features.add("repeat_final_cycle")

    if getattr(field, "make_quv_images", False):
        features.add("full_stokes_imaging")
        if getattr(field, "disable_iquv_clean", False):
            features.add("clean_disabled_full_stokes")
    if getattr(field, "save_image_cube", False):
        features.add("image_cube")
    if getattr(field, "use_mpi", False) or _truthy(
        _parset_get(parset, "imaging_specific", "use_mpi", False)
    ):
        features.add("mpi_wsclean")
    if getattr(field, "dde_mode", None) == "hybrid":
        features.add("hybrid_screens")
    if _truthy(_parset_get(parset, "imaging_specific", "shared_facet_rw", False)):
        features.add("shared_facet_rw")

    for step in strategy_steps:
        if step.get("do_calibrate"):
            features.add("calibration")
            strategy = step.get("calibration_strategy", {})
            enabled_modes = _enabled_calibration_modes(strategy)
            if enabled_modes == ["di", "dd"]:
                features.add("calibration_di_then_dd")
            elif enabled_modes == ["dd", "di"]:
                features.add("calibration_dd_then_di")

            for mode in strategy.keys():
                if mode not in {"di", "dd"}:
                    features.add(f"calibration_mode_{mode}")
            for mode in enabled_modes:
                features.add(f"calibration_{mode}")
                for solve in strategy.get(mode, []):
                    features.add(f"solve_{mode}_{solve}")
        if step.get("do_predict"):
            features.add("predict_dd")
        if step.get("do_image"):
            features.add("image")
        if step.get("do_normalize"):
            features.add("normalize")
        if step.get("do_check"):
            features.add("selfcal_check")
        if step.get("peel_outliers"):
            features.add("peel_outliers")
        if step.get("peel_bright_sources"):
            features.add("peel_bright_sources")

    if strategy_steps:
        features.add("final_cycle")

    return features


def calibration_mode_flags(strategy: dict) -> dict[str, bool]:
    """Return enabled calibration modes while preserving strategy order."""
    calibration_modes = ["di", "dd"]
    if not any(mode in strategy for mode in calibration_modes):
        raise ValueError(
            f"Calibration strategy {strategy} does not contain any of the "
            f"calibration modes {calibration_modes}"
        )
    return {mode: bool(strategy.get(mode, [])) for mode in strategy.keys()}


def _truthy(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "on"}
    return bool(value)


def _parset_get(parset: dict, section: str, key: str, default: object = None) -> object:
    section_values = parset.get(section, {})
    if isinstance(section_values, dict):
        return section_values.get(key, default)
    return default


def _enabled_calibration_modes(strategy: dict) -> list[str]:
    return [mode for mode, solves in strategy.items() if mode in {"di", "dd"} and solves]

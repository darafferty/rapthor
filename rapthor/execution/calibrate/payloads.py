"""Payload contracts and builders for calibration execution."""

import os
from typing import Mapping, Optional, TypedDict

from rapthor.execution.calibrate.commands import parse_steps
from rapthor.execution.payloads import (
    assert_serializable_payload,
)
from rapthor.execution.payloads import (
    validate_basename as _validate_basename,
)
from rapthor.execution.payloads import (
    validate_int_list as _validate_int_list,
)
from rapthor.execution.payloads import (
    validate_string_list as _validate_string_list,
)
from rapthor.lib.records import directory_record_path, file_record_path

# These are the DP3 step names supported for the standalone applycal that runs
# before DD solves. `fastphase` applies the phase000 soltab from the selected
# scalar h5parm; for DI fast+medium this is already the combined scalar product.
SUPPORTED_DD_PREAPPLY_STEPS = {"fastphase", "slowgain", "fulljones", "normalization"}


class CalibrateOutputPayload(TypedDict):
    """Serializable filename/path pair for calibration output h5parms."""

    filename: str
    path: str


class CalibrateSolveSlotPayload(TypedDict, total=False):
    """Serializable payload for one DP3 solve slot inside a calibration chunk."""

    slot: int
    h5parm: str
    h5parm_path: str
    solint: int
    mode: str
    nchan: int
    solutions_per_direction: Optional[list[object]]
    smoothness_dd_factors: Optional[list[object]]
    smoothnessconstraint: Optional[object]
    smoothnessreffrequency: Optional[object]
    smoothnessrefdistance: Optional[object]
    antennaconstraint: Optional[object]
    keepmodel: Optional[str]
    reusemodel: Optional[str]
    modeldatacolumns: str
    datause: object
    initialsolutions_h5parm: Optional[str]


class CalibrateChunkPayload(TypedDict, total=False):
    """Serializable payload for one calibration or screen-generation chunk."""

    msin: str
    starttime: str
    ntimes: int
    output_h5parm: str
    output_h5parm_path: str
    solve1_solint: int
    solve1_nchan: int
    solve_slots: list[CalibrateSolveSlotPayload]
    bda_maxinterval: object
    bda_minchannels: object
    solint_fast: int
    solint_slow: int


class CalibrateImagePredictPayload(TypedDict):
    """Serializable payload for calibration image-based prediction setup."""

    skymodel: Optional[str]
    model_image_root: str
    model_image_ra_dec: list[str]
    model_image_imsize: list[int]
    model_image_cellsize: object
    model_image_frequency_bandwidth: list[object]
    num_spectral_terms: int
    model_images: list[str]
    ra_mid: object
    dec_mid: object
    facet_region_width_ra: object
    facet_region_width_dec: object
    facet_region_file: str
    facet_region_path: str


class CalibratePayload(TypedDict, total=False):
    """Serializable payload submitted to the calibrate flow."""

    mode: str
    calibration_kind: str
    pipeline_working_dir: str
    data_colname: str
    modeldatacolumn: Optional[str]
    dp3_steps: str
    image_based_predict: bool
    image_predict: Optional[CalibrateImagePredictPayload]
    max_threads: int
    maxiter: int
    llssolver: str
    propagatesolutions: bool
    solveralgorithm: str
    solverlbfgs_dof: float
    solverlbfgs_iter: int
    solverlbfgs_minibatches: int
    stepsize: float
    stepsigma: float
    tolerance: float
    uvlambdamin: float
    correctfreqsmearing: bool
    correcttimesmearing: bool
    collected_h5parm: str
    collected_h5parm_path: str
    collected_h5parms: dict[str, CalibrateOutputPayload]
    combined_h5parm: Optional[CalibrateOutputPayload]
    combined_h5parms: dict[str, CalibrateOutputPayload]
    calibrator_patch_names: list[str]
    calibrator_fluxes: list[float]
    chunks: list[CalibrateChunkPayload]


def _optional_file_path(record: object, name: str) -> Optional[str]:
    if record is None:
        return None
    if isinstance(record, str):
        return record
    if isinstance(record, Mapping) and record.get("class") == "File":
        return file_record_path(record)
    raise ValueError(f"{name} must be a File record, path string, or None")


def _require_sequence(value: object, name: str, length: Optional[int] = None) -> list[object]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    if length is not None and len(value) != length:
        raise ValueError(f"{name} must contain exactly {length} entries")
    return value


def _plain_payload_value(value: object) -> object:
    """Convert array-like scatter values to plain Python containers."""
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        return _plain_payload_value(value.tolist())
    if isinstance(value, tuple):
        return [_plain_payload_value(item) for item in value]
    if isinstance(value, list):
        return [_plain_payload_value(item) for item in value]
    return value


def _scatter_value(value: object, index: int, name: str) -> object:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    if index >= len(value):
        raise ValueError(f"{name} must have at least {index + 1} entries")
    return _plain_payload_value(value[index])


def _optional_scatter_value(input_parms: Mapping[str, object], name: str, index: int) -> object:
    if name not in input_parms:
        return None
    return _scatter_value(input_parms[name], index, name)


def _active_solve_steps(input_parms: Mapping[str, object]) -> list[str]:
    return [step for step in parse_steps(input_parms.get("dp3_steps")) if step.startswith("solve")]


def _first_solve_output(input_parms: Mapping[str, object], slot: int) -> str:
    outputs = input_parms.get(f"output_solve{slot}_h5parm", [])
    return str(outputs[0]) if isinstance(outputs, list) and outputs else ""


def _di_solve_type(input_parms: Mapping[str, object], slot: int) -> str:
    output_name = _first_solve_output(input_parms, slot)
    solve_mode = str(input_parms.get(f"solve{slot}_mode"))
    if output_name.startswith("fulljones_gain_") and solve_mode == "fulljones":
        return "full_jones"
    if output_name.startswith("fast_phase_di_") and solve_mode == "scalarphase":
        return "fast_phase"
    if (
        output_name.startswith("medium1_phase_di_") or output_name.startswith("medium2_phase_di_")
    ) and solve_mode == "scalarphase":
        return "medium_phase"
    if output_name.startswith("slow_gains_di_") and solve_mode == "diagonal":
        return "slow_gains"
    return "unsupported"


def _uses_image_based_predict(steps: list[str]) -> bool:
    return "predict" in steps or "applybeam" in steps


def _validate_applycal_inputs(
    mode: str,
    steps: list[str],
    input_parms: Mapping[str, object],
) -> None:
    if "applycal" not in steps:
        return
    if mode != "dd":
        raise ValueError("Pre-application is only supported for DD calibration")

    applycal_steps_record = input_parms.get("applycal_steps")
    if applycal_steps_record is None:
        raise ValueError("DD pre-application requires applycal_steps")

    applycal_steps = parse_steps(applycal_steps_record)
    if not applycal_steps:
        raise ValueError("DD pre-application requires applycal_steps")

    unsupported_steps = [step for step in applycal_steps if step not in SUPPORTED_DD_PREAPPLY_STEPS]
    if unsupported_steps:
        raise ValueError("Unsupported DD pre-apply step(s): " + ",".join(sorted(unsupported_steps)))

    missing_inputs = []
    if (
        any(step in {"fastphase", "slowgain"} for step in applycal_steps)
        and input_parms.get("applycal_h5parm") is None
    ):
        missing_inputs.append("applycal_h5parm")
    if "fulljones" in applycal_steps and input_parms.get("fulljones_h5parm") is None:
        missing_inputs.append("fulljones_h5parm")
    if "normalization" in applycal_steps and input_parms.get("normalize_h5parm") is None:
        missing_inputs.append("normalize_h5parm")

    if missing_inputs:
        raise ValueError("DD pre-application requires " + ", ".join(missing_inputs))


def _validate_image_predict_inputs(
    mode: str,
    steps: list[str],
    input_parms: Mapping[str, object],
) -> None:
    if not _uses_image_based_predict(steps):
        return
    if mode != "dd":
        raise ValueError("Image-based prediction is only supported for DD calibration")
    if "predict" not in steps or "applybeam" not in steps:
        raise ValueError("DD image-based prediction requires predict and applybeam steps")

    required = [
        "calibration_skymodel_file",
        "model_image_root",
        "model_image_ra_dec",
        "model_image_imsize",
        "model_image_cellsize",
        "model_image_frequency_bandwidth",
        "num_spectral_terms",
        "ra_mid",
        "dec_mid",
        "facet_region_width_ra",
        "facet_region_width_dec",
        "facet_region_file",
    ]
    missing = [name for name in required if input_parms.get(name) is None]
    if missing:
        raise ValueError("DD image-based prediction requires " + ", ".join(missing))


def _validate_screen_inputs(mode: str, input_parms: Mapping[str, object]) -> None:
    if not input_parms.get("generate_screens"):
        return
    if mode != "dd":
        raise ValueError("Screen generation is only supported for DD calibration")

    required = [
        "calibration_skymodel_file",
        "model_image_root",
        "model_image_ra_dec",
        "model_image_imsize",
        "model_image_cellsize",
        "model_image_frequency_bandwidth",
        "num_spectral_terms",
        "ra_mid",
        "dec_mid",
        "facet_region_width_ra",
        "facet_region_width_dec",
        "facet_region_file",
        "output_idgcal_h5parm",
        "solint_solve1_timestep",
        "idgcal_antennaconstraint",
        "combined_h5parms",
    ]
    if input_parms.get("do_slowgain_solve"):
        required.append("solint_slow_timestep")
    missing = [name for name in required if input_parms.get(name) is None]
    if missing:
        raise ValueError("Screen generation requires " + ", ".join(missing))


def _validate_slow_gain_processing_inputs(input_parms: Mapping[str, object]) -> None:
    required = [
        "max_normalization_delta",
        "scale_normalization_delta",
        "phase_center_ra",
        "phase_center_dec",
    ]
    missing = [name for name in required if input_parms.get(name) is None]
    if missing:
        raise ValueError("Slow-gain processing requires " + ", ".join(missing))


def _supported_calibration_kind(mode: str, input_parms: Mapping[str, object]) -> str:
    steps = parse_steps(input_parms.get("dp3_steps"))
    _validate_screen_inputs(mode, input_parms)
    if input_parms.get("generate_screens"):
        return "dd_screen"
    _validate_applycal_inputs(mode, steps, input_parms)
    _validate_image_predict_inputs(mode, steps, input_parms)

    active_solves = [step for step in steps if step.startswith("solve")]
    solve1_mode = str(input_parms.get("solve1_mode"))
    solve2_mode = str(input_parms.get("solve2_mode"))
    solve3_mode = str(input_parms.get("solve3_mode"))
    solve4_mode = str(input_parms.get("solve4_mode"))
    if mode == "di":
        solve_types = [
            _di_solve_type(input_parms, int(step.removeprefix("solve"))) for step in active_solves
        ]
        if solve_types == ["full_jones"]:
            return "di_fulljones"
        if solve_types == ["fast_phase"]:
            return "di_fast_phase"
        if solve_types == ["slow_gains"]:
            _validate_slow_gain_processing_inputs(input_parms)
            return "di_slow"
        if solve_types == ["fast_phase", "medium_phase"]:
            return "di_scalar_phase"
        if solve_types == ["fast_phase", "medium_phase", "slow_gains"]:
            _validate_slow_gain_processing_inputs(input_parms)
            return "di_phase_slow"
        raise ValueError(
            "Only DI full-Jones, fast phase, slow gain, fast/medium phase, "
            "and fast/medium/slow calibration are supported"
        )

    if mode == "dd":
        output_solve1 = input_parms.get("output_solve1_h5parm", [])
        first_solve1_output = output_solve1[0] if isinstance(output_solve1, list) else ""
        if (
            active_solves == ["solve1"]
            and solve1_mode == "scalarphase"
            and str(first_solve1_output).startswith("fast_phase_")
        ):
            return "dd_fast_phase"
        if (
            active_solves == ["solve1"]
            and solve1_mode == "diagonal"
            and str(first_solve1_output).startswith("slow_gain_")
        ):
            _validate_slow_gain_processing_inputs(input_parms)
            return "dd_slow"
        if (
            active_solves == ["solve1", "solve2"]
            and solve1_mode == solve2_mode == "scalarphase"
            and str(first_solve1_output).startswith("fast_phase_")
        ):
            return "dd_phase"
        if (
            active_solves
            in (["solve1", "solve2", "solve3"], ["solve1", "solve2", "solve3", "solve4"])
            and solve1_mode == solve2_mode == "scalarphase"
            and solve3_mode == "diagonal"
            and ("solve4" not in active_solves or solve4_mode == "scalarphase")
            and str(first_solve1_output).startswith("fast_phase_")
        ):
            _validate_slow_gain_processing_inputs(input_parms)
            return "dd_phase_slow"
        raise ValueError(
            "Only DD fast/medium phase and slow-gain calibration is supported in this slice"
        )

    raise ValueError("Only DI and DD calibration payloads are supported")


def _solve_slots_for_kind(calibration_kind: str, input_parms: Mapping[str, object]) -> list[int]:
    if calibration_kind == "dd_screen":
        return []
    if calibration_kind in {"di_fulljones", "di_fast_phase", "di_slow", "dd_fast_phase", "dd_slow"}:
        return [1]
    if calibration_kind == "di_scalar_phase":
        return [1, 2]
    if calibration_kind == "di_phase_slow":
        return [1, 2, 3]
    if calibration_kind in {"dd_phase", "dd_phase_slow"}:
        return [int(step.removeprefix("solve")) for step in _active_solve_steps(input_parms)]
    raise ValueError(f"Unsupported calibration kind: {calibration_kind}")


def _solver_payload_from_inputs(input_parms: Mapping[str, object]) -> dict:
    return {
        "max_threads": int(input_parms["max_threads"]),
        "maxiter": int(input_parms["maxiter"]),
        "llssolver": str(input_parms["llssolver"]),
        "propagatesolutions": bool(input_parms["propagatesolutions"]),
        "solveralgorithm": str(input_parms["solveralgorithm"]),
        "solverlbfgs_dof": float(input_parms["solverlbfgs_dof"]),
        "solverlbfgs_iter": int(input_parms["solverlbfgs_iter"]),
        "solverlbfgs_minibatches": int(input_parms["solverlbfgs_minibatches"]),
        "stepsize": float(input_parms["stepsize"]),
        "stepsigma": float(input_parms["stepsigma"]),
        "tolerance": float(input_parms["tolerance"]),
        "uvlambdamin": float(input_parms["uvlambdamin"]),
        "correctfreqsmearing": bool(input_parms["correctfreqsmearing"]),
        "correcttimesmearing": bool(input_parms["correcttimesmearing"]),
    }


def _solve_slot_from_inputs(
    input_parms: Mapping[str, object],
    pipeline_dir: str,
    slot: int,
    index: int,
    keepmodel: Optional[str] = None,
    reusemodel: Optional[str] = None,
    modeldatacolumns: Optional[str] = None,
) -> CalibrateSolveSlotPayload:
    h5parm = _validate_basename(
        _scatter_value(
            input_parms[f"output_solve{slot}_h5parm"], index, f"output_solve{slot}_h5parm"
        ),
        f"output_solve{slot}_h5parm[{index}]",
    )
    initial_h5parm_keys = {
        1: "fast_initialsolutions_h5parm",
        2: "medium1_initialsolutions_h5parm",
        3: "solve3_initialsolutions_h5parm",
        4: "solve4_initialsolutions_h5parm",
    }
    slot_record = {
        "slot": slot,
        "h5parm": h5parm,
        "h5parm_path": os.path.join(pipeline_dir, h5parm),
        "solint": int(
            _scatter_value(
                input_parms[f"solint_solve{slot}_timestep"], index, f"solint_solve{slot}_timestep"
            )
        ),
        "mode": str(input_parms[f"solve{slot}_mode"]),
        "nchan": int(
            _scatter_value(
                input_parms[f"solint_solve{slot}_freqstep"], index, f"solint_solve{slot}_freqstep"
            )
        ),
        "solutions_per_direction": _optional_scatter_value(
            input_parms, f"solve{slot}_solutions_per_direction", index
        ),
        "smoothness_dd_factors": _optional_scatter_value(
            input_parms, f"solve{slot}_smoothness_dd_factors", index
        ),
        "smoothnessconstraint": input_parms.get(f"solve{slot}_smoothnessconstraint"),
        "smoothnessreffrequency": _optional_scatter_value(
            input_parms, f"solve{slot}_smoothnessreffrequency", index
        ),
        "smoothnessrefdistance": input_parms.get(f"solve{slot}_smoothnessrefdistance"),
        "antennaconstraint": input_parms.get(f"solve{slot}_antennaconstraint"),
        "keepmodel": keepmodel,
        "reusemodel": reusemodel,
    }
    if modeldatacolumns is not None:
        slot_record["modeldatacolumns"] = modeldatacolumns
    datause_key = f"solve{slot}_datause"
    if datause_key in input_parms:
        slot_record["datause"] = input_parms.get(datause_key)
    initial_h5parm_key = initial_h5parm_keys[slot]
    if initial_h5parm_key in input_parms:
        slot_record["initialsolutions_h5parm"] = _optional_file_path(
            input_parms.get(initial_h5parm_key), initial_h5parm_key
        )
    return slot_record


def _image_predict_payload_from_inputs(
    input_parms: Mapping[str, object],
    pipeline_dir: str,
) -> Optional[CalibrateImagePredictPayload]:
    steps = parse_steps(input_parms.get("dp3_steps"))
    if not input_parms.get("generate_screens") and not _uses_image_based_predict(steps):
        return None

    model_root = _validate_basename(input_parms.get("model_image_root"), "model_image_root")
    numterms = int(input_parms["num_spectral_terms"])
    if numterms < 1:
        raise ValueError("num_spectral_terms must be at least 1")

    region_filename = _validate_basename(input_parms.get("facet_region_file"), "facet_region_file")
    return {
        "skymodel": _optional_file_path(
            input_parms.get("calibration_skymodel_file"), "calibration_skymodel_file"
        ),
        "model_image_root": model_root,
        "model_image_ra_dec": [
            str(value)
            for value in _require_sequence(
                input_parms.get("model_image_ra_dec"), "model_image_ra_dec", length=2
            )
        ],
        "model_image_imsize": [
            int(value)
            for value in _require_sequence(
                input_parms.get("model_image_imsize"), "model_image_imsize", length=2
            )
        ],
        "model_image_cellsize": input_parms["model_image_cellsize"],
        "model_image_frequency_bandwidth": list(
            _require_sequence(
                input_parms.get("model_image_frequency_bandwidth"),
                "model_image_frequency_bandwidth",
                length=2,
            )
        ),
        "num_spectral_terms": numterms,
        "model_images": [
            os.path.join(pipeline_dir, f"{model_root}-term-{index}.fits")
            for index in range(numterms)
        ],
        "ra_mid": input_parms["ra_mid"],
        "dec_mid": input_parms["dec_mid"],
        "facet_region_width_ra": input_parms["facet_region_width_ra"],
        "facet_region_width_dec": input_parms["facet_region_width_dec"],
        "facet_region_file": region_filename,
        "facet_region_path": os.path.join(pipeline_dir, region_filename),
    }


def calibrate_payload_from_inputs(
    mode: str,
    input_parms: Mapping[str, object],
    pipeline_working_dir: object,
) -> CalibratePayload:
    """Create a serializable Calibrate flow payload from operation inputs."""
    calibration_kind = _supported_calibration_kind(mode, input_parms)
    dp3_steps = parse_steps(input_parms.get("dp3_steps"))
    image_based_predict = bool(input_parms.get("generate_screens")) or _uses_image_based_predict(
        dp3_steps
    )

    pipeline_dir = str(pipeline_working_dir)
    filenames = input_parms.get("timechunk_filename", [])
    starttimes = input_parms.get("starttime", [])
    ntimes = input_parms.get("ntimes", [])

    if calibration_kind == "dd_screen":
        output_h5parms = input_parms.get("output_idgcal_h5parm", [])
        solint_fast = input_parms.get("solint_solve1_timestep", [])
        scatter_inputs = [filenames, starttimes, ntimes, output_h5parms, solint_fast]
        do_slowgain_solve = bool(input_parms.get("do_slowgain_solve", False))
        solint_slow = input_parms.get("solint_slow_timestep", [])
        if do_slowgain_solve:
            scatter_inputs.append(solint_slow)
        if not all(isinstance(value, list) for value in scatter_inputs):
            raise ValueError("Screen-generation scatter inputs must be lists")
        chunk_count = len(output_h5parms)
        if any(len(value) != chunk_count for value in scatter_inputs):
            raise ValueError("Screen-generation scatter inputs must have the same length")

        combined = _validate_basename(input_parms.get("combined_h5parms"), "combined_h5parms")
        chunks: list[CalibrateChunkPayload] = []
        for index in range(chunk_count):
            h5parm = _validate_basename(
                _scatter_value(output_h5parms, index, "output_idgcal_h5parm"),
                f"output_idgcal_h5parm[{index}]",
            )
            chunk = {
                "msin": directory_record_path(filenames[index]),
                "starttime": str(starttimes[index]),
                "ntimes": int(ntimes[index]),
                "output_h5parm": h5parm,
                "output_h5parm_path": os.path.join(pipeline_dir, h5parm),
                "solint_fast": int(_scatter_value(solint_fast, index, "solint_solve1_timestep")),
            }
            if do_slowgain_solve:
                chunk["solint_slow"] = int(
                    _scatter_value(solint_slow, index, "solint_slow_timestep")
                )
            chunks.append(chunk)

        payload: CalibratePayload = {
            "mode": mode,
            "calibration_kind": calibration_kind,
            "pipeline_working_dir": pipeline_dir,
            "image_based_predict": True,
            "image_predict": _image_predict_payload_from_inputs(input_parms, pipeline_dir),
            "max_threads": int(input_parms["max_threads"]),
            "solverlbfgs_iter": int(input_parms["solverlbfgs_iter"]),
            "idgcal_antennaconstraint": str(input_parms["idgcal_antennaconstraint"]),
            "do_slowgain_solve": do_slowgain_solve,
            "combined_h5parm": {
                "filename": combined,
                "path": os.path.join(pipeline_dir, combined),
            },
            "chunks": chunks,
        }
        assert_serializable_payload(payload)
        return payload

    solve_slots = _solve_slots_for_kind(calibration_kind, input_parms)
    output_solve1_h5parms = input_parms.get("output_solve1_h5parm", [])
    scatter_inputs = [filenames, starttimes, ntimes, output_solve1_h5parms]
    for slot in solve_slots:
        if slot != 1:
            scatter_inputs.append(input_parms.get(f"output_solve{slot}_h5parm", []))
    if not all(isinstance(value, list) for value in scatter_inputs):
        raise ValueError("Calibration scatter inputs must be lists")
    chunk_count = len(output_solve1_h5parms)
    if any(len(value) != chunk_count for value in scatter_inputs):
        raise ValueError("Calibration scatter inputs must have the same length")

    collected_h5parms = {}
    for slot in solve_slots:
        collected = _validate_basename(
            input_parms.get(f"collected_solve{slot}_h5parm"),
            f"collected_solve{slot}_h5parm",
        )
        collected_h5parms[f"solve{slot}"] = {
            "filename": collected,
            "path": os.path.join(pipeline_dir, collected),
        }
    combined_h5parm = None
    if calibration_kind == "di_scalar_phase":
        combined = _validate_basename(
            input_parms.get("combined_solve1_solve2_h5parm"),
            "combined_solve1_solve2_h5parm",
        )
        combined_h5parm = {"filename": combined, "path": os.path.join(pipeline_dir, combined)}

    combined_h5parms = {}
    if calibration_kind in {"dd_phase", "dd_phase_slow", "di_phase_slow"}:
        combined = _validate_basename(
            input_parms.get("combined_solve1_solve2_h5parm"),
            "combined_solve1_solve2_h5parm",
        )
        combined_h5parms["solve1_solve2"] = {
            "filename": combined,
            "path": os.path.join(pipeline_dir, combined),
        }
    if calibration_kind == "dd_phase_slow" and 4 in solve_slots:
        combined = _validate_basename(
            input_parms.get("combined_solve1_solve2_solve4_h5parm"),
            "combined_solve1_solve2_solve4_h5parm",
        )
        combined_h5parms["solve1_solve2_solve4"] = {
            "filename": combined,
            "path": os.path.join(pipeline_dir, combined),
        }
    if calibration_kind in {"dd_phase_slow", "di_phase_slow"}:
        combined = _validate_basename(
            input_parms.get("combined_h5parms"),
            "combined_h5parms",
        )
        combined_h5parms["final"] = {
            "filename": combined,
            "path": os.path.join(pipeline_dir, combined),
        }

    modeldatacolumn = input_parms.get("modeldatacolumn")
    if modeldatacolumn is not None and (
        not isinstance(modeldatacolumn, str) or not modeldatacolumn
    ):
        raise ValueError("modeldatacolumn must be a non-empty string or None")

    chunks: list[CalibrateChunkPayload] = []
    uses_modeldatacolumn = modeldatacolumn is not None and not image_based_predict
    for index in range(chunk_count):
        chunk_solve_slots: list[CalibrateSolveSlotPayload] = []
        for slot in solve_slots:
            keepmodel = None
            reusemodel = None
            slot_modeldatacolumns = None
            if slot == 1:
                keepmodel = "True"
                if image_based_predict:
                    reusemodel = "[predict.*]"
            elif calibration_kind in {
                "di_scalar_phase",
                "di_phase_slow",
                "dd_phase",
                "dd_phase_slow",
            }:
                if image_based_predict:
                    reusemodel = "[predict.*]"
                elif uses_modeldatacolumn and calibration_kind not in {"di_scalar_phase"}:
                    slot_modeldatacolumns = modeldatacolumn
                else:
                    reusemodel = "[solve1.*]"
            if calibration_kind == "dd_phase_slow" and slot in {2, 3}:
                keepmodel = "true"
            chunk_solve_slots.append(
                _solve_slot_from_inputs(
                    input_parms,
                    pipeline_dir,
                    slot=slot,
                    index=index,
                    keepmodel=keepmodel,
                    reusemodel=reusemodel,
                    modeldatacolumns=slot_modeldatacolumns,
                )
            )
        chunk = {
            "msin": directory_record_path(filenames[index]),
            "starttime": str(starttimes[index]),
            "ntimes": int(ntimes[index]),
            "output_h5parm": chunk_solve_slots[0]["h5parm"],
            "output_h5parm_path": chunk_solve_slots[0]["h5parm_path"],
            "solve1_solint": chunk_solve_slots[0]["solint"],
            "solve1_nchan": chunk_solve_slots[0]["nchan"],
            "solve_slots": chunk_solve_slots,
        }
        if "bda_maxinterval" in input_parms:
            chunk["bda_maxinterval"] = _scatter_value(
                input_parms["bda_maxinterval"], index, "bda_maxinterval"
            )
        if "bda_minchannels" in input_parms:
            chunk["bda_minchannels"] = _scatter_value(
                input_parms["bda_minchannels"], index, "bda_minchannels"
            )
        chunks.append(chunk)

    payload: CalibratePayload = {
        "mode": mode,
        "calibration_kind": calibration_kind,
        "pipeline_working_dir": pipeline_dir,
        "data_colname": str(input_parms["data_colname"]),
        "modeldatacolumn": modeldatacolumn,
        "dp3_steps": str(input_parms["dp3_steps"]),
        "image_based_predict": image_based_predict,
        "image_predict": _image_predict_payload_from_inputs(input_parms, pipeline_dir),
        **_solver_payload_from_inputs(input_parms),
        "collected_h5parm": collected_h5parms["solve1"]["filename"],
        "collected_h5parm_path": collected_h5parms["solve1"]["path"],
        "collected_h5parms": collected_h5parms,
        "combined_h5parm": combined_h5parm,
        "combined_h5parms": combined_h5parms,
        "calibrator_patch_names": [
            str(name) for name in input_parms.get("calibrator_patch_names", [])
        ],
        "calibrator_fluxes": [float(flux) for flux in input_parms.get("calibrator_fluxes", [])],
        "chunks": chunks,
    }
    if "smoothnessconstraint_fulljones" in input_parms:
        payload["smoothnessconstraint_fulljones"] = float(
            input_parms["smoothnessconstraint_fulljones"]
        )
    payload.update(
        {
            "applycal_steps": input_parms.get("applycal_steps"),
            "applycal_h5parm": _optional_file_path(
                input_parms.get("applycal_h5parm"), "applycal_h5parm"
            ),
            "fulljones_h5parm": _optional_file_path(
                input_parms.get("fulljones_h5parm"), "fulljones_h5parm"
            ),
            "normalize_h5parm": _optional_file_path(
                input_parms.get("normalize_h5parm"), "normalize_h5parm"
            ),
            "bda_timebase": input_parms.get("bda_timebase"),
            "bda_frequencybase": input_parms.get("bda_frequencybase"),
            "onebeamperpatch": input_parms.get("onebeamperpatch"),
            "parallelbaselines": input_parms.get("parallelbaselines"),
            "sagecalpredict": input_parms.get("sagecalpredict"),
            "sourcedb": _optional_file_path(
                input_parms.get("calibration_skymodel_file"), "calibration_skymodel_file"
            ),
            "directions": None
            if input_parms.get("solve_directions") is None
            else [str(direction) for direction in input_parms["solve_directions"]],
            "do_slowgain_solve": bool(input_parms.get("do_slowgain_solve", False)),
            "solution_combine_mode": input_parms.get("solution_combine_mode"),
            "max_normalization_delta": input_parms.get("max_normalization_delta"),
            "scale_normalization_delta": input_parms.get("scale_normalization_delta"),
            "phase_center_ra": input_parms.get("phase_center_ra"),
            "phase_center_dec": input_parms.get("phase_center_dec"),
        }
    )
    assert_serializable_payload(payload)
    return payload


def _validate_calibrate_solve_slot(
    solve_slot: Mapping[str, object],
    chunk_index: int,
    slot_index: int,
) -> CalibrateSolveSlotPayload:
    _ = int(solve_slot["slot"])
    _validate_basename(
        solve_slot["h5parm"],
        f"chunks[{chunk_index}].solve_slots[{slot_index}].h5parm",
    )
    _ = str(solve_slot["h5parm_path"])
    _ = int(solve_slot["solint"])
    _ = str(solve_slot["mode"])
    _ = int(solve_slot["nchan"])
    for list_key in ("solutions_per_direction", "smoothness_dd_factors"):
        value = solve_slot.get(list_key)
        if value is not None and not isinstance(value, list):
            raise ValueError(
                f"chunks[{chunk_index}].solve_slots[{slot_index}].{list_key} must be a list"
            )
    return solve_slot


def _validate_calibrate_chunk(
    chunk: Mapping[str, object],
    index: int,
    *,
    screen: bool,
) -> CalibrateChunkPayload:
    _ = str(chunk["msin"])
    _ = str(chunk["starttime"])
    _ = int(chunk["ntimes"])
    _validate_basename(chunk["output_h5parm"], f"chunks[{index}].output_h5parm")
    _ = str(chunk["output_h5parm_path"])
    if screen:
        _ = int(chunk["solint_fast"])
        if "solint_slow" in chunk:
            _ = int(chunk["solint_slow"])
        return chunk

    raw_solve_slots = chunk.get("solve_slots", [])
    if not isinstance(raw_solve_slots, list) or not raw_solve_slots:
        raise ValueError(f"chunks[{index}].solve_slots must be a non-empty list")
    for slot_index, solve_slot in enumerate(raw_solve_slots):
        if not isinstance(solve_slot, Mapping):
            raise ValueError(f"chunks[{index}].solve_slots[{slot_index}] must be a mapping")
        _validate_calibrate_solve_slot(solve_slot, index, slot_index)
    return chunk


def _validate_calibrate_image_predict(
    image_predict: object,
) -> Optional[CalibrateImagePredictPayload]:
    if image_predict is None:
        return None
    if not isinstance(image_predict, Mapping):
        raise ValueError("image_predict must be a mapping")
    _ = _optional_file_path(image_predict.get("skymodel"), "image_predict.skymodel")
    _validate_basename(image_predict["model_image_root"], "image_predict.model_image_root")
    _validate_string_list(
        image_predict.get("model_image_ra_dec"), "image_predict.model_image_ra_dec"
    )
    _validate_int_list(
        image_predict.get("model_image_imsize"), "image_predict.model_image_imsize", length=2
    )
    _validate_string_list(image_predict.get("model_images"), "image_predict.model_images")
    _validate_basename(image_predict["facet_region_file"], "image_predict.facet_region_file")
    _ = str(image_predict["facet_region_path"])
    return image_predict


def validate_calibrate_payload(payload: Mapping[str, object]) -> CalibratePayload:
    """Validate an incoming Calibrate flow payload and return its typed shape."""
    mode = str(payload["mode"])
    if mode not in {"di", "dd"}:
        raise ValueError("mode must be 'di' or 'dd'")
    calibration_kind = str(payload["calibration_kind"])
    supported_kinds = {
        "di_fast_phase",
        "di_fulljones",
        "di_phase_slow",
        "di_scalar_phase",
        "di_slow",
        "dd_fast_phase",
        "dd_phase",
        "dd_phase_slow",
        "dd_screen",
        "dd_slow",
    }
    if calibration_kind not in supported_kinds:
        raise ValueError("Unsupported calibration kind")
    _ = str(payload["pipeline_working_dir"])
    raw_chunks = payload.get("chunks", [])
    if not isinstance(raw_chunks, list) or not raw_chunks:
        raise ValueError("chunks must be a non-empty list")
    for index, chunk in enumerate(raw_chunks):
        if not isinstance(chunk, Mapping):
            raise ValueError(f"chunks[{index}] must be a mapping")
        _validate_calibrate_chunk(chunk, index, screen=calibration_kind == "dd_screen")
    if payload.get("image_based_predict"):
        _validate_calibrate_image_predict(payload.get("image_predict"))
    return payload

"""Payload builders for calibration execution."""

import os
from typing import Mapping, Optional

from rapthor.execution.calibrate.commands import parse_steps
from rapthor.execution.calibrate.contracts import (
    CalibrateChunkPayload,
    CalibrateImagePredictPayload,
    CalibratePayload,
    CalibrateSolveSlotPayload,
)
from rapthor.execution.payloads import (
    assert_serializable_payload,
    optional_file_path,
    validate_basename,
    validate_required_list,
)
from rapthor.lib.records import directory_record_path

# These are the DP3 step names supported for the standalone applycal that runs
# before DD solves. `fastphase` applies the phase000 soltab from the selected
# scalar h5parm; for DI fast+medium this is already the combined scalar product.
SUPPORTED_DD_PREAPPLY_STEPS = {"fastphase", "slowgain", "fulljones", "normalization"}
MODE_BY_SOLVE_TYPE = {
    "fast_phase": "scalarphase",
    "medium_phase": "scalarphase",
    "slow_gains": "diagonal",
    "full_jones": "fulljones",
}
SOLUTION_LABELS_BY_SOLVE_TYPE = {
    "fast_phase": {"fast"},
    "medium_phase": {"medium1", "medium2"},
    "slow_gains": {"slow"},
    "full_jones": {"fulljones"},
}


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


def _solve_type_from_slot(input_parms: Mapping[str, object], slot: int) -> str:
    explicit_type = input_parms.get(f"solve{slot}_type")
    if explicit_type is None:
        return "unsupported"

    solve_type = str(explicit_type)
    if solve_type not in MODE_BY_SOLVE_TYPE:
        return "unsupported"
    if str(input_parms.get(f"solve{slot}_mode")) != MODE_BY_SOLVE_TYPE[solve_type]:
        return "unsupported"

    solution_label = input_parms.get(f"solve{slot}_solution_label")
    if (
        solution_label is None
        or str(solution_label) not in SOLUTION_LABELS_BY_SOLVE_TYPE[solve_type]
    ):
        return "unsupported"
    return solve_type


def _solve_label_from_slot(input_parms: Mapping[str, object], slot: int) -> str:
    label = input_parms.get(f"solve{slot}_solution_label")
    if label is None:
        raise ValueError(f"solve{slot}_solution_label is required")
    return str(label)


def _medium_index_from_slot(input_parms: Mapping[str, object], slot: int) -> Optional[int]:
    medium_index = input_parms.get(f"solve{slot}_medium_index")
    if medium_index is None:
        return None
    return int(medium_index)


def _phase_combination_key(phase_index: int) -> str:
    """Return the payload key for combining phase solves up to this 1-based index."""
    return "phase_" + "_".join(str(index) for index in range(1, phase_index + 1))


def _phase_combination_input_key(phase_index: int) -> str:
    if phase_index == 2:
        return "combined_phase_1_2_h5parm"
    if phase_index == 3:
        return "combined_phase_1_2_3_h5parm"
    raise ValueError(f"Unsupported phase-combination index: {phase_index}")


def _keeps_model_for_phase_slow(slot: int, first_solve_slot: int, solve_type: str) -> bool:
    """Return whether this solve should retain the predicted model for later solves."""
    if slot == first_solve_slot:
        return False
    return solve_type in {"medium_phase", "slow_gains"}


def _active_solve_types(input_parms: Mapping[str, object]) -> list[str]:
    return [
        _solve_type_from_slot(input_parms, int(step.removeprefix("solve")))
        for step in _active_solve_steps(input_parms)
    ]


def _unsupported_solve_slot_message(
    mode: str,
    active_solves: list[str],
    solve_types: list[str],
    input_parms: Mapping[str, object],
) -> str:
    unsupported_slots = []
    for step, solve_type in zip(active_solves, solve_types):
        if solve_type != "unsupported":
            continue
        slot = int(step.removeprefix("solve"))
        unsupported_slots.append(
            f"{step} type={input_parms.get(f'solve{slot}_type')!r} "
            f"mode={input_parms.get(f'solve{slot}_mode')!r} "
            f"output={_first_solve_output(input_parms, slot)!r}"
        )

    return f"Unsupported {mode.upper()} calibration solve slot(s): " + "; ".join(unsupported_slots)


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
    if input_parms.get("has_slow_gain_solve"):
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
    if not active_solves:
        raise ValueError("Calibration requires at least one solve step")

    solve_types = _active_solve_types(input_parms)
    if "unsupported" in solve_types:
        raise ValueError(
            _unsupported_solve_slot_message(mode, active_solves, solve_types, input_parms)
        )
    if solve_types.count("slow_gains") > 1:
        raise ValueError("A calibration cycle can contain at most one slow_gains solve")
    if "slow_gains" in solve_types:
        _validate_slow_gain_processing_inputs(input_parms)

    if mode == "di":
        if solve_types == ["full_jones"]:
            return "di_fulljones"
        if solve_types == ["fast_phase"]:
            return "di_fast_phase"
        if solve_types == ["slow_gains"]:
            return "di_slow"
        if solve_types == ["fast_phase", "medium_phase"]:
            return "di_scalar_phase"
        if solve_types == ["fast_phase", "medium_phase", "slow_gains"]:
            return "di_phase_slow"
        return "di_calibration"

    if mode == "dd":
        if solve_types == ["fast_phase"]:
            return "dd_fast_phase"
        if solve_types == ["slow_gains"]:
            return "dd_slow"
        if solve_types == ["fast_phase", "medium_phase"]:
            return "dd_phase"
        if solve_types in (
            ["fast_phase", "medium_phase", "slow_gains"],
            ["fast_phase", "medium_phase", "slow_gains", "medium_phase"],
        ):
            return "dd_phase_slow"
        return "dd_calibration"

    raise ValueError("Only DI and DD calibration payloads are supported")


def _solve_slots_for_kind(calibration_kind: str, input_parms: Mapping[str, object]) -> list[int]:
    if calibration_kind == "dd_screen":
        return []
    return [int(step.removeprefix("solve")) for step in _active_solve_steps(input_parms)]


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
    h5parm = validate_basename(
        _scatter_value(
            input_parms[f"output_solve{slot}_h5parm"], index, f"output_solve{slot}_h5parm"
        ),
        f"output_solve{slot}_h5parm[{index}]",
    )
    slot_record = {
        "slot": slot,
        "solve_type": _solve_type_from_slot(input_parms, slot),
        "solution_label": _solve_label_from_slot(input_parms, slot),
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
    medium_index = _medium_index_from_slot(input_parms, slot)
    if medium_index is not None:
        slot_record["medium_index"] = medium_index
    if modeldatacolumns is not None:
        slot_record["modeldatacolumns"] = modeldatacolumns
    datause_key = f"solve{slot}_datause"
    if datause_key in input_parms:
        slot_record["datause"] = input_parms.get(datause_key)
    initial_h5parm_key = f"solve{slot}_initialsolutions_h5parm"
    if initial_h5parm_key in input_parms:
        slot_record["initialsolutions_h5parm"] = optional_file_path(
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

    model_root = validate_basename(input_parms.get("model_image_root"), "model_image_root")
    numterms = int(input_parms["num_spectral_terms"])
    if numterms < 1:
        raise ValueError("num_spectral_terms must be at least 1")

    region_filename = validate_basename(input_parms.get("facet_region_file"), "facet_region_file")
    return {
        "skymodel": optional_file_path(
            input_parms.get("calibration_skymodel_file"), "calibration_skymodel_file"
        ),
        "model_image_root": model_root,
        "model_image_ra_dec": [
            str(value)
            for value in validate_required_list(
                input_parms.get("model_image_ra_dec"), "model_image_ra_dec", length=2
            )
        ],
        "model_image_imsize": [
            int(value)
            for value in validate_required_list(
                input_parms.get("model_image_imsize"), "model_image_imsize", length=2
            )
        ],
        "model_image_cellsize": input_parms["model_image_cellsize"],
        "model_image_frequency_bandwidth": list(
            validate_required_list(
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
        has_slow_gain_solve = bool(input_parms.get("has_slow_gain_solve", False))
        solint_slow = input_parms.get("solint_slow_timestep", [])
        if has_slow_gain_solve:
            scatter_inputs.append(solint_slow)
        if not all(isinstance(value, list) for value in scatter_inputs):
            raise ValueError("Screen-generation scatter inputs must be lists")
        chunk_count = len(output_h5parms)
        if any(len(value) != chunk_count for value in scatter_inputs):
            raise ValueError("Screen-generation scatter inputs must have the same length")

        combined = validate_basename(input_parms.get("combined_h5parms"), "combined_h5parms")
        chunks: list[CalibrateChunkPayload] = []
        for index in range(chunk_count):
            h5parm = validate_basename(
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
            if has_slow_gain_solve:
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
            "has_slow_gain_solve": has_slow_gain_solve,
            "combined_h5parm": {
                "filename": combined,
                "path": os.path.join(pipeline_dir, combined),
            },
            "chunks": chunks,
        }
        assert_serializable_payload(payload)
        return payload

    solve_slots = _solve_slots_for_kind(calibration_kind, input_parms)
    solve_types = {slot: _solve_type_from_slot(input_parms, slot) for slot in solve_slots}
    phase_solve_count = sum(
        solve_type in {"fast_phase", "medium_phase"} for solve_type in solve_types.values()
    )
    slow_solve_count = sum(solve_type == "slow_gains" for solve_type in solve_types.values())
    first_solve_slot = solve_slots[0]
    first_output_key = f"output_solve{first_solve_slot}_h5parm"
    first_output_h5parms = input_parms.get(first_output_key, [])
    scatter_inputs = [filenames, starttimes, ntimes, first_output_h5parms]
    for slot in solve_slots:
        if slot != first_solve_slot:
            scatter_inputs.append(input_parms.get(f"output_solve{slot}_h5parm", []))
    if not all(isinstance(value, list) for value in scatter_inputs):
        raise ValueError("Calibration scatter inputs must be lists")
    chunk_count = len(first_output_h5parms)
    if any(len(value) != chunk_count for value in scatter_inputs):
        raise ValueError("Calibration scatter inputs must have the same length")

    collected_h5parms = {}
    for slot in solve_slots:
        collected = validate_basename(
            input_parms.get(f"collected_solve{slot}_h5parm"),
            f"collected_solve{slot}_h5parm",
        )
        collected_h5parms[f"solve{slot}"] = {
            "filename": collected,
            "path": os.path.join(pipeline_dir, collected),
        }
    combined_h5parm = None
    if calibration_kind == "di_scalar_phase":
        combined = validate_basename(
            input_parms.get(_phase_combination_input_key(2)),
            _phase_combination_input_key(2),
        )
        combined_h5parm = {"filename": combined, "path": os.path.join(pipeline_dir, combined)}

    combined_h5parms = {}
    for phase_index in range(2, phase_solve_count + 1):
        input_key = _phase_combination_input_key(phase_index)
        combined = validate_basename(
            input_parms.get(input_key),
            input_key,
        )
        combined_h5parms[_phase_combination_key(phase_index)] = {
            "filename": combined,
            "path": os.path.join(pipeline_dir, combined),
        }
    if (
        (phase_solve_count >= 1 and slow_solve_count > 0)
        or phase_solve_count >= 4
        or calibration_kind in {"dd_phase_slow", "di_phase_slow"}
    ):
        combined = validate_basename(
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
            solve_type = solve_types[slot]
            keepmodel = None
            reusemodel = None
            slot_modeldatacolumns = None
            if slot == first_solve_slot:
                keepmodel = "True"
                if image_based_predict:
                    reusemodel = "[predict.*]"
            else:
                if image_based_predict:
                    reusemodel = "[predict.*]"
                elif uses_modeldatacolumn and not (mode == "di" and slow_solve_count == 0):
                    slot_modeldatacolumns = modeldatacolumn
                else:
                    reusemodel = f"[solve{first_solve_slot}.*]"
            if calibration_kind == "dd_phase_slow" and _keeps_model_for_phase_slow(
                slot,
                first_solve_slot,
                solve_type,
            ):
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
            "applycal_h5parm": optional_file_path(
                input_parms.get("applycal_h5parm"), "applycal_h5parm"
            ),
            "fulljones_h5parm": optional_file_path(
                input_parms.get("fulljones_h5parm"), "fulljones_h5parm"
            ),
            "normalize_h5parm": optional_file_path(
                input_parms.get("normalize_h5parm"), "normalize_h5parm"
            ),
            "bda_timebase": input_parms.get("bda_timebase"),
            "bda_frequencybase": input_parms.get("bda_frequencybase"),
            "onebeamperpatch": input_parms.get("onebeamperpatch"),
            "parallelbaselines": input_parms.get("parallelbaselines"),
            "sagecalpredict": input_parms.get("sagecalpredict"),
            "sourcedb": optional_file_path(
                input_parms.get("calibration_skymodel_file"), "calibration_skymodel_file"
            ),
            "directions": None
            if input_parms.get("solve_directions") is None
            else [str(direction) for direction in input_parms["solve_directions"]],
            "has_slow_gain_solve": any(
                solve_type == "slow_gains" for solve_type in solve_types.values()
            ),
            "solution_combine_mode": input_parms.get("solution_combine_mode"),
            "max_normalization_delta": input_parms.get("max_normalization_delta"),
            "scale_normalization_delta": input_parms.get("scale_normalization_delta"),
            "phase_center_ra": input_parms.get("phase_center_ra"),
            "phase_center_dec": input_parms.get("phase_center_dec"),
        }
    )
    assert_serializable_payload(payload)
    return payload

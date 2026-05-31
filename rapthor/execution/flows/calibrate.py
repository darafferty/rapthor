"""Prefect flows for the Calibrate operation."""

import glob
import os
from typing import Mapping, Optional

from prefect import flow, task

from rapthor.execution.commands import normalize_command
from rapthor.execution.config import ExecutionConfig
from rapthor.execution.flows.runtime import run_flow_with_task_runner
from rapthor.execution.outputs import file_record, validate_output_record
from rapthor.execution.payloads import assert_serializable_payload
from rapthor.execution.shell import ShellCommand, run_shell_command

DDECAL_SOLVE_ARGUMENTS = [
    "msout=",
    "applybeam.type=applybeam",
    "applybeam.beammode=array_factor",
    "applybeam.usemodeldata=True",
    "applybeam.invert=False",
    "applycal.type=applycal",
    "applycal.correction=phase000",
    "applycal.fastphase.correction=phase000",
    "applycal.fastphase.solset=sol000",
    "applycal.slowgain.correction=amplitude000",
    "applycal.slowgain.solset=sol000",
    "applycal.fulljones.correction=fulljones",
    "applycal.fulljones.solset=sol000",
    "applycal.fulljones.soltab=[amplitude000,phase000]",
    "applycal.normalization.correction=amplitude000",
    "applycal.normalization.solset=sol000",
    "applycal.normalization.usemodeldata=True",
    "applycal.normalization.invert=False",
    "avg.type=bdaaverager",
    "predict.type=wgridderpredict",
    "solve1.type=ddecal",
    "solve1.usebeammodel=True",
    "solve1.beam_interval=120",
    "solve1.beammode=array_factor",
    "solve1.initialsolutions.missingantennabehavior=unit",
    "solve1.applycal.normalization.correction=amplitude000",
    "solve1.applycal.normalization.solset=sol000",
    "solve2.type=ddecal",
    "solve2.initialsolutions.missingantennabehavior=unit",
    "solve2.applycal.normalization.correction=amplitude000",
    "solve2.applycal.normalization.solset=sol000",
    "solve3.type=ddecal",
    "solve3.initialsolutions.missingantennabehavior=unit",
    "solve3.applycal.normalization.correction=amplitude000",
    "solve3.applycal.normalization.solset=sol000",
    "solve4.type=ddecal",
    "solve4.initialsolutions.missingantennabehavior=unit",
    "solve4.applycal.normalization.correction=amplitude000",
    "solve4.applycal.normalization.solset=sol000",
]

SOLVE_SLOT_ARGUMENTS = [
    ("h5parm", "h5parm"),
    ("solint", "solint"),
    ("mode", "mode"),
    ("nchan", "nchan"),
    ("solutions_per_direction", "solutions_per_direction"),
    ("llssolver", "llssolver"),
    ("maxiter", "maxiter"),
    ("propagatesolutions", "propagatesolutions"),
    ("initialsolutions_h5parm", "initialsolutions.h5parm"),
    ("initialsolutions_soltab", "initialsolutions.soltab"),
    ("solveralgorithm", "solveralgorithm"),
    ("solverlbfgs_dof", "solverlbfgs.dof"),
    ("solverlbfgs_iter", "solverlbfgs.iter"),
    ("solverlbfgs_minibatches", "solverlbfgs.minibatches"),
    ("datause", "datause"),
    ("stepsize", "stepsize"),
    ("stepsigma", "stepsigma"),
    ("tolerance", "tolerance"),
    ("uvlambdamin", "uvlambdamin"),
    ("smoothness_dd_factors", "smoothness_dd_factors"),
    ("smoothnessconstraint", "smoothnessconstraint"),
    ("smoothnessreffrequency", "smoothnessreffrequency"),
    ("smoothnessrefdistance", "smoothnessrefdistance"),
    ("antennaconstraint", "antennaconstraint"),
    ("correctfreqsmearing", "correctfreqsmearing"),
    ("correcttimesmearing", "correcttimesmearing"),
    ("keepmodel", "keepmodel"),
    ("reusemodel", "reusemodel"),
    ("normalize_h5parm", "applycal.normalization.parmdb"),
    ("applycal_steps", "applycal.steps"),
]


def _bool_token(value: bool) -> str:
    return "True" if value else "False"


def _list_token(values: list[object]) -> str:
    return f"[{','.join(str(value) for value in values)}]"


def _record_path(record: object, path_class: str) -> str:
    if isinstance(record, Mapping) and record.get("class") == path_class:
        path = record.get("path")
        if isinstance(path, str) and path:
            return path
    raise ValueError(f"Expected a {path_class} output record, got {record!r}")


def _path_record_path(record: object) -> str:
    return _record_path(record, "Directory")


def _validate_basename(filename: object, name: str) -> str:
    if not isinstance(filename, str) or not filename:
        raise ValueError(f"{name} must be a non-empty string")
    if os.path.isabs(filename) or os.path.basename(filename) != filename:
        raise ValueError(f"{name} must be a basename")
    return filename


def _require_file(path: str, label: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} was not created: {path}")
    return file_record(path)


def _scatter_value(value: object, index: int, name: str) -> object:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    if index >= len(value):
        raise ValueError(f"{name} must have at least {index + 1} entries")
    return value[index]


def _optional_scatter_value(input_parms: Mapping[str, object], name: str, index: int) -> object:
    if name not in input_parms:
        return None
    return _scatter_value(input_parms[name], index, name)


def _append_option(command: list[str], prefix: str, value: object) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        value = _bool_token(value)
    elif isinstance(value, list):
        if any(item is None for item in value):
            return
        value = _list_token(value)
    command.append(f"{prefix}={value}")


def build_ddecal_solve_command(
    msin: str,
    data_colname: str,
    starttime: str,
    ntimes: int,
    steps: str,
    solve_slots: list[Mapping[str, object]],
    numthreads: int,
    modeldatacolumn: Optional[str] = None,
    applycal_steps: Optional[str] = None,
    applycal_h5parm: Optional[str] = None,
    fulljones_h5parm: Optional[str] = None,
    normalize_h5parm: Optional[str] = None,
    timebase: Optional[object] = None,
    maxinterval: Optional[object] = None,
    frequencybase: Optional[object] = None,
    minchannels: Optional[object] = None,
    onebeamperpatch: Optional[bool] = None,
    parallelbaselines: Optional[bool] = None,
    sagecalpredict: Optional[bool] = None,
    sourcedb: Optional[str] = None,
    directions: Optional[list[str]] = None,
    predict_regions: Optional[str] = None,
    predict_images: Optional[list[str]] = None,
) -> list[str]:
    """Build the DP3 DDECal solve command for one calibration chunk."""
    command = ["DP3", *DDECAL_SOLVE_ARGUMENTS]
    common_options = [
        ("msin", msin),
        ("msin.datacolumn", data_colname),
        ("msin.starttime", starttime),
        ("msin.ntimes", ntimes),
        ("steps", steps),
        ("applycal.steps", applycal_steps),
        ("applycal.parmdb", applycal_h5parm),
        ("applycal.fulljones.parmdb", fulljones_h5parm),
        ("applycal.normalization.parmdb", normalize_h5parm),
        ("avg.timebase", timebase),
        ("avg.maxinterval", maxinterval),
        ("avg.frequencybase", frequencybase),
        ("avg.minchannels", minchannels),
        ("solve1.modeldatacolumns", modeldatacolumn),
        ("solve1.onebeamperpatch", onebeamperpatch),
        ("solve1.parallelbaselines", parallelbaselines),
        ("solve1.sagecalpredict", sagecalpredict),
        ("predict.regions", predict_regions),
        (
            "predict.images",
            None
            if predict_images is None
            else f"[{','.join(str(path) for path in predict_images)}]",
        ),
        ("solve1.sourcedb", sourcedb),
        ("solve1.directions", directions),
    ]
    for prefix, value in common_options:
        _append_option(command, prefix, value)

    for slot in solve_slots:
        slot_index = int(slot["slot"])
        for key, suffix in SOLVE_SLOT_ARGUMENTS:
            _append_option(command, f"solve{slot_index}.{suffix}", slot.get(key))

    _append_option(command, "numthreads", numthreads)
    return command


def build_collect_h5parms_command(inh5parms: list[str], outputh5parm: str) -> list[str]:
    """Build the h5parm collection command."""
    return [
        "H5parm_collector.py",
        "-c",
        ",".join(inh5parms),
        f"--outh5parm={outputh5parm}",
    ]


def build_combine_h5parms_command(
    inh5parm1: str,
    inh5parm2: str,
    outh5parm: str,
    mode: str,
    reweight: bool,
    calibrator_names: Optional[list[str]] = None,
    calibrator_fluxes: Optional[list[float]] = None,
) -> list[str]:
    """Build the h5parm combination command."""
    command = [
        "combine_h5parms.py",
        inh5parm1,
        inh5parm2,
        outh5parm,
        mode,
        f"--reweight={_bool_token(reweight)}",
    ]
    if calibrator_names is not None:
        command.append(f"--cal_names={','.join(str(name) for name in calibrator_names)}")
    if calibrator_fluxes is not None:
        command.append(f"--cal_fluxes={','.join(str(flux) for flux in calibrator_fluxes)}")
    return command


def build_plot_solutions_command(
    h5parm: str,
    soltype: str,
    root: Optional[str] = None,
) -> list[str]:
    """Build the solution plotting command."""
    command = ["plotrapthor", h5parm, soltype]
    if root is not None:
        command.append(f"--root={root}")
    return command


def _parse_steps(steps: object) -> list[str]:
    return [step.strip() for step in str(steps).strip("[]").split(",") if step.strip()]


def _supported_di_kind(input_parms: Mapping[str, object]) -> str:
    steps = _parse_steps(input_parms.get("dp3_steps"))
    solve1_mode = str(input_parms.get("solve1_mode"))
    solve2_mode = str(input_parms.get("solve2_mode"))
    if steps == ["solve1"] and solve1_mode == "fulljones":
        return "di_fulljones"
    if steps == ["solve1", "solve2"] and solve1_mode == solve2_mode == "scalarphase":
        return "di_scalar_phase"
    raise ValueError("Only DI full-Jones and DI scalar phase calibration are supported")


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
) -> dict:
    h5parm = _validate_basename(
        _scatter_value(
            input_parms[f"output_solve{slot}_h5parm"], index, f"output_solve{slot}_h5parm"
        ),
        f"output_solve{slot}_h5parm[{index}]",
    )
    return {
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


def calibrate_payload_from_inputs(
    mode: str,
    input_parms: Mapping[str, object],
    pipeline_working_dir: object,
) -> dict:
    """Create a serializable Calibrate flow payload from operation inputs."""
    if mode != "di":
        raise ValueError("Only DI calibration payloads are supported in this slice")
    calibration_kind = _supported_di_kind(input_parms)

    pipeline_dir = str(pipeline_working_dir)
    filenames = input_parms.get("timechunk_filename", [])
    starttimes = input_parms.get("starttime", [])
    ntimes = input_parms.get("ntimes", [])
    solve_slots = [1] if calibration_kind == "di_fulljones" else [1, 2]
    output_solve1_h5parms = input_parms.get("output_solve1_h5parm", [])
    scatter_inputs = [filenames, starttimes, ntimes, output_solve1_h5parms]
    if 2 in solve_slots:
        scatter_inputs.append(input_parms.get("output_solve2_h5parm", []))
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

    modeldatacolumn = input_parms.get("modeldatacolumn")
    if not isinstance(modeldatacolumn, str) or not modeldatacolumn:
        raise ValueError("modeldatacolumn must be a non-empty string")

    chunks = []
    for index in range(chunk_count):
        chunk_solve_slots = [
            _solve_slot_from_inputs(
                input_parms,
                pipeline_dir,
                slot=1,
                index=index,
                keepmodel="True",
            )
        ]
        if 2 in solve_slots:
            chunk_solve_slots.append(
                _solve_slot_from_inputs(
                    input_parms,
                    pipeline_dir,
                    slot=2,
                    index=index,
                    reusemodel="[solve1.*]",
                )
            )
        chunks.append(
            {
                "msin": _path_record_path(filenames[index]),
                "starttime": str(starttimes[index]),
                "ntimes": int(ntimes[index]),
                "output_h5parm": chunk_solve_slots[0]["h5parm"],
                "output_h5parm_path": chunk_solve_slots[0]["h5parm_path"],
                "solve1_solint": chunk_solve_slots[0]["solint"],
                "solve1_nchan": chunk_solve_slots[0]["nchan"],
                "solve_slots": chunk_solve_slots,
            }
        )

    payload = {
        "mode": mode,
        "calibration_kind": calibration_kind,
        "pipeline_working_dir": pipeline_dir,
        "data_colname": str(input_parms["data_colname"]),
        "modeldatacolumn": modeldatacolumn,
        "dp3_steps": str(input_parms["dp3_steps"]),
        **_solver_payload_from_inputs(input_parms),
        "smoothnessconstraint_fulljones": float(input_parms["smoothnessconstraint_fulljones"]),
        "collected_h5parm": collected_h5parms["solve1"]["filename"],
        "collected_h5parm_path": collected_h5parms["solve1"]["path"],
        "collected_h5parms": collected_h5parms,
        "combined_h5parm": combined_h5parm,
        "calibrator_patch_names": [
            str(name) for name in input_parms.get("calibrator_patch_names", [])
        ],
        "calibrator_fluxes": [float(flux) for flux in input_parms.get("calibrator_fluxes", [])],
        "chunks": chunks,
    }
    return assert_serializable_payload(payload)


def _di_fulljones_solve_slots(
    payload: Mapping[str, object],
    chunk: Mapping[str, object],
) -> list[Mapping[str, object]]:
    return [
        {
            "slot": 1,
            "h5parm": chunk["output_h5parm"],
            "solint": chunk["solve1_solint"],
            "mode": "fulljones",
            "nchan": chunk["solve1_nchan"],
            "llssolver": payload["llssolver"],
            "maxiter": payload["maxiter"],
            "propagatesolutions": payload["propagatesolutions"],
            "initialsolutions_soltab": "[phase000]",
            "solveralgorithm": payload["solveralgorithm"],
            "solverlbfgs_dof": payload["solverlbfgs_dof"],
            "solverlbfgs_iter": payload["solverlbfgs_iter"],
            "solverlbfgs_minibatches": payload["solverlbfgs_minibatches"],
            "stepsize": payload["stepsize"],
            "stepsigma": payload["stepsigma"],
            "tolerance": payload["tolerance"],
            "uvlambdamin": payload["uvlambdamin"],
            "smoothnessconstraint": payload["smoothnessconstraint_fulljones"],
            "antennaconstraint": "[]",
            "correctfreqsmearing": payload["correctfreqsmearing"],
            "correcttimesmearing": payload["correcttimesmearing"],
            "keepmodel": "True",
        }
    ]


def _solve_slots_for_chunk(
    payload: Mapping[str, object],
    chunk: Mapping[str, object],
) -> list[Mapping[str, object]]:
    if payload["calibration_kind"] == "di_fulljones":
        return _di_fulljones_solve_slots(payload, chunk)

    solve_slots = []
    for slot in chunk["solve_slots"]:
        solve_slots.append(
            {
                "slot": slot["slot"],
                "h5parm": slot["h5parm"],
                "solint": slot["solint"],
                "mode": slot["mode"],
                "nchan": slot["nchan"],
                "solutions_per_direction": slot["solutions_per_direction"],
                "llssolver": payload["llssolver"],
                "maxiter": payload["maxiter"],
                "propagatesolutions": payload["propagatesolutions"],
                "initialsolutions_soltab": "[phase000]",
                "solveralgorithm": payload["solveralgorithm"],
                "solverlbfgs_dof": payload["solverlbfgs_dof"],
                "solverlbfgs_iter": payload["solverlbfgs_iter"],
                "solverlbfgs_minibatches": payload["solverlbfgs_minibatches"],
                "stepsize": payload["stepsize"],
                "stepsigma": payload["stepsigma"],
                "tolerance": payload["tolerance"],
                "uvlambdamin": payload["uvlambdamin"],
                "smoothness_dd_factors": slot["smoothness_dd_factors"],
                "smoothnessconstraint": slot["smoothnessconstraint"],
                "smoothnessreffrequency": slot["smoothnessreffrequency"],
                "smoothnessrefdistance": slot["smoothnessrefdistance"],
                "antennaconstraint": slot["antennaconstraint"],
                "keepmodel": slot["keepmodel"],
                "reusemodel": slot["reusemodel"],
            }
        )
    solve_slots[0]["correctfreqsmearing"] = payload["correctfreqsmearing"]
    solve_slots[0]["correcttimesmearing"] = payload["correcttimesmearing"]
    return solve_slots


def _run_shell(
    command: list[str],
    pipeline_working_dir: str,
    execution_config: ExecutionConfig,
    shell_operation_cls=None,
) -> None:
    run_shell_command(
        ShellCommand(command=command, working_directory=pipeline_working_dir),
        execution_config,
        shell_operation_cls=shell_operation_cls,
    )


def run_calibrate_chunk(
    payload: Mapping[str, object],
    chunk: Mapping[str, object],
    execution_config: Optional[ExecutionConfig] = None,
    shell_operation_cls=None,
) -> dict:
    """Run one DI calibration chunk."""
    config = execution_config or ExecutionConfig(task_runner="sync")
    pipeline_working_dir = str(payload["pipeline_working_dir"])
    command = build_ddecal_solve_command(
        msin=str(chunk["msin"]),
        data_colname=str(payload["data_colname"]),
        starttime=str(chunk["starttime"]),
        ntimes=int(chunk["ntimes"]),
        steps=str(payload["dp3_steps"]),
        solve_slots=_solve_slots_for_chunk(payload, chunk),
        numthreads=int(payload["max_threads"]),
        modeldatacolumn=str(payload["modeldatacolumn"]),
    )
    _run_shell(command, pipeline_working_dir, config, shell_operation_cls=shell_operation_cls)
    output_records = {}
    for slot in chunk["solve_slots"]:
        label = (
            "DI full-Jones h5parm"
            if payload["calibration_kind"] == "di_fulljones"
            else f"DI solve{slot['slot']} h5parm"
        )
        output_records[f"solve{slot['slot']}"] = _require_file(str(slot["h5parm_path"]), label)
    if payload["calibration_kind"] == "di_fulljones":
        return output_records["solve1"]
    return output_records


@task(name="calibrate_chunk")
def calibrate_chunk_task(
    payload: Mapping[str, object],
    chunk: Mapping[str, object],
    execution_config: Optional[ExecutionConfig] = None,
    shell_operation_cls=None,
) -> dict:
    """Prefect task wrapper for one calibration chunk."""
    return run_calibrate_chunk(
        payload,
        chunk,
        execution_config=execution_config,
        shell_operation_cls=shell_operation_cls,
    )


def _collect_and_plot_fulljones(
    payload: Mapping[str, object],
    solve_records: list[dict],
    execution_config: ExecutionConfig,
    shell_operation_cls=None,
) -> dict:
    pipeline_working_dir = str(payload["pipeline_working_dir"])
    collected_h5parm = str(payload["collected_h5parm"])
    collect_command = build_collect_h5parms_command(
        [record["path"] for record in solve_records],
        collected_h5parm,
    )
    _run_shell(
        collect_command,
        pipeline_working_dir,
        execution_config,
        shell_operation_cls=shell_operation_cls,
    )
    collected_record = _require_file(
        str(payload["collected_h5parm_path"]), "Collected DI full-Jones h5parm"
    )

    before_plots = set(glob.glob(os.path.join(pipeline_working_dir, "*.png")))
    plot_command = build_plot_solutions_command(collected_record["path"], "phase")
    _run_shell(
        plot_command,
        pipeline_working_dir,
        execution_config,
        shell_operation_cls=shell_operation_cls,
    )
    after_plots = set(glob.glob(os.path.join(pipeline_working_dir, "*.png")))
    plot_records = [file_record(path) for path in sorted(after_plots - before_plots)]
    if not plot_records:
        plot_records = [file_record(path) for path in sorted(after_plots)]

    result = {
        "combined_solutions": collected_record,
        "fast_phase_solutions": collected_record,
        "fast_phase_plots": plot_records,
    }
    for value in result.values():
        validate_output_record(value)
    return result


def _run_collect_h5parm(
    input_records: list[dict],
    output: Mapping[str, object],
    pipeline_working_dir: str,
    execution_config: ExecutionConfig,
    label: str,
    shell_operation_cls=None,
) -> dict:
    collect_command = build_collect_h5parms_command(
        [record["path"] for record in input_records],
        str(output["filename"]),
    )
    _run_shell(
        collect_command,
        pipeline_working_dir,
        execution_config,
        shell_operation_cls=shell_operation_cls,
    )
    return _require_file(str(output["path"]), label)


def _run_plot_solutions(
    h5parm_record: Mapping[str, str],
    soltype: str,
    pipeline_working_dir: str,
    execution_config: ExecutionConfig,
    root: Optional[str] = None,
    shell_operation_cls=None,
) -> list[dict]:
    before_plots = set(glob.glob(os.path.join(pipeline_working_dir, "*.png")))
    plot_command = build_plot_solutions_command(h5parm_record["path"], soltype, root=root)
    _run_shell(
        plot_command,
        pipeline_working_dir,
        execution_config,
        shell_operation_cls=shell_operation_cls,
    )
    after_plots = set(glob.glob(os.path.join(pipeline_working_dir, "*.png")))
    plot_records = [file_record(path) for path in sorted(after_plots - before_plots)]
    if not plot_records:
        plot_records = [file_record(path) for path in sorted(after_plots)]
    return plot_records


def _collect_plot_and_combine_scalar_phase(
    payload: Mapping[str, object],
    solve_records: list[dict],
    execution_config: ExecutionConfig,
    shell_operation_cls=None,
) -> dict:
    pipeline_working_dir = str(payload["pipeline_working_dir"])
    collected_h5parms = payload["collected_h5parms"]
    fast_record = _run_collect_h5parm(
        [record["solve1"] for record in solve_records],
        collected_h5parms["solve1"],
        pipeline_working_dir,
        execution_config,
        "Collected DI fast phase h5parm",
        shell_operation_cls=shell_operation_cls,
    )
    fast_plots = _run_plot_solutions(
        fast_record,
        "phase",
        pipeline_working_dir,
        execution_config,
        shell_operation_cls=shell_operation_cls,
    )

    medium_record = _run_collect_h5parm(
        [record["solve2"] for record in solve_records],
        collected_h5parms["solve2"],
        pipeline_working_dir,
        execution_config,
        "Collected DI medium phase h5parm",
        shell_operation_cls=shell_operation_cls,
    )
    medium_plots = _run_plot_solutions(
        medium_record,
        "phase",
        pipeline_working_dir,
        execution_config,
        root="medium1_phase_",
        shell_operation_cls=shell_operation_cls,
    )

    combined_h5parm = payload["combined_h5parm"]
    combine_command = build_combine_h5parms_command(
        fast_record["path"],
        medium_record["path"],
        str(combined_h5parm["filename"]),
        "p1p2_scalar",
        reweight=False,
        calibrator_names=list(payload["calibrator_patch_names"]),
        calibrator_fluxes=list(payload["calibrator_fluxes"]),
    )
    _run_shell(
        combine_command,
        pipeline_working_dir,
        execution_config,
        shell_operation_cls=shell_operation_cls,
    )
    combined_record = _require_file(str(combined_h5parm["path"]), "Combined DI scalar h5parm")

    result = {
        "combined_solutions": combined_record,
        "fast_phase_solutions": fast_record,
        "medium1_phase_solutions": medium_record,
        "fast_phase_plots": fast_plots,
        "medium1_phase_plots": medium_plots,
    }
    for value in result.values():
        validate_output_record(value)
    return result


def _collect_plot_and_combine(
    payload: Mapping[str, object],
    solve_records: list[dict],
    execution_config: ExecutionConfig,
    shell_operation_cls=None,
) -> dict:
    if payload["calibration_kind"] == "di_fulljones":
        return _collect_and_plot_fulljones(
            payload,
            solve_records,
            execution_config,
            shell_operation_cls=shell_operation_cls,
        )
    return _collect_plot_and_combine_scalar_phase(
        payload,
        solve_records,
        execution_config,
        shell_operation_cls=shell_operation_cls,
    )


def run_calibrate_flow(
    payload: Mapping[str, object],
    execution_config: Optional[ExecutionConfig] = None,
    shell_operation_cls=None,
) -> dict:
    """Run calibration commands and return finalizer-compatible outputs."""
    assert_serializable_payload(payload)
    config = execution_config or ExecutionConfig(task_runner="sync")
    solve_records = []
    for chunk in payload["chunks"]:
        solve_records.append(
            run_calibrate_chunk(
                payload,
                chunk,
                execution_config=config,
                shell_operation_cls=shell_operation_cls,
            )
        )
    return _collect_plot_and_combine(
        payload,
        solve_records,
        config,
        shell_operation_cls=shell_operation_cls,
    )


def _run_calibrate_prefect_tasks(
    payload: Mapping[str, object],
    execution_config: Optional[ExecutionConfig] = None,
) -> dict:
    config = execution_config or ExecutionConfig(task_runner="sync")
    solve_records = [
        calibrate_chunk_task.submit(payload, chunk, execution_config=config)
        for chunk in payload["chunks"]
    ]
    solve_records = [record.result() for record in solve_records]
    return _collect_plot_and_combine(payload, solve_records, config)


@flow(name="calibrate")
def _calibrate_flow(
    payload: Mapping[str, object],
    execution_config: Optional[ExecutionConfig] = None,
):
    """Prefect implementation for Calibrate."""
    return _run_calibrate_prefect_tasks(
        payload,
        execution_config=execution_config,
    )


def calibrate_flow(
    payload: Mapping[str, object],
    execution_config: Optional[ExecutionConfig] = None,
):
    """Prefect entry point for Calibrate."""
    return run_flow_with_task_runner(
        _calibrate_flow,
        payload,
        execution_config=execution_config,
    )


def normalized_ddecal_solve_command(**kwargs) -> list[str]:
    """Return normalized DDECal command tokens for fixture comparisons."""
    return normalize_command(build_ddecal_solve_command(**kwargs))


def normalized_collect_h5parms_command(
    inh5parms: list[str],
    outputh5parm: str,
) -> list[str]:
    """Return normalized h5parm collection command tokens."""
    return normalize_command(build_collect_h5parms_command(inh5parms, outputh5parm))


def normalized_combine_h5parms_command(
    inh5parm1: str,
    inh5parm2: str,
    outh5parm: str,
    mode: str,
    reweight: bool,
    calibrator_names: Optional[list[str]] = None,
    calibrator_fluxes: Optional[list[float]] = None,
) -> list[str]:
    """Return normalized h5parm combination command tokens."""
    return normalize_command(
        build_combine_h5parms_command(
            inh5parm1,
            inh5parm2,
            outh5parm,
            mode,
            reweight,
            calibrator_names=calibrator_names,
            calibrator_fluxes=calibrator_fluxes,
        )
    )


def normalized_plot_solutions_command(
    h5parm: str,
    soltype: str,
    root: Optional[str] = None,
) -> list[str]:
    """Return normalized solution plotting command tokens."""
    return normalize_command(build_plot_solutions_command(h5parm, soltype, root=root))

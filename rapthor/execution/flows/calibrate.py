"""Prefect flows for the Calibrate operation."""

import glob
import os
from typing import Mapping, Optional

from prefect import flow, task

from rapthor.execution.commands import normalize_command
from rapthor.execution.config import ExecutionConfig
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


def calibrate_payload_from_inputs(
    mode: str,
    input_parms: Mapping[str, object],
    pipeline_working_dir: object,
) -> dict:
    """Create a serializable Calibrate flow payload from operation inputs."""
    if mode != "di":
        raise ValueError("Only DI calibration payloads are supported in this slice")
    if str(input_parms.get("solve1_mode")) != "fulljones":
        raise ValueError("Only DI full-Jones calibration is supported in this slice")
    if str(input_parms.get("dp3_steps")) != "[solve1]":
        raise ValueError("Only single-solve DI full-Jones calibration is supported")

    pipeline_dir = str(pipeline_working_dir)
    filenames = input_parms.get("timechunk_filename", [])
    starttimes = input_parms.get("starttime", [])
    ntimes = input_parms.get("ntimes", [])
    output_solve1_h5parms = input_parms.get("output_solve1_h5parm", [])
    scatter_inputs = [filenames, starttimes, ntimes, output_solve1_h5parms]
    if not all(isinstance(value, list) for value in scatter_inputs):
        raise ValueError("Calibration scatter inputs must be lists")
    chunk_count = len(output_solve1_h5parms)
    if any(len(value) != chunk_count for value in scatter_inputs):
        raise ValueError("Calibration scatter inputs must have the same length")

    collected_h5parm = _validate_basename(
        input_parms.get("collected_solve1_h5parm"), "collected_solve1_h5parm"
    )
    modeldatacolumn = input_parms.get("modeldatacolumn")
    if not isinstance(modeldatacolumn, str) or not modeldatacolumn:
        raise ValueError("modeldatacolumn must be a non-empty string")

    chunks = []
    for index in range(chunk_count):
        output_h5parm = _validate_basename(
            output_solve1_h5parms[index], f"output_solve1_h5parm[{index}]"
        )
        chunks.append(
            {
                "msin": _path_record_path(filenames[index]),
                "starttime": str(starttimes[index]),
                "ntimes": int(ntimes[index]),
                "output_h5parm": output_h5parm,
                "output_h5parm_path": os.path.join(pipeline_dir, output_h5parm),
                "solve1_solint": int(
                    _scatter_value(
                        input_parms["solint_solve1_timestep"], index, "solint_solve1_timestep"
                    )
                ),
                "solve1_nchan": int(
                    _scatter_value(
                        input_parms["solint_solve1_freqstep"], index, "solint_solve1_freqstep"
                    )
                ),
            }
        )

    payload = {
        "mode": mode,
        "pipeline_working_dir": pipeline_dir,
        "data_colname": str(input_parms["data_colname"]),
        "modeldatacolumn": modeldatacolumn,
        "dp3_steps": str(input_parms["dp3_steps"]),
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
        "smoothnessconstraint_fulljones": float(input_parms["smoothnessconstraint_fulljones"]),
        "correctfreqsmearing": bool(input_parms["correctfreqsmearing"]),
        "correcttimesmearing": bool(input_parms["correcttimesmearing"]),
        "collected_h5parm": collected_h5parm,
        "collected_h5parm_path": os.path.join(pipeline_dir, collected_h5parm),
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
    """Run one DI full-Jones calibration chunk."""
    config = execution_config or ExecutionConfig(task_runner="sync")
    pipeline_working_dir = str(payload["pipeline_working_dir"])
    command = build_ddecal_solve_command(
        msin=str(chunk["msin"]),
        data_colname=str(payload["data_colname"]),
        starttime=str(chunk["starttime"]),
        ntimes=int(chunk["ntimes"]),
        steps=str(payload["dp3_steps"]),
        solve_slots=_di_fulljones_solve_slots(payload, chunk),
        numthreads=int(payload["max_threads"]),
        modeldatacolumn=str(payload["modeldatacolumn"]),
    )
    _run_shell(command, pipeline_working_dir, config, shell_operation_cls=shell_operation_cls)
    return _require_file(str(chunk["output_h5parm_path"]), "DI full-Jones h5parm")


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
    return _collect_and_plot_fulljones(
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
        calibrate_chunk_task(payload, chunk, execution_config=config) for chunk in payload["chunks"]
    ]
    return _collect_and_plot_fulljones(payload, solve_records, config)


@flow(name="calibrate")
def calibrate_flow(
    payload: Mapping[str, object],
    execution_config: Optional[ExecutionConfig] = None,
):
    """Prefect entry point for Calibrate."""
    return _run_calibrate_prefect_tasks(
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


def normalized_plot_solutions_command(
    h5parm: str,
    soltype: str,
    root: Optional[str] = None,
) -> list[str]:
    """Return normalized solution plotting command tokens."""
    return normalize_command(build_plot_solutions_command(h5parm, soltype, root=root))

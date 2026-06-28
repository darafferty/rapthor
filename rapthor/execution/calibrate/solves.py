"""Calibration solve chunk execution helpers."""

from typing import Mapping, Optional

from rapthor.execution.calibrate.commands import (
    DdecalSolveOptions,
    build_ddecal_solve_command,
    build_idgcal_solve_phase_and_gain_command,
    build_idgcal_solve_phase_command,
)
from rapthor.execution.calibrate.payloads import CalibrateChunkPayload, CalibratePayload
from rapthor.execution.config import ExecutionConfig
from rapthor.execution.outputs import require_file
from rapthor.execution.shell import run_external_command

SOLVE_TYPE_LABELS = {
    "fast_phase": "fast phase",
    "medium_phase": "medium phase",
    "slow_gains": "slow gains",
    "full_jones": "full-Jones",
}


def _require_sequence(value: object, name: str) -> list[object]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    return value


def _solve_type_label(solve_type: object) -> str:
    """Return a human-readable solve type for logs and output errors."""
    return SOLVE_TYPE_LABELS.get(str(solve_type), str(solve_type).replace("_", " "))


def _solve_slots_for_chunk(
    payload: CalibratePayload,
    chunk: CalibrateChunkPayload,
) -> list[Mapping[str, object]]:
    solve_slots = []
    for slot in chunk["solve_slots"]:
        solve_type = str(slot["solve_type"])
        smoothnessconstraint = slot["smoothnessconstraint"]
        antennaconstraint = slot["antennaconstraint"]
        if solve_type == "full_jones":
            smoothnessconstraint = smoothnessconstraint or payload.get(
                "smoothnessconstraint_fulljones"
            )
            antennaconstraint = antennaconstraint or "[]"
        solve_slots.append(
            {
                "slot": slot["slot"],
                "solve_type": solve_type,
                "h5parm": slot["h5parm"],
                "solint": slot["solint"],
                "mode": slot["mode"],
                "nchan": slot["nchan"],
                "solutions_per_direction": slot["solutions_per_direction"],
                "llssolver": payload["llssolver"],
                "maxiter": payload["maxiter"],
                "propagatesolutions": payload["propagatesolutions"],
                "initialsolutions_soltab": (
                    "[phase000,amplitude000]" if solve_type == "slow_gains" else "[phase000]"
                ),
                "solveralgorithm": payload["solveralgorithm"],
                "solverlbfgs_dof": payload["solverlbfgs_dof"],
                "solverlbfgs_iter": payload["solverlbfgs_iter"],
                "solverlbfgs_minibatches": payload["solverlbfgs_minibatches"],
                "initialsolutions_h5parm": slot.get("initialsolutions_h5parm"),
                "stepsize": payload["stepsize"],
                "stepsigma": payload["stepsigma"],
                "tolerance": payload["tolerance"],
                "uvlambdamin": payload["uvlambdamin"],
                "datause": slot.get("datause"),
                "smoothness_dd_factors": slot["smoothness_dd_factors"],
                "smoothnessconstraint": smoothnessconstraint,
                "smoothnessreffrequency": slot["smoothnessreffrequency"],
                "smoothnessrefdistance": slot["smoothnessrefdistance"],
                "antennaconstraint": antennaconstraint,
                "keepmodel": slot["keepmodel"],
                "reusemodel": slot["reusemodel"],
                "modeldatacolumns": slot.get("modeldatacolumns"),
            }
        )
    solve_slots[0]["correctfreqsmearing"] = payload["correctfreqsmearing"]
    solve_slots[0]["correcttimesmearing"] = payload["correcttimesmearing"]
    return solve_slots


def _ddecal_options_for_chunk(
    payload: CalibratePayload,
    chunk: CalibrateChunkPayload,
) -> DdecalSolveOptions:
    return DdecalSolveOptions(
        msin=str(chunk["msin"]),
        data_colname=str(payload["data_colname"]),
        starttime=str(chunk["starttime"]),
        ntimes=int(chunk["ntimes"]),
        steps=str(payload["dp3_steps"]),
        solve_slots=_solve_slots_for_chunk(payload, chunk),
        num_threads=int(payload["max_threads"]),
        modeldatacolumn=None
        if payload.get("modeldatacolumn") is None
        else str(payload["modeldatacolumn"]),
        applycal_steps=payload.get("applycal_steps"),
        applycal_h5parm=payload.get("applycal_h5parm"),
        fulljones_h5parm=payload.get("fulljones_h5parm"),
        normalize_h5parm=payload.get("normalize_h5parm"),
        timebase=payload.get("bda_timebase"),
        maxinterval=chunk.get("bda_maxinterval"),
        frequencybase=payload.get("bda_frequencybase"),
        minchannels=chunk.get("bda_minchannels"),
        onebeamperpatch=payload.get("onebeamperpatch"),
        parallelbaselines=payload.get("parallelbaselines"),
        sagecalpredict=payload.get("sagecalpredict"),
        sourcedb=None if payload.get("image_based_predict") else payload.get("sourcedb"),
        directions=None if payload.get("image_based_predict") else payload.get("directions"),
        predict_regions=payload.get("predict_regions"),
        predict_images=payload.get("predict_images"),
    )


def run_calibrate_chunk(
    payload: CalibratePayload,
    chunk: CalibrateChunkPayload,
    execution_config: Optional[ExecutionConfig] = None,
    shell_operation_cls=None,
) -> dict:
    """Run one DDECal calibration chunk."""
    config = execution_config or ExecutionConfig(task_runner="sync")
    pipeline_working_dir = str(payload["pipeline_working_dir"])
    command = build_ddecal_solve_command(_ddecal_options_for_chunk(payload, chunk))
    run_external_command(
        command,
        pipeline_working_dir,
        config,
        shell_operation_cls=shell_operation_cls,
    )
    output_records = {}
    for slot in chunk["solve_slots"]:
        label = (
            f"{str(payload['mode']).upper()} solve{slot['slot']} "
            f"{_solve_type_label(slot['solve_type'])} h5parm"
        )
        output_records[f"solve{slot['slot']}"] = require_file(str(slot["h5parm_path"]), label)
    return output_records


def run_calibrate_screen_chunk(
    payload: CalibratePayload,
    chunk: CalibrateChunkPayload,
    execution_config: Optional[ExecutionConfig] = None,
    shell_operation_cls=None,
) -> dict:
    """Run one IDGCal screen-generation chunk."""
    config = execution_config or ExecutionConfig(task_runner="sync")
    pipeline_working_dir = str(payload["pipeline_working_dir"])
    model_images = list(_require_sequence(payload.get("predict_images"), "predict_images"))
    if payload.get("do_slowgain_solve"):
        command = build_idgcal_solve_phase_and_gain_command(
            msin=str(chunk["msin"]),
            starttime=str(chunk["starttime"]),
            ntimes=int(chunk["ntimes"]),
            h5parm=str(chunk["output_h5parm"]),
            solint_fast=int(chunk["solint_fast"]),
            solint_slow=int(chunk["solint_slow"]),
            model_images=[str(path) for path in model_images],
            maxiter=int(payload["solverlbfgs_iter"]),
            antennaconstraint=str(payload["idgcal_antennaconstraint"]),
            numthreads=int(payload["max_threads"]),
        )
    else:
        command = build_idgcal_solve_phase_command(
            msin=str(chunk["msin"]),
            starttime=str(chunk["starttime"]),
            ntimes=int(chunk["ntimes"]),
            h5parm=str(chunk["output_h5parm"]),
            solint=int(chunk["solint_fast"]),
            model_images=[str(path) for path in model_images],
            maxiter=int(payload["solverlbfgs_iter"]),
            antennaconstraint=str(payload["idgcal_antennaconstraint"]),
            numthreads=int(payload["max_threads"]),
        )
    run_external_command(
        command,
        pipeline_working_dir,
        config,
        shell_operation_cls=shell_operation_cls,
    )
    return require_file(str(chunk["output_h5parm_path"]), "IDGCal screen h5parm")

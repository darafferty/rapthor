"""Scheduler-independent runners for calibration execution."""

import glob
import os
from typing import Mapping, Optional

from rapthor.execution.artifacts import publish_plot_file_records
from rapthor.execution.calibrate.commands import (
    build_adjust_h5parm_sources_command,
    build_collect_h5parms_command,
    build_collect_screen_h5parms_command,
    build_combine_h5parms_command,
    build_ddecal_solve_command,
    build_draw_model_command,
    build_idgcal_solve_phase_and_gain_command,
    build_idgcal_solve_phase_command,
    build_make_region_file_command,
    build_plot_solutions_command,
    build_process_gains_command,
)
from rapthor.execution.calibrate.payloads import (
    CalibrateChunkPayload,
    CalibrateImagePredictPayload,
    CalibratePayload,
)
from rapthor.execution.config import ExecutionConfig
from rapthor.execution.shell import ShellCommand, run_shell_command
from rapthor.lib.records import file_record, validate_output_record


def _require_file(path: str, description: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{description} was not created: {path}")
    return file_record(path)


def _require_sequence(value: object, name: str) -> list[object]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    return value


def _di_fulljones_solve_slots(
    payload: CalibratePayload,
    chunk: CalibrateChunkPayload,
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
    payload: CalibratePayload,
    chunk: CalibrateChunkPayload,
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
                "initialsolutions_soltab": (
                    "[phase000,amplitude000]" if int(slot["slot"]) == 3 else "[phase000]"
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
                "smoothnessconstraint": slot["smoothnessconstraint"],
                "smoothnessreffrequency": slot["smoothnessreffrequency"],
                "smoothnessrefdistance": slot["smoothnessrefdistance"],
                "antennaconstraint": slot["antennaconstraint"],
                "keepmodel": slot["keepmodel"],
                "reusemodel": slot["reusemodel"],
                "modeldatacolumns": slot.get("modeldatacolumns"),
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


def _run_draw_model(
    payload: CalibratePayload,
    image_predict: CalibrateImagePredictPayload,
    execution_config: ExecutionConfig,
    shell_operation_cls=None,
) -> list[dict]:
    pipeline_working_dir = str(payload["pipeline_working_dir"])
    command = build_draw_model_command(
        skymodel=str(image_predict["skymodel"]),
        numterms=int(image_predict["num_spectral_terms"]),
        name=str(image_predict["model_image_root"]),
        ra_dec=list(image_predict["model_image_ra_dec"]),
        frequency_bandwidth=list(image_predict["model_image_frequency_bandwidth"]),
        cellsize_deg=image_predict["model_image_cellsize"],
        imsize=list(image_predict["model_image_imsize"]),
        numthreads=int(payload["max_threads"]),
    )
    _run_shell(
        command,
        pipeline_working_dir,
        execution_config,
        shell_operation_cls=shell_operation_cls,
    )
    return [
        _require_file(path, "Calibration model image") for path in image_predict["model_images"]
    ]


def _run_make_region_file(
    image_predict: CalibrateImagePredictPayload,
    payload: CalibratePayload,
    execution_config: ExecutionConfig,
    shell_operation_cls=None,
) -> dict:
    pipeline_working_dir = str(payload["pipeline_working_dir"])
    command = build_make_region_file_command(
        skymodel=str(image_predict["skymodel"]),
        ra_mid=image_predict["ra_mid"],
        dec_mid=image_predict["dec_mid"],
        width_ra=image_predict["facet_region_width_ra"],
        width_dec=image_predict["facet_region_width_dec"],
        outfile=str(image_predict["facet_region_file"]),
        enclose_names=False,
    )
    _run_shell(
        command,
        pipeline_working_dir,
        execution_config,
        shell_operation_cls=shell_operation_cls,
    )
    return _require_file(str(image_predict["facet_region_path"]), "Calibration region file")


def prepare_image_based_predict(
    payload: CalibratePayload,
    execution_config: ExecutionConfig,
    shell_operation_cls=None,
) -> CalibratePayload:
    """Prepare model-image inputs when calibration uses image-based prediction."""
    if not payload.get("image_based_predict"):
        return payload

    image_predict = payload.get("image_predict")
    if not isinstance(image_predict, Mapping):
        raise ValueError("Image-based prediction payload is missing")

    model_images = _run_draw_model(
        payload,
        image_predict,
        execution_config,
        shell_operation_cls=shell_operation_cls,
    )
    region_file = _run_make_region_file(
        image_predict,
        payload,
        execution_config,
        shell_operation_cls=shell_operation_cls,
    )
    prepared_payload = dict(payload)
    prepared_payload["predict_images"] = [record["path"] for record in model_images]
    prepared_payload["predict_regions"] = region_file["path"]
    if payload.get("normalize_h5parm"):
        normalize_record = _run_adjust_h5parm_sources(
            file_record(str(payload["normalize_h5parm"])),
            payload,
            str(payload["pipeline_working_dir"]),
            execution_config,
            "Adjusted normalization h5parm",
            shell_operation_cls=shell_operation_cls,
        )
        prepared_payload["normalize_h5parm"] = normalize_record["path"]
    return prepared_payload


def run_calibrate_chunk(
    payload: CalibratePayload,
    chunk: CalibrateChunkPayload,
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
    _run_shell(command, pipeline_working_dir, config, shell_operation_cls=shell_operation_cls)
    output_records = {}
    for slot in chunk["solve_slots"]:
        label = (
            "DI full-Jones h5parm"
            if payload["calibration_kind"] == "di_fulljones"
            else f"{str(payload['mode']).upper()} solve{slot['slot']} h5parm"
        )
        output_records[f"solve{slot['slot']}"] = _require_file(str(slot["h5parm_path"]), label)
    if payload["calibration_kind"] == "di_fulljones":
        return output_records["solve1"]
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
    _run_shell(command, pipeline_working_dir, config, shell_operation_cls=shell_operation_cls)
    return _require_file(str(chunk["output_h5parm_path"]), "IDGCal screen h5parm")


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

    plot_records = run_plot_solutions(
        collected_record,
        "phase",
        pipeline_working_dir,
        execution_config,
        first_dir=_plot_first_direction(payload),
        shell_operation_cls=shell_operation_cls,
    )

    result = {
        "combined_solutions": collected_record,
        "fast_phase_solutions": collected_record,
        "fast_phase_plots": plot_records,
    }
    for value in result.values():
        validate_output_record(value)
    return result


def _collect_and_plot_fast_phase(
    payload: Mapping[str, object],
    solve_records: list[dict],
    execution_config: ExecutionConfig,
    shell_operation_cls=None,
) -> dict:
    pipeline_working_dir = str(payload["pipeline_working_dir"])
    fast_record = _run_collect_h5parm(
        [record["solve1"] for record in solve_records],
        payload["collected_h5parms"]["solve1"],
        pipeline_working_dir,
        execution_config,
        "Collected DD fast phase h5parm",
        shell_operation_cls=shell_operation_cls,
    )
    fast_plots = run_plot_solutions(
        fast_record,
        "phase",
        pipeline_working_dir,
        execution_config,
        first_dir=_plot_first_direction(payload),
        shell_operation_cls=shell_operation_cls,
    )

    result = {
        "combined_solutions": fast_record,
        "fast_phase_solutions": fast_record,
        "fast_phase_plots": fast_plots,
    }
    for value in result.values():
        validate_output_record(value)
    return result


def _collect_process_and_plot_slow_gain(
    payload: Mapping[str, object],
    solve_records: list[dict],
    execution_config: ExecutionConfig,
    shell_operation_cls=None,
) -> dict:
    pipeline_working_dir = str(payload["pipeline_working_dir"])
    slow_record = _run_collect_h5parm(
        [record["solve1"] for record in solve_records],
        payload["collected_h5parms"]["solve1"],
        pipeline_working_dir,
        execution_config,
        "Collected slow gains h5parm",
        shell_operation_cls=shell_operation_cls,
    )
    processed_slow_record = _run_process_gains(
        slow_record,
        payload,
        pipeline_working_dir,
        execution_config,
        shell_operation_cls=shell_operation_cls,
    )
    slow_phase_plots = run_plot_solutions(
        processed_slow_record,
        "phase",
        pipeline_working_dir,
        execution_config,
        root="slow_phase_",
        first_dir=_plot_first_direction(payload),
        shell_operation_cls=shell_operation_cls,
    )
    slow_amp_plots = run_plot_solutions(
        processed_slow_record,
        "amplitude",
        pipeline_working_dir,
        execution_config,
        root="slow_amplitude_",
        first_dir=_plot_first_direction(payload),
        shell_operation_cls=shell_operation_cls,
    )

    result = {
        "combined_solutions": processed_slow_record,
        "slow_gain_solutions": slow_record,
        "slow_phase_plots": slow_phase_plots,
        "slow_amp_plots": slow_amp_plots,
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


def _run_collect_screen_h5parms(
    input_records: list[dict],
    output: Mapping[str, object],
    pipeline_working_dir: str,
    execution_config: ExecutionConfig,
    shell_operation_cls=None,
) -> dict:
    collect_command = build_collect_screen_h5parms_command(
        [record["path"] for record in input_records],
        str(output["filename"]),
    )
    _run_shell(
        collect_command,
        pipeline_working_dir,
        execution_config,
        shell_operation_cls=shell_operation_cls,
    )
    return _require_file(str(output["path"]), "Combined screen h5parm")


def collect_screen_solutions(
    payload: Mapping[str, object],
    screen_records: list[dict],
    execution_config: ExecutionConfig,
    shell_operation_cls=None,
) -> dict:
    """Collect screen-generation h5parm outputs into the final solution file."""
    pipeline_working_dir = str(payload["pipeline_working_dir"])
    combined_record = _run_collect_screen_h5parms(
        screen_records,
        payload["combined_h5parm"],
        pipeline_working_dir,
        execution_config,
        shell_operation_cls=shell_operation_cls,
    )
    result = {"combined_solutions": combined_record}
    validate_output_record(result["combined_solutions"])
    return result


def run_plot_solutions(
    h5parm_record: Mapping[str, str],
    soltype: str,
    pipeline_working_dir: str,
    execution_config: ExecutionConfig,
    root: Optional[str] = None,
    first_dir: bool = False,
    shell_operation_cls=None,
) -> list[dict]:
    """Run solution plotting and return only newly created plot records."""
    before_plots = set(glob.glob(os.path.join(pipeline_working_dir, "*.png")))
    plot_command = build_plot_solutions_command(
        h5parm_record["path"], soltype, root=root, first_dir=first_dir
    )
    _run_shell(
        plot_command,
        pipeline_working_dir,
        execution_config,
        shell_operation_cls=shell_operation_cls,
    )
    after_plots = set(glob.glob(os.path.join(pipeline_working_dir, "*.png")))
    plot_records = [file_record(path) for path in sorted(after_plots - before_plots)]
    publish_plot_file_records(plot_records, pipeline_working_dir)
    return plot_records


def _plot_first_direction(payload: Mapping[str, object]) -> bool:
    return str(payload.get("mode")) == "di"


def _run_combine_h5parms(
    input_record1: Mapping[str, str],
    input_record2: Mapping[str, str],
    output: Mapping[str, object],
    mode: str,
    payload: Mapping[str, object],
    pipeline_working_dir: str,
    execution_config: ExecutionConfig,
    label: str,
    shell_operation_cls=None,
) -> dict:
    combine_command = build_combine_h5parms_command(
        input_record1["path"],
        input_record2["path"],
        str(output["filename"]),
        mode,
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
    return _require_file(str(output["path"]), label)


def _run_process_gains(
    h5parm_record: Mapping[str, str],
    payload: Mapping[str, object],
    pipeline_working_dir: str,
    execution_config: ExecutionConfig,
    shell_operation_cls=None,
) -> dict:
    process_command = build_process_gains_command(
        h5parm_record["path"],
        flag=True,
        smooth=True,
        max_station_delta=float(payload["max_normalization_delta"]),
        scale_station_delta=str(payload["scale_normalization_delta"]),
        phase_center_ra=payload["phase_center_ra"],
        phase_center_dec=payload["phase_center_dec"],
    )
    _run_shell(
        process_command,
        pipeline_working_dir,
        execution_config,
        shell_operation_cls=shell_operation_cls,
    )
    return _require_file(h5parm_record["path"], "Processed DD slow gains h5parm")


def _should_adjust_dd_sources(payload: Mapping[str, object]) -> bool:
    return len(payload.get("calibrator_patch_names", [])) > 1


def _run_adjust_h5parm_sources(
    h5parm_record: Mapping[str, str],
    payload: Mapping[str, object],
    pipeline_working_dir: str,
    execution_config: ExecutionConfig,
    label: str,
    shell_operation_cls=None,
) -> dict:
    skymodel = payload.get("sourcedb")
    if not skymodel:
        raise ValueError("DD source adjustment requires calibration_skymodel_file")
    adjust_command = build_adjust_h5parm_sources_command(str(skymodel), h5parm_record["path"])
    _run_shell(
        adjust_command,
        pipeline_working_dir,
        execution_config,
        shell_operation_cls=shell_operation_cls,
    )
    return _require_file(h5parm_record["path"], label)


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
    fast_plots = run_plot_solutions(
        fast_record,
        "phase",
        pipeline_working_dir,
        execution_config,
        first_dir=_plot_first_direction(payload),
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
    medium_plots = run_plot_solutions(
        medium_record,
        "phase",
        pipeline_working_dir,
        execution_config,
        root="medium1_phase_",
        first_dir=_plot_first_direction(payload),
        shell_operation_cls=shell_operation_cls,
    )

    combined_h5parm = payload["combined_h5parm"]
    combined_record = _run_combine_h5parms(
        fast_record,
        medium_record,
        combined_h5parm,
        "p1p2_scalar",
        payload,
        pipeline_working_dir,
        execution_config,
        "Combined DI scalar h5parm",
        shell_operation_cls=shell_operation_cls,
    )

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


def _collect_plot_and_combine_dd_phase(
    payload: Mapping[str, object],
    solve_records: list[dict],
    execution_config: ExecutionConfig,
    shell_operation_cls=None,
) -> dict:
    pipeline_working_dir = str(payload["pipeline_working_dir"])
    collected_h5parms = payload["collected_h5parms"]
    combined_h5parms = payload["combined_h5parms"]
    plot_first_dir = _plot_first_direction(payload)

    fast_record = _run_collect_h5parm(
        [record["solve1"] for record in solve_records],
        collected_h5parms["solve1"],
        pipeline_working_dir,
        execution_config,
        "Collected DD fast phase h5parm",
        shell_operation_cls=shell_operation_cls,
    )
    fast_plots = run_plot_solutions(
        fast_record,
        "phase",
        pipeline_working_dir,
        execution_config,
        first_dir=plot_first_dir,
        shell_operation_cls=shell_operation_cls,
    )

    medium1_record = _run_collect_h5parm(
        [record["solve2"] for record in solve_records],
        collected_h5parms["solve2"],
        pipeline_working_dir,
        execution_config,
        "Collected DD medium1 phase h5parm",
        shell_operation_cls=shell_operation_cls,
    )
    medium1_plots = run_plot_solutions(
        medium1_record,
        "phase",
        pipeline_working_dir,
        execution_config,
        root="medium1_phase_",
        first_dir=plot_first_dir,
        shell_operation_cls=shell_operation_cls,
    )
    combined_phase_record = _run_combine_h5parms(
        fast_record,
        medium1_record,
        combined_h5parms["solve1_solve2"],
        "p1p2_scalar",
        payload,
        pipeline_working_dir,
        execution_config,
        "Combined DD fast and medium1 phase h5parm",
        shell_operation_cls=shell_operation_cls,
    )

    result = {
        "combined_solutions": combined_phase_record,
        "fast_phase_solutions": fast_record,
        "medium1_phase_solutions": medium1_record,
        "fast_phase_plots": fast_plots,
        "medium1_phase_plots": medium1_plots,
    }

    if "solve3" in collected_h5parms:
        slow_record = _run_collect_h5parm(
            [record["solve3"] for record in solve_records],
            collected_h5parms["solve3"],
            pipeline_working_dir,
            execution_config,
            "Collected DD slow gains h5parm",
            shell_operation_cls=shell_operation_cls,
        )
        processed_slow_record = _run_process_gains(
            slow_record,
            payload,
            pipeline_working_dir,
            execution_config,
            shell_operation_cls=shell_operation_cls,
        )
        slow_phase_plots = run_plot_solutions(
            processed_slow_record,
            "phase",
            pipeline_working_dir,
            execution_config,
            root="slow_phase_",
            first_dir=plot_first_dir,
            shell_operation_cls=shell_operation_cls,
        )
        slow_amp_plots = run_plot_solutions(
            processed_slow_record,
            "amplitude",
            pipeline_working_dir,
            execution_config,
            root="slow_amplitude_",
            first_dir=plot_first_dir,
            shell_operation_cls=shell_operation_cls,
        )
        result.update(
            {
                "slow_gain_solutions": slow_record,
                "slow_phase_plots": slow_phase_plots,
                "slow_amp_plots": slow_amp_plots,
            }
        )

        phase_record = combined_phase_record
        has_medium2 = "solve4" in collected_h5parms
        if has_medium2:
            medium2_record = _run_collect_h5parm(
                [record["solve4"] for record in solve_records],
                collected_h5parms["solve4"],
                pipeline_working_dir,
                execution_config,
                "Collected DD medium2 phase h5parm",
                shell_operation_cls=shell_operation_cls,
            )
            medium2_plots = run_plot_solutions(
                medium2_record,
                "phase",
                pipeline_working_dir,
                execution_config,
                root="medium2_phase_",
                first_dir=plot_first_dir,
                shell_operation_cls=shell_operation_cls,
            )
            phase_record = _run_combine_h5parms(
                combined_phase_record,
                medium2_record,
                combined_h5parms["solve1_solve2_solve4"],
                "p1p2_scalar",
                payload,
                pipeline_working_dir,
                execution_config,
                "Combined DD fast, medium1, and medium2 phase h5parm",
                shell_operation_cls=shell_operation_cls,
            )
            result.update(
                {
                    "medium2_phase_solutions": medium2_record,
                    "medium2_phase_plots": medium2_plots,
                }
            )

        if str(payload.get("mode")) == "dd":
            final_mode = str(payload.get("solution_combine_mode") or "p1p2a2_scalar")
        else:
            final_mode = "p1a2"
        final_record = _run_combine_h5parms(
            phase_record,
            processed_slow_record,
            combined_h5parms["final"],
            final_mode,
            payload,
            pipeline_working_dir,
            execution_config,
            "Combined DD phase and slow gain h5parm",
            shell_operation_cls=shell_operation_cls,
        )
        if _should_adjust_dd_sources(payload):
            final_record = _run_adjust_h5parm_sources(
                final_record,
                payload,
                pipeline_working_dir,
                execution_config,
                "Adjusted DD combined h5parm",
                shell_operation_cls=shell_operation_cls,
            )
        result["combined_solutions"] = final_record

    elif _should_adjust_dd_sources(payload):
        result["combined_solutions"] = _run_adjust_h5parm_sources(
            combined_phase_record,
            payload,
            pipeline_working_dir,
            execution_config,
            "Adjusted DD phase h5parm",
            shell_operation_cls=shell_operation_cls,
        )

    for value in result.values():
        validate_output_record(value)
    return result


def collect_plot_and_combine(
    payload: Mapping[str, object],
    solve_records: list[dict],
    execution_config: ExecutionConfig,
    shell_operation_cls=None,
) -> dict:
    """Collect, plot, process, and combine calibration solution products."""
    if payload["calibration_kind"] == "di_fulljones":
        return _collect_and_plot_fulljones(
            payload,
            solve_records,
            execution_config,
            shell_operation_cls=shell_operation_cls,
        )
    if payload["calibration_kind"] in {"di_fast_phase", "dd_fast_phase"}:
        return _collect_and_plot_fast_phase(
            payload,
            solve_records,
            execution_config,
            shell_operation_cls=shell_operation_cls,
        )
    if payload["calibration_kind"] == "di_slow":
        return _collect_process_and_plot_slow_gain(
            payload,
            solve_records,
            execution_config,
            shell_operation_cls=shell_operation_cls,
        )
    if payload["calibration_kind"] == "dd_slow":
        return _collect_process_and_plot_slow_gain(
            payload,
            solve_records,
            execution_config,
            shell_operation_cls=shell_operation_cls,
        )
    if payload["calibration_kind"] in {"di_phase_slow", "dd_phase", "dd_phase_slow"}:
        return _collect_plot_and_combine_dd_phase(
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

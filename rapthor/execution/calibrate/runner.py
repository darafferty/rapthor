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
from rapthor.execution.outputs import require_file
from rapthor.execution.shell import ShellCommand, run_shell_command
from rapthor.lib.records import file_record, validate_output_record

PHASE_SOLVE_TYPES = {"fast_phase", "medium_phase"}
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
    return [require_file(path, "Calibration model image") for path in image_predict["model_images"]]


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
    return require_file(str(image_predict["facet_region_path"]), "Calibration region file")


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
    """Run one DDECal calibration chunk."""
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
    _run_shell(command, pipeline_working_dir, config, shell_operation_cls=shell_operation_cls)
    return require_file(str(chunk["output_h5parm_path"]), "IDGCal screen h5parm")


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
    return require_file(str(output["path"]), label)


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
    return require_file(str(output["path"]), "Combined screen h5parm")


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


def _mode_label(payload: Mapping[str, object]) -> str:
    return str(payload.get("mode", "calibration")).upper()


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
    return require_file(str(output["path"]), label)


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
    return require_file(h5parm_record["path"], "Processed slow-gain h5parm")


def _should_adjust_dd_sources(payload: Mapping[str, object]) -> bool:
    return str(payload.get("mode")) == "dd" and len(payload.get("calibrator_patch_names", [])) > 1


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
    return require_file(h5parm_record["path"], label)


def _final_phase_plus_slow_gain_combine_mode(payload: Mapping[str, object]) -> str:
    """Return the combine mode for the final phase-plus-slow-gain h5parm."""
    if str(payload.get("mode")) == "dd":
        return str(payload.get("solution_combine_mode") or "p1p2a2_scalar")
    return "p1a2"


def _solve_key(solve_slot: Mapping[str, object]) -> str:
    return f"solve{solve_slot['slot']}"


def _solve_type(solve_slot: Mapping[str, object]) -> str:
    return str(solve_slot["solve_type"])


def _is_phase_solve(solve_slot: Mapping[str, object]) -> bool:
    return _solve_type(solve_slot) in PHASE_SOLVE_TYPES


def _medium_phase_label(solve_slot: Mapping[str, object]) -> str:
    h5parm = str(solve_slot["h5parm"])
    return "medium2" if h5parm.startswith("medium2_") else "medium1"


def _solution_output_key(solve_slot: Mapping[str, object]) -> str:
    solve_type = _solve_type(solve_slot)
    if solve_type == "fast_phase":
        return "fast_phase_solutions"
    if solve_type == "medium_phase":
        return f"{_medium_phase_label(solve_slot)}_phase_solutions"
    if solve_type == "slow_gains":
        return "slow_gain_solutions"
    if solve_type == "full_jones":
        return "fulljones_solutions"
    return f"{_solve_key(solve_slot)}_solutions"


def _phase_plot_root(solve_slot: Mapping[str, object]) -> Optional[str]:
    if _solve_type(solve_slot) == "medium_phase":
        return f"{_medium_phase_label(solve_slot)}_phase_"
    if _solve_type(solve_slot) == "full_jones":
        return "fulljones_phase_"
    return None


def _plot_output_key(solve_slot: Mapping[str, object], soltype: str) -> str:
    solve_type = _solve_type(solve_slot)
    if solve_type == "fast_phase":
        return "fast_phase_plots"
    if solve_type == "medium_phase":
        return f"{_medium_phase_label(solve_slot)}_phase_plots"
    if solve_type == "slow_gains":
        return "slow_amp_plots" if soltype == "amplitude" else "slow_phase_plots"
    if solve_type == "full_jones":
        return "fulljones_phase_plots"
    return f"{_solve_key(solve_slot)}_{soltype}_plots"


def _combine_phase_outputs(
    payload: Mapping[str, object],
    phase_count: int,
    *,
    has_slow_gain: bool,
) -> list[Mapping[str, object]]:
    combined_h5parms = payload["combined_h5parms"]
    outputs = []
    for key in ("solve1_solve2", "solve1_solve2_solve4"):
        if key in combined_h5parms:
            outputs.append(combined_h5parms[key])
    if not has_slow_gain and "final" in combined_h5parms:
        outputs.append(combined_h5parms["final"])

    required = max(0, phase_count - 1)
    if len(outputs) < required:
        raise ValueError("Calibration phase combination outputs are incomplete")
    return outputs[:required]


def _collect_strategy_solve_products(
    payload: Mapping[str, object],
    solve_records: list[dict],
    execution_config: ExecutionConfig,
    shell_operation_cls=None,
) -> dict:
    pipeline_working_dir = str(payload["pipeline_working_dir"])
    solve_slots = list(payload["chunks"][0]["solve_slots"])
    collected_h5parms = payload["collected_h5parms"]
    plot_first_dir = _plot_first_direction(payload)

    result = {}
    combine_records = {}

    for solve_slot in solve_slots:
        solve_key = _solve_key(solve_slot)
        solve_type = _solve_type(solve_slot)
        collected_record = _run_collect_h5parm(
            [record[solve_key] for record in solve_records],
            collected_h5parms[solve_key],
            pipeline_working_dir,
            execution_config,
            f"Collected {_mode_label(payload)} {solve_key} {solve_type} h5parm",
            shell_operation_cls=shell_operation_cls,
        )
        result[_solution_output_key(solve_slot)] = collected_record

        if solve_type == "slow_gains":
            combine_record = _run_process_gains(
                collected_record,
                payload,
                pipeline_working_dir,
                execution_config,
                shell_operation_cls=shell_operation_cls,
            )
            for soltype in ("phase", "amplitude"):
                root = "slow_amplitude_" if soltype == "amplitude" else "slow_phase_"
                result[_plot_output_key(solve_slot, soltype)] = run_plot_solutions(
                    combine_record,
                    soltype,
                    pipeline_working_dir,
                    execution_config,
                    root=root,
                    first_dir=plot_first_dir,
                    shell_operation_cls=shell_operation_cls,
                )
        else:
            combine_record = collected_record
            result[_plot_output_key(solve_slot, "phase")] = run_plot_solutions(
                collected_record,
                "phase",
                pipeline_working_dir,
                execution_config,
                root=_phase_plot_root(solve_slot),
                first_dir=plot_first_dir,
                shell_operation_cls=shell_operation_cls,
            )
        combine_records[solve_key] = combine_record

    phase_slots = [slot for slot in solve_slots if _is_phase_solve(slot)]
    slow_slots = [slot for slot in solve_slots if _solve_type(slot) == "slow_gains"]
    fulljones_slots = [slot for slot in solve_slots if _solve_type(slot) == "full_jones"]

    phase_record = None
    if phase_slots:
        phase_record = combine_records[_solve_key(phase_slots[0])]
        for output_record, solve_slot in zip(
            _combine_phase_outputs(
                payload,
                len(phase_slots),
                has_slow_gain=bool(slow_slots),
            ),
            phase_slots[1:],
        ):
            phase_record = _run_combine_h5parms(
                phase_record,
                combine_records[_solve_key(solve_slot)],
                output_record,
                "p1p2_scalar",
                payload,
                pipeline_working_dir,
                execution_config,
                f"Combined {_mode_label(payload)} phase h5parm",
                shell_operation_cls=shell_operation_cls,
            )

    if slow_slots:
        if len(slow_slots) > 1:
            raise ValueError("A calibration cycle can contain at most one slow_gains solve")
        slow_record = combine_records[_solve_key(slow_slots[0])]
        if phase_record is None:
            active_record = slow_record
        else:
            final_output = payload["combined_h5parms"].get("final")
            if final_output is None:
                raise ValueError("Calibration final combination output is missing")
            active_record = _run_combine_h5parms(
                phase_record,
                slow_record,
                final_output,
                _final_phase_plus_slow_gain_combine_mode(payload),
                payload,
                pipeline_working_dir,
                execution_config,
                f"Combined {_mode_label(payload)} phase and slow-gain h5parm",
                shell_operation_cls=shell_operation_cls,
            )
    elif phase_record is not None:
        active_record = phase_record
    elif fulljones_slots:
        active_record = combine_records[_solve_key(fulljones_slots[0])]
    else:
        raise ValueError("Calibration produced no active solution product")

    if _should_adjust_dd_sources(payload) and len(phase_slots) > 1:
        active_record = _run_adjust_h5parm_sources(
            active_record,
            payload,
            pipeline_working_dir,
            execution_config,
            f"Adjusted {_mode_label(payload)} combined h5parm",
            shell_operation_cls=shell_operation_cls,
        )
    result["combined_solutions"] = active_record

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
    return _collect_strategy_solve_products(
        payload,
        solve_records,
        execution_config,
        shell_operation_cls=shell_operation_cls,
    )

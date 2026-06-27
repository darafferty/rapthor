"""Prefect flows for the Calibrate operation."""

import glob
import os
from typing import Mapping, Optional

from prefect import flow, task

from rapthor.execution.artifacts import publish_plot_file_records
from rapthor.execution.commands import (
    append_key_value,
    bool_token,
    bracketed_list_token,
    comma_join,
    normalize_command,
)
from rapthor.execution.config import ExecutionConfig
from rapthor.execution.flows.runtime import run_flow_with_task_runner
from rapthor.execution.outputs import (
    directory_record_path,
    file_record,
    file_record_path,
    validate_output_record,
)
from rapthor.execution.payloads import (
    CalibrateChunkPayload,
    CalibrateImagePredictPayload,
    CalibratePayload,
    CalibrateSolveSlotPayload,
    assert_serializable_payload,
)
from rapthor.execution.prefect_logging import publish_python_logs_to_prefect
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

IDGCAL_PHASE_ARGUMENTS = [
    "msin.datacolumn=DATA",
    "msout=",
    "steps=[solve]",
    "solve.type=python",
    "solve.python.module=idg.idgcaldpstep_phase_only_dirac",
    "solve.python.class=IDGCalDPStepPhaseOnlyDirac",
    "solve.nrcorrelations=4",
    "solve.subgridsize=32",
    "solve.tapersupport=7",
    "solve.wtermsupport=5",
    "solve.atermsupport=5",
    "solve.solverupdategain=0.5",
    "solve.tolerancepinv=1e-9",
    "solve.polynomialdegphase=2",
    "solve.nr_channels_per_block=30",
    "solve.lbfgshistory=10",
    "solve.lbfgsminibatches=3",
    "solve.lbfgsepochs=3",
]

IDGCAL_PHASE_AND_GAIN_ARGUMENTS = [
    *IDGCAL_PHASE_ARGUMENTS[:4],
    "solve.python.module=idg.idgcaldpstep_rapthor_dirac",
    "solve.python.class=IDGCalDPStepRapthorDirac",
    *IDGCAL_PHASE_ARGUMENTS[6:14],
    "solve.polynomialdegamplitude=2",
    *IDGCAL_PHASE_ARGUMENTS[14:],
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
    ("modeldatacolumns", "modeldatacolumns"),
    ("normalize_h5parm", "applycal.normalization.parmdb"),
    ("applycal_steps", "applycal.steps"),
]

# These are the DP3 step names supported for the standalone applycal that runs
# before DD solves. `fastphase` applies the phase000 soltab from the selected
# scalar h5parm; for DI fast+medium this is already the combined scalar product.
SUPPORTED_DD_PREAPPLY_STEPS = {"fastphase", "slowgain", "fulljones", "normalization"}


def _optional_file_path(record: object, name: str) -> Optional[str]:
    if record is None:
        return None
    if isinstance(record, str):
        return record
    if isinstance(record, Mapping) and record.get("class") == "File":
        return file_record_path(record)
    raise ValueError(f"{name} must be a File record, path string, or None")


def _validate_basename(filename: object, name: str) -> str:
    if not isinstance(filename, str) or not filename:
        raise ValueError(f"{name} must be a non-empty string")
    if os.path.isabs(filename) or os.path.basename(filename) != filename:
        raise ValueError(f"{name} must be a basename")
    return filename


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


def _require_file(path: str, label: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} was not created: {path}")
    return file_record(path)


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
    if "null" in _parse_steps(steps):
        command.append("null.type=null")
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
            None if predict_images is None else bracketed_list_token(predict_images),
        ),
        ("solve1.sourcedb", sourcedb),
        ("solve1.directions", directions),
    ]
    for prefix, value in common_options:
        append_key_value(command, prefix, value)

    for slot in solve_slots:
        slot_index = int(slot["slot"])
        for key, suffix in SOLVE_SLOT_ARGUMENTS:
            append_key_value(command, f"solve{slot_index}.{suffix}", slot.get(key))

    append_key_value(command, "numthreads", numthreads)
    return command


def build_draw_model_command(
    skymodel: str,
    numterms: int,
    name: str,
    ra_dec: list[str],
    frequency_bandwidth: list[object],
    cellsize_deg: object,
    imsize: list[int],
    numthreads: int,
) -> list[str]:
    """Build the WSClean command that draws calibration model images."""
    return [
        "wsclean",
        "-j",
        str(numthreads),
        "-draw-model",
        skymodel,
        "-draw-spectral-terms",
        str(numterms),
        "-name",
        name,
        "-draw-centre",
        *[str(value) for value in ra_dec],
        "-draw-frequencies",
        *[str(value) for value in frequency_bandwidth],
        "-size",
        *[str(value) for value in imsize],
        "-scale",
        str(cellsize_deg),
    ]


def build_make_region_file_command(
    skymodel: str,
    ra_mid: object,
    dec_mid: object,
    width_ra: object,
    width_dec: object,
    outfile: str,
    enclose_names: bool = False,
) -> list[str]:
    """Build the field-level region-file command for calibration image predict."""
    return [
        "make_region_file.py",
        skymodel,
        str(ra_mid),
        str(dec_mid),
        str(width_ra),
        str(width_dec),
        outfile,
        f"--enclose_names={bool_token(enclose_names)}",
    ]


def _first_model_image(model_images: list[str]) -> str:
    return str(_require_sequence(model_images, "model_images")[0])


def build_idgcal_solve_phase_command(
    msin: str,
    starttime: str,
    ntimes: int,
    h5parm: str,
    solint: int,
    model_images: list[str],
    maxiter: int,
    antennaconstraint: str,
    numthreads: int,
) -> list[str]:
    """Build the DP3/IDGCal phase-screen solve command for one chunk."""
    return [
        "DP3",
        *IDGCAL_PHASE_ARGUMENTS,
        f"msin={msin}",
        f"msin.starttime={starttime}",
        f"msin.ntimes={ntimes}",
        f"solve.h5parm={h5parm}",
        f"solve.solintphase={solint}",
        f"solve.modelimage={_first_model_image(model_images)}",
        f"solve.maxiter={maxiter}",
        f"solve.antennaconstraint={antennaconstraint}",
        f"numthreads={numthreads}",
    ]


def build_idgcal_solve_phase_and_gain_command(
    msin: str,
    starttime: str,
    ntimes: int,
    h5parm: str,
    solint_fast: int,
    solint_slow: int,
    model_images: list[str],
    maxiter: int,
    antennaconstraint: str,
    numthreads: int,
) -> list[str]:
    """Build the DP3/IDGCal phase-and-gain screen solve command for one chunk."""
    return [
        "DP3",
        *IDGCAL_PHASE_AND_GAIN_ARGUMENTS,
        f"msin={msin}",
        f"msin.starttime={starttime}",
        f"msin.ntimes={ntimes}",
        f"solve.h5parm={h5parm}",
        f"solve.solintphase={solint_fast}",
        f"solve.solintamplitude={solint_slow}",
        f"solve.modelimage={_first_model_image(model_images)}",
        f"solve.maxiter={maxiter}",
        f"solve.antennaconstraint={antennaconstraint}",
        f"numthreads={numthreads}",
    ]


def build_collect_h5parms_command(inh5parms: list[str], outputh5parm: str) -> list[str]:
    """Build the h5parm collection command."""
    return [
        "H5parm_collector.py",
        "-c",
        comma_join(inh5parms),
        f"--outh5parm={outputh5parm}",
    ]


def build_collect_screen_h5parms_command(inh5parms: list[str], outputh5parm: str) -> list[str]:
    """Build the screen h5parm collection command."""
    return [
        "collect_screen_h5parms.py",
        "-c",
        comma_join(inh5parms),
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
        f"--reweight={bool_token(reweight)}",
    ]
    if calibrator_names is not None:
        command.append(f"--cal_names={comma_join(calibrator_names)}")
    if calibrator_fluxes is not None:
        command.append(f"--cal_fluxes={comma_join(calibrator_fluxes)}")
    return command


def build_process_gains_command(
    h5parm: str,
    flag: bool,
    smooth: bool,
    max_station_delta: float,
    scale_station_delta: str,
    phase_center_ra: object,
    phase_center_dec: object,
) -> list[str]:
    """Build the gain processing command."""
    return [
        "process_gains.py",
        "--normalize=True",
        h5parm,
        f"--smooth={bool_token(smooth)}",
        f"--flag={bool_token(flag)}",
        f"--max_station_delta={max_station_delta}",
        f"--scale_delta_with_dist={scale_station_delta}",
        f"--phase_center_ra={phase_center_ra}",
        f"--phase_center_dec={phase_center_dec}",
    ]


def build_adjust_h5parm_sources_command(skymodel: str, h5parm: str) -> list[str]:
    """Build the h5parm source-adjustment command."""
    return ["adjust_h5parm_sources.py", skymodel, h5parm]


def build_plot_solutions_command(
    h5parm: str,
    soltype: str,
    root: Optional[str] = None,
    first_dir: bool = False,
) -> list[str]:
    """Build the solution plotting command."""
    command = ["plotrapthor", h5parm, soltype]
    if root is not None:
        command.append(f"--root={root}")
    if first_dir:
        command.append("--first-dir")
    return command


def _parse_steps(steps: object) -> list[str]:
    return [step.strip() for step in str(steps).strip("[]").split(",") if step.strip()]


def _active_solve_steps(input_parms: Mapping[str, object]) -> list[str]:
    return [step for step in _parse_steps(input_parms.get("dp3_steps")) if step.startswith("solve")]


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

    applycal_steps = _parse_steps(applycal_steps_record)
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
    steps = _parse_steps(input_parms.get("dp3_steps"))
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
    steps = _parse_steps(input_parms.get("dp3_steps"))
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
    dp3_steps = _parse_steps(input_parms.get("dp3_steps"))
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


def _validate_string_list(values: object, name: str) -> list[str]:
    if not isinstance(values, list) or not all(
        isinstance(value, str) and value for value in values
    ):
        raise ValueError(f"{name} must be a list of strings")
    return values


def _validate_int_list(values: object, name: str, length: Optional[int] = None) -> list[int]:
    if not isinstance(values, list) or not all(isinstance(value, int) for value in values):
        raise ValueError(f"{name} must be a list of integers")
    if length is not None and len(values) != length:
        raise ValueError(f"{name} must contain exactly {length} entries")
    return values


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


def _validate_calibrate_payload(payload: Mapping[str, object]) -> CalibratePayload:
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


def _prepare_image_based_predict(
    payload: CalibratePayload,
    execution_config: ExecutionConfig,
    shell_operation_cls=None,
) -> CalibratePayload:
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


@task(name="calibrate_chunk")
def calibrate_chunk_task(
    payload: CalibratePayload,
    chunk: CalibrateChunkPayload,
    execution_config: Optional[ExecutionConfig] = None,
    shell_operation_cls=None,
) -> dict:
    """Prefect task wrapper for one calibration chunk."""
    with publish_python_logs_to_prefect():
        return run_calibrate_chunk(
            payload,
            chunk,
            execution_config=execution_config,
            shell_operation_cls=shell_operation_cls,
        )


@task(name="calibrate_screen_chunk")
def calibrate_screen_chunk_task(
    payload: CalibratePayload,
    chunk: CalibrateChunkPayload,
    execution_config: Optional[ExecutionConfig] = None,
    shell_operation_cls=None,
) -> dict:
    """Prefect task wrapper for one screen-generation chunk."""
    with publish_python_logs_to_prefect():
        return run_calibrate_screen_chunk(
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

    plot_records = _run_plot_solutions(
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
    fast_plots = _run_plot_solutions(
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
    slow_phase_plots = _run_plot_solutions(
        processed_slow_record,
        "phase",
        pipeline_working_dir,
        execution_config,
        root="slow_phase_",
        first_dir=_plot_first_direction(payload),
        shell_operation_cls=shell_operation_cls,
    )
    slow_amp_plots = _run_plot_solutions(
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


def _collect_screen_solutions(
    payload: Mapping[str, object],
    screen_records: list[dict],
    execution_config: ExecutionConfig,
    shell_operation_cls=None,
) -> dict:
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


def _run_plot_solutions(
    h5parm_record: Mapping[str, str],
    soltype: str,
    pipeline_working_dir: str,
    execution_config: ExecutionConfig,
    root: Optional[str] = None,
    first_dir: bool = False,
    shell_operation_cls=None,
) -> list[dict]:
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
    fast_plots = _run_plot_solutions(
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
    medium_plots = _run_plot_solutions(
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
    fast_plots = _run_plot_solutions(
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
    medium1_plots = _run_plot_solutions(
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
        slow_phase_plots = _run_plot_solutions(
            processed_slow_record,
            "phase",
            pipeline_working_dir,
            execution_config,
            root="slow_phase_",
            first_dir=plot_first_dir,
            shell_operation_cls=shell_operation_cls,
        )
        slow_amp_plots = _run_plot_solutions(
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
            medium2_plots = _run_plot_solutions(
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


def run_calibrate_flow(
    payload: Mapping[str, object],
    execution_config: Optional[ExecutionConfig] = None,
    shell_operation_cls=None,
) -> dict:
    """Run calibration commands and return finalizer-compatible outputs."""
    assert_serializable_payload(payload)
    config = execution_config or ExecutionConfig(task_runner="sync")
    payload = _validate_calibrate_payload(payload)
    payload = _prepare_image_based_predict(
        payload,
        config,
        shell_operation_cls=shell_operation_cls,
    )
    if payload["calibration_kind"] == "dd_screen":
        screen_records = []
        for chunk in payload["chunks"]:
            screen_records.append(
                run_calibrate_screen_chunk(
                    payload,
                    chunk,
                    execution_config=config,
                    shell_operation_cls=shell_operation_cls,
                )
            )
        return _collect_screen_solutions(
            payload,
            screen_records,
            config,
            shell_operation_cls=shell_operation_cls,
        )

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
    payload = _validate_calibrate_payload(payload)
    payload = _prepare_image_based_predict(payload, config)
    if payload["calibration_kind"] == "dd_screen":
        screen_records = [
            calibrate_screen_chunk_task.submit(payload, chunk, execution_config=config)
            for chunk in payload["chunks"]
        ]
        screen_records = [record.result() for record in screen_records]
        return _collect_screen_solutions(payload, screen_records, config)

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
    with publish_python_logs_to_prefect():
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


def normalized_draw_model_command(**kwargs) -> list[str]:
    """Return normalized WSClean draw-model command tokens."""
    return normalize_command(build_draw_model_command(**kwargs))


def normalized_make_region_file_command(**kwargs) -> list[str]:
    """Return normalized region-file command tokens."""
    return normalize_command(build_make_region_file_command(**kwargs))


def normalized_idgcal_solve_phase_command(**kwargs) -> list[str]:
    """Return normalized IDGCal phase-screen solve command tokens."""
    return normalize_command(build_idgcal_solve_phase_command(**kwargs))


def normalized_idgcal_solve_phase_and_gain_command(**kwargs) -> list[str]:
    """Return normalized IDGCal phase-and-gain solve command tokens."""
    return normalize_command(build_idgcal_solve_phase_and_gain_command(**kwargs))


def normalized_collect_h5parms_command(
    inh5parms: list[str],
    outputh5parm: str,
) -> list[str]:
    """Return normalized h5parm collection command tokens."""
    return normalize_command(build_collect_h5parms_command(inh5parms, outputh5parm))


def normalized_collect_screen_h5parms_command(
    inh5parms: list[str],
    outputh5parm: str,
) -> list[str]:
    """Return normalized screen h5parm collection command tokens."""
    return normalize_command(build_collect_screen_h5parms_command(inh5parms, outputh5parm))


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


def normalized_process_gains_command(
    h5parm: str,
    flag: bool,
    smooth: bool,
    max_station_delta: float,
    scale_station_delta: str,
    phase_center_ra: object,
    phase_center_dec: object,
) -> list[str]:
    """Return normalized gain processing command tokens."""
    return normalize_command(
        build_process_gains_command(
            h5parm,
            flag,
            smooth,
            max_station_delta,
            scale_station_delta,
            phase_center_ra,
            phase_center_dec,
        )
    )


def normalized_adjust_h5parm_sources_command(skymodel: str, h5parm: str) -> list[str]:
    """Return normalized h5parm source-adjustment command tokens."""
    return normalize_command(build_adjust_h5parm_sources_command(skymodel, h5parm))


def normalized_plot_solutions_command(
    h5parm: str,
    soltype: str,
    root: Optional[str] = None,
    first_dir: bool = False,
) -> list[str]:
    """Return normalized solution plotting command tokens."""
    return normalize_command(
        build_plot_solutions_command(h5parm, soltype, root=root, first_dir=first_dir)
    )

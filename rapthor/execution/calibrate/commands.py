"""Command builders for calibration execution."""

from dataclasses import dataclass
from typing import Mapping, Optional

from rapthor.execution.commands import (
    append_key_value,
    bool_token,
    bracketed_list_token,
    comma_join,
)

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


@dataclass(frozen=True)
class DdecalSolveOptions:
    """DP3 DDECal options for one calibration chunk."""

    msin: str
    data_colname: str
    starttime: str
    ntimes: int
    steps: str
    solve_slots: list[Mapping[str, object]]
    num_threads: int
    modeldatacolumn: Optional[str] = None
    applycal_steps: Optional[str] = None
    applycal_h5parm: Optional[str] = None
    fulljones_h5parm: Optional[str] = None
    normalize_h5parm: Optional[str] = None
    timebase: Optional[object] = None
    maxinterval: Optional[object] = None
    frequencybase: Optional[object] = None
    minchannels: Optional[object] = None
    onebeamperpatch: Optional[bool] = None
    parallelbaselines: Optional[bool] = None
    sagecalpredict: Optional[bool] = None
    sourcedb: Optional[str] = None
    directions: Optional[list[str]] = None
    predict_regions: Optional[str] = None
    predict_images: Optional[list[str]] = None


def parse_steps(steps: object) -> list[str]:
    """Parse a DP3 steps token into individual step names."""
    return [step.strip() for step in str(steps).strip("[]").split(",") if step.strip()]


def build_ddecal_solve_command(options: DdecalSolveOptions) -> list[str]:
    """Build the DP3 DDECal solve command for one calibration chunk."""
    command = ["DP3", *DDECAL_SOLVE_ARGUMENTS]
    if "null" in parse_steps(options.steps):
        command.append("null.type=null")
    common_options = [
        ("msin", options.msin),
        ("msin.datacolumn", options.data_colname),
        ("msin.starttime", options.starttime),
        ("msin.ntimes", options.ntimes),
        ("steps", options.steps),
        ("applycal.steps", options.applycal_steps),
        ("applycal.parmdb", options.applycal_h5parm),
        ("applycal.fulljones.parmdb", options.fulljones_h5parm),
        ("applycal.normalization.parmdb", options.normalize_h5parm),
        ("avg.timebase", options.timebase),
        ("avg.maxinterval", options.maxinterval),
        ("avg.frequencybase", options.frequencybase),
        ("avg.minchannels", options.minchannels),
        ("solve1.modeldatacolumns", options.modeldatacolumn),
        ("solve1.onebeamperpatch", options.onebeamperpatch),
        ("solve1.parallelbaselines", options.parallelbaselines),
        ("solve1.sagecalpredict", options.sagecalpredict),
        ("predict.regions", options.predict_regions),
        (
            "predict.images",
            None
            if options.predict_images is None
            else bracketed_list_token(options.predict_images),
        ),
        ("solve1.sourcedb", options.sourcedb),
        ("solve1.directions", options.directions),
    ]
    for prefix, value in common_options:
        append_key_value(command, prefix, value)

    for slot in options.solve_slots:
        slot_index = int(slot["slot"])
        for key, suffix in SOLVE_SLOT_ARGUMENTS:
            append_key_value(command, f"solve{slot_index}.{suffix}", slot.get(key))

    append_key_value(command, "numthreads", options.num_threads)
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
    if not isinstance(model_images, list) or not model_images:
        raise ValueError("model_images must be a non-empty list")
    return str(model_images[0])


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

"""Payload contracts for calibration execution."""

from typing import Optional, TypedDict


class CalibrateOutputPayload(TypedDict):
    """Serializable filename/path pair for calibration output h5parms."""

    filename: str
    path: str


class CalibrateSolveSlotPayload(TypedDict, total=False):
    """Serializable payload for one DP3 solve slot inside a calibration chunk."""

    slot: int
    solve_type: str
    solution_label: str
    medium_index: Optional[int]
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
    wsclean_predict: bool
    image_predict: Optional[CalibrateImagePredictPayload]
    predict_regions: str
    predict_images: list[str]
    has_slow_gain_solve: bool
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
    collected_h5parms: dict[str, CalibrateOutputPayload]
    combined_h5parm: Optional[CalibrateOutputPayload]
    combined_h5parms: dict[str, CalibrateOutputPayload]
    calibrator_patch_names: list[str]
    calibrator_fluxes: list[float]
    chunks: list[CalibrateChunkPayload]

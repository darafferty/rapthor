"""Serializable task-payload validation helpers."""

from typing import Any, Mapping, Optional, TypedDict, Union


class ConcatenateEpochPayload(TypedDict):
    """Serializable inputs and expected output for one concatenate epoch."""

    input_filenames: list[str]
    output_filename: str
    output_path: str


class ConcatenatePayload(TypedDict):
    """Serializable payload submitted to the concatenate flow."""

    pipeline_working_dir: str
    data_colname: str
    epochs: list[ConcatenateEpochPayload]


class MosaicImageTypePayload(TypedDict):
    """Serializable inputs and expected outputs for one mosaic image type."""

    sector_image_filenames: list[str]
    sector_vertices_filenames: list[str]
    template_image_filename: str
    template_image_path: str
    regridded_image_filenames: list[str]
    mosaic_filename: str
    mosaic_path: str


class MosaicPayload(TypedDict):
    """Serializable payload submitted to the mosaic flow."""

    pipeline_working_dir: str
    compress_images: bool
    skip_processing: bool
    image_types: list[MosaicImageTypePayload]


class PredictModelTaskPayload(TypedDict):
    """Serializable payload for one DP3 model prediction task."""

    msin: str
    data_colname: str
    msout: str
    msout_path: str
    starttime: str
    ntimes: int
    onebeamperpatch: bool
    correctfreqsmearing: bool
    correcttimesmearing: bool
    sagecalpredict: bool
    sourcedb: str
    directions: list[str]
    numthreads: int
    h5parm: Optional[str]
    applycal_steps: Optional[str]
    normalize_h5parm: Optional[str]


class PredictPostprocessTaskPayload(TypedDict):
    """Serializable payload for one DI predict post-processing task."""

    msobs: str
    data_colname: str
    obs_starttime: str
    infix: str


class PredictDDPostprocessTaskPayload(PredictPostprocessTaskPayload):
    """Serializable DD-specific extension for predict post-processing."""

    solint_sec: float
    solint_hz: float
    min_uv_lambda: float
    max_uv_lambda: float
    nr_outliers: int
    peel_outliers: bool
    nr_bright: int
    peel_bright: bool
    reweight: bool


PredictPostprocessPayload = Union[
    PredictPostprocessTaskPayload,
    PredictDDPostprocessTaskPayload,
]


class PredictPayload(TypedDict):
    """Serializable payload submitted to the predict flow."""

    mode: str
    pipeline_working_dir: str
    predict_tasks: list[PredictModelTaskPayload]
    postprocess_tasks: list[PredictPostprocessPayload]


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


class PayloadSerializationError(TypeError):
    """Raised when a task payload is not safe to send to a Dask worker."""


def assert_serializable_payload(value: Any, path: str = "payload") -> Any:
    """Validate that *value* contains only simple serializable payload values.

    Task payloads intentionally use strings for paths and plain Python
    containers. This avoids passing live Rapthor domain objects to Dask workers.
    The validated value is returned unchanged for convenient inline use.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, list):
        for index, item in enumerate(value):
            assert_serializable_payload(item, f"{path}[{index}]")
        return value

    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise PayloadSerializationError(f"{path} has non-string key {key!r}")
            assert_serializable_payload(item, f"{path}.{key}")
        return value

    raise PayloadSerializationError(
        f"{path} contains unsupported value {value!r} of type {type(value).__name__}"
    )

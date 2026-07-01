"""Payload contracts for image execution."""

from typing import Optional, TypedDict, Union

ImageLinkPolarizations = Union[bool, str]


class ImagePrepareTaskPayload(TypedDict):
    """Serializable payload for one observation prepared before imaging."""

    msin: str
    msout: str
    msout_path: str
    starttime: str
    ntimes: int
    freqstep: int
    timestep: int
    maxinterval: Optional[int]


class ImageCubeSpecPayload(TypedDict):
    """Serializable image-cube output specification for one Stokes plane."""

    pol: str
    filename: str
    path: str


class ImageSectorPayload(TypedDict):
    """Serializable payload for one image-sector task."""

    image_name: str
    apply_screens: bool
    use_facets: bool
    use_mpi: bool
    compress_images: bool
    make_image_cube: bool
    normalize_flux_scale: bool
    peel_bright_sources: bool
    save_filtered_model_image: bool
    bright_skymodel_pb: Optional[str]
    data_colname: str
    prepare_tasks: list[ImagePrepareTaskPayload]
    concat_filename: str
    concat_path: str
    previous_mask_filename: Optional[str]
    mask_filename: str
    mask_path: str
    timebase: Optional[float]
    phasecenter: str
    h5parm: Optional[str]
    fulljones_h5parm: Optional[str]
    input_normalize_h5parm: Optional[str]
    prepare_data_steps: str
    prepare_data_applycal_steps: Optional[str]
    central_patch_name: Optional[str]
    channels_out: int
    deconvolution_channels: int
    fit_spectral_pol: int
    ra: float
    dec: float
    wsclean_imsize: list[int]
    vertices_file: str
    region_file: Optional[str]
    facet_skymodel: Optional[str]
    facet_region_filename: Optional[str]
    facet_region_path: Optional[str]
    filtered_model_image_filename: Optional[str]
    filtered_model_image_path: Optional[str]
    image_I_cube_filename: Optional[str]
    image_I_cube_path: Optional[str]
    image_cube_specs: list[ImageCubeSpecPayload]
    output_source_catalog_filename: Optional[str]
    output_source_catalog_path: Optional[str]
    output_normalize_h5parm_filename: Optional[str]
    output_normalize_h5parm_path: Optional[str]
    ra_mid: Optional[float]
    dec_mid: Optional[float]
    width_ra: Optional[float]
    width_dec: Optional[float]
    wsclean_niter: int
    wsclean_nmiter: int
    skip_final_iteration: bool
    robust: float
    cellsize_deg: float
    min_uv_lambda: float
    max_uv_lambda: float
    mgain: float
    taper_arcsec: float
    local_rms_strength: float
    local_rms_window: float
    local_rms_method: str
    auto_mask: float
    auto_mask_nmiter: int
    idg_mode: str
    wsclean_mem: float
    threshisl: float
    threshpix: float
    do_multiscale: bool
    dd_psf_grid: list[int]
    interval: Optional[list[int]]
    soltabs: Optional[str]
    parallel_gridding_threads: Optional[int]
    scalar_visibilities: Optional[bool]
    diagonal_visibilities: Optional[bool]
    shared_facet_reads: Optional[bool]
    shared_facet_writes: Optional[bool]
    pol: str
    save_source_list: bool
    link_polarizations: ImageLinkPolarizations
    join_polarizations: bool
    filter_by_mask: bool
    source_finder: str
    apply_time_frequency_smearing: bool
    max_threads: int
    deconvolution_threads: int
    mpi_nnodes: Optional[int]
    mpi_cpus_per_task: Optional[int]
    allow_internet_access: bool
    photometry_skymodel: Optional[str]
    astrometry_skymodel: Optional[str]
    obs_original_paths: list[str]
    obs_starttime: list[str]
    obs_ntimes: list[int]


class ImagePayload(TypedDict):
    """Serializable payload submitted to the image flow."""

    mode: str
    use_mpi: bool
    pipeline_working_dir: str
    sectors: list[ImageSectorPayload]

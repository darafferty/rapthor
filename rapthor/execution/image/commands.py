"""Command builders for the Image execution flow."""

from dataclasses import dataclass, replace
from typing import Optional

from rapthor.execution.commands import (
    append_flag,
    append_option_value,
    append_option_values,
    append_prefixed_value,
    bool_token,
    comma_join,
    python_module_command,
)

ATERM_CONFIG_FILENAME = "aterm_plus_beam.cfg"
CUBE_CATALOG_MODULE = "rapthor.execution.image.cube_catalog_cli"
FILTER_SKYMODEL_MODULE = "rapthor.execution.image.skymodel_filter_cli"


@dataclass(frozen=True)
class PrepareImagingDataOptions:
    """DP3 options used to prepare one observation for imaging."""

    msin: str
    data_colname: str
    msout: str
    starttime: str
    ntimes: int
    phasecenter: str
    freqstep: int
    timestep: int
    beamdir: str
    num_threads: int
    steps: str
    maxinterval: Optional[int] = None
    timebase: Optional[float] = None
    h5parm: Optional[str] = None
    fulljones_h5parm: Optional[str] = None
    normalize_h5parm: Optional[str] = None
    central_patch_name: Optional[str] = None
    applycal_steps: Optional[str] = None


@dataclass(frozen=True)
class WscleanOptions:
    """Common WSClean options shared by image modes."""

    msin: str
    name: str
    mask: str
    imsize: list[int]
    niter: int
    nmiter: int
    robust: float
    min_uv_lambda: float
    max_uv_lambda: float
    mgain: float
    multiscale: bool
    save_source_list: bool
    pol: str
    link_polarizations: Optional[str]
    join_polarizations: bool
    skip_final_iteration: bool
    cellsize_deg: float
    channels_out: int
    deconvolution_channels: int
    fit_spectral_pol: int
    taper_arcsec: float
    local_rms_strength: float
    local_rms_window: float
    local_rms_method: str
    memory_gb: float
    auto_mask: float
    auto_mask_nmiter: int
    idg_mode: str
    num_threads: int
    num_deconvolution_threads: int
    dd_psf_grid: list[int]
    apply_time_frequency_smearing: bool
    temp_dir: str


@dataclass(frozen=True)
class WscleanFacetOptions:
    """Facet-specific WSClean options."""

    common: WscleanOptions
    scalar_visibilities: bool
    diagonal_visibilities: bool
    h5parm: str
    soltabs: str
    region_file: str
    num_gridding_threads: Optional[int]
    shared_facet_reads: bool
    shared_facet_writes: bool


@dataclass(frozen=True)
class WscleanScreenOptions:
    """Screen-specific WSClean options."""

    common: WscleanOptions
    interval: list[int]
    aterm_config: str = ATERM_CONFIG_FILENAME


def _strip_wrapping_shell_quotes(value: str) -> str:
    """Remove caller-supplied grouping quotes before `shlex.join` applies shell quoting."""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def build_aterm_config_content(h5parm: str) -> str:
    """Build the a-term configuration file content used by screen imaging."""
    return (
        "aterms = [idgcalsolutions, beam]\n"
        "idgcalsolutions.type = h5parm\n"
        f"idgcalsolutions.files = [{h5parm}]\n"
        "idgcalsolutions.update_interval = 8\n"
        "beam.differential = true\n"
        "beam.update_interval = 120\n"
        "beam.usechannelfreq = true\n"
    )


def build_prepare_imaging_data_command(options: PrepareImagingDataOptions) -> list[str]:
    """Build the DP3 command that prepares one observation for imaging."""
    command = [
        "DP3",
        "msout.overwrite=True",
        "shift.type=phaseshifter",
        "avg.type=squash",
        "bdaavg.type=bdaaverager",
        "bdaavg.minchannels=1",
        "bdaavg.frequencybase=0.0",
        "applycal.type=applycal",
        "applycal.correction=phase000",
        "applycal.slowgain.correction=amplitude000",
        "applycal.slowgain.solset=sol000",
        "applycal.fastphase.correction=phase000",
        "applycal.fastphase.solset=sol000",
        "applycal.fulljones.correction=fulljones",
        "applycal.fulljones.solset=sol000",
        "applycal.fulljones.soltab=[amplitude000,phase000]",
        "applycal.normalization.correction=amplitude000",
        "applycal.normalization.solset=sol000",
        "msout.storagemanager=Dysco",
        f"msin={options.msin}",
        f"msin.datacolumn={options.data_colname}",
        f"msout={options.msout}",
        f"msin.starttime={options.starttime}",
        f"msin.ntimes={options.ntimes}",
        f"shift.phasecenter={_strip_wrapping_shell_quotes(options.phasecenter)}",
        f"avg.freqstep={options.freqstep}",
        f"avg.timestep={options.timestep}",
    ]
    append_prefixed_value(command, "bdaavg.timebase=", options.timebase)
    append_prefixed_value(command, "bdaavg.maxinterval=", options.maxinterval)
    command.append(f"applybeam.direction={_strip_wrapping_shell_quotes(options.beamdir)}")
    append_prefixed_value(command, "applycal.parmdb=", options.h5parm)
    append_prefixed_value(command, "applycal.fulljones.parmdb=", options.fulljones_h5parm)
    append_prefixed_value(command, "applycal.normalization.parmdb=", options.normalize_h5parm)
    if options.central_patch_name is not None:
        command.append(f"applycal.direction=[{options.central_patch_name}]")
    command.extend([f"numthreads={options.num_threads}", f"steps={options.steps}"])
    append_prefixed_value(command, "applycal.steps=", options.applycal_steps)
    return command


def build_compress_sector_images_command(images: list[str]) -> list[str]:
    """Build the `fpack` command for sector image compression."""
    return ["fpack", *images]


def build_wsclean_restore_command(
    residual_image: str,
    source_list: str,
    output_image: str,
    numthreads: int,
) -> list[str]:
    """Build the WSClean command that restores a source list into an image."""
    return [
        "wsclean",
        "-j",
        str(numthreads),
        "-restore-list",
        residual_image,
        source_list,
        output_image,
    ]


def _wsclean_command_base() -> list[str]:
    return ["wsclean", "-no-update-model-required", "-local-rms", "-join-channels"]


def _wsclean_common_options(options: WscleanOptions) -> list[tuple[str, object]]:
    return [
        ("-name", options.name),
        ("-fits-mask", options.mask),
        ("-size", options.imsize),
        ("-niter", options.niter),
        ("-nmiter", options.nmiter),
        ("-minuv-l", options.min_uv_lambda),
        ("-maxuv-l", options.max_uv_lambda),
        ("-mgain", options.mgain),
        ("-pol", options.pol),
        ("-scale", options.cellsize_deg),
        ("-channels-out", options.channels_out),
        ("-deconvolution-channels", options.deconvolution_channels),
        ("-fit-spectral-pol", options.fit_spectral_pol),
        ("-taper-gaussian", options.taper_arcsec),
        ("-local-rms-strength", options.local_rms_strength),
        ("-local-rms-window", options.local_rms_window),
        ("-local-rms-method", options.local_rms_method),
        ("-abs-mem", options.memory_gb),
        ("-auto-mask", options.auto_mask),
        ("-auto-mask-nmiter", options.auto_mask_nmiter),
        ("-idg-mode", options.idg_mode),
        ("-j", options.num_threads),
        ("-deconvolution-threads", options.num_deconvolution_threads),
        ("-dd-psf-grid", options.dd_psf_grid),
    ]


def build_wsclean_no_dde_command(options: WscleanOptions) -> list[str]:
    """Build the serial no-DD WSClean command for one imaging sector."""
    command = _wsclean_command_base() + [
        "-apply-primary-beam",
        "-log-time",
        "-gridder",
        "wgridder",
        "-temp-dir",
        options.temp_dir,
        "-parallel-deconvolution",
        "2048",
        "-multiscale-scale-bias",
        "0.8",
        "-auto-threshold",
        "1.0",
        "-mgain-boosting",
        "1.3",
        "-weight",
        "briggs",
        str(options.robust),
    ]
    append_option_values(command, _wsclean_common_options(options))
    append_flag(command, "-multiscale", options.multiscale)
    append_flag(command, "-save-source-list", options.save_source_list)
    if options.link_polarizations:
        append_option_value(command, "-link-polarizations", options.link_polarizations)
    append_flag(command, "-join-polarizations", options.join_polarizations)
    append_flag(command, "-skip-final-iteration", options.skip_final_iteration)
    append_flag(command, "-apply-time-frequency-smearing", options.apply_time_frequency_smearing)
    command.append(options.msin)
    return command


def build_wsclean_facets_command(options: WscleanFacetOptions) -> list[str]:
    """Build the serial facet-corrected WSClean command for one imaging sector."""
    common = options.common
    command = _wsclean_command_base() + [
        "-apply-facet-beam",
        "-log-time",
        "-gridder",
        "wgridder",
        "-major-iteration-mode",
        "single",
        "-temp-dir",
        common.temp_dir,
        "-parallel-deconvolution",
        "2048",
        "-multiscale-scale-bias",
        "0.8",
        "-auto-threshold",
        "1.0",
        "-mgain-boosting",
        "1.3",
        "-facet-beam-update",
        "120",
        "-weight",
        "briggs",
        str(common.robust),
        "-apply-facet-solutions",
        options.h5parm,
        options.soltabs,
    ]
    option_values = [
        *_wsclean_common_options(common),
        ("-parallel-gridding", options.num_gridding_threads),
        ("-facet-regions", options.region_file),
    ]
    append_option_values(command, option_values)
    append_flag(command, "-multiscale", common.multiscale)
    append_flag(command, "-scalar-visibilities", options.scalar_visibilities)
    append_flag(command, "-diagonal-visibilities", options.diagonal_visibilities)
    append_flag(command, "-save-source-list", common.save_source_list)
    if common.link_polarizations:
        append_option_value(command, "-link-polarizations", common.link_polarizations)
    append_flag(command, "-join-polarizations", common.join_polarizations)
    append_flag(command, "-skip-final-iteration", common.skip_final_iteration)
    append_flag(command, "-apply-time-frequency-smearing", common.apply_time_frequency_smearing)
    append_flag(command, "-shared-facet-reads", options.shared_facet_reads)
    append_flag(command, "-shared-facet-writes", options.shared_facet_writes)
    command.append(common.msin)
    return command


def build_wsclean_screens_command(options: WscleanScreenOptions) -> list[str]:
    """Build the serial screen-corrected WSClean command for one imaging sector."""
    common = options.common
    command = _wsclean_command_base() + [
        "-gridder",
        "idg",
        "-major-iteration-mode",
        "single",
        "-log-time",
        "-temp-dir",
        common.temp_dir,
        "-multiscale-scale-bias",
        "0.8",
        "-parallel-deconvolution",
        "2048",
        "-auto-threshold",
        "1.0",
        "-mgain-boosting",
        "1.3",
        "-aterm-kernel-size",
        "32",
        "-aterm-config",
        options.aterm_config,
        "-weight",
        "briggs",
        str(common.robust),
    ]
    option_values = [
        *_wsclean_common_options(common),
        ("-interval", options.interval),
    ]
    append_option_values(command, option_values)
    append_flag(command, "-multiscale", common.multiscale)
    append_flag(command, "-save-source-list", common.save_source_list)
    if common.link_polarizations:
        append_option_value(command, "-link-polarizations", common.link_polarizations)
    append_flag(command, "-join-polarizations", common.join_polarizations)
    append_flag(command, "-skip-final-iteration", common.skip_final_iteration)
    append_flag(command, "-apply-time-frequency-smearing", common.apply_time_frequency_smearing)
    command.append(common.msin)
    return command


def build_mpi_wsclean_launch_command(mpi_nnodes: int) -> list[str]:
    """Build the MPI launcher prefix used for WSClean-MP."""
    if mpi_nnodes < 1:
        raise ValueError("mpi_nnodes must be >= 1")
    return [
        "mpirun",
        "--bind-to",
        "none",
        "-x",
        "OPENBLAS_NUM_THREADS",
        "-npernode",
        "1",
        "-np",
        str(mpi_nnodes),
        "wsclean-mp",
    ]


def _mpi_wsclean_command(wsclean_command: list[str], mpi_nnodes: int) -> list[str]:
    if not wsclean_command or wsclean_command[0] != "wsclean":
        raise ValueError("MPI WSClean wrapping requires a serial wsclean command")
    return build_mpi_wsclean_launch_command(mpi_nnodes) + wsclean_command[1:]


def build_wsclean_mpi_no_dde_command(mpi_nnodes: int, options: WscleanOptions) -> list[str]:
    """Build the MPI no-DD WSClean command for one imaging sector."""
    return _mpi_wsclean_command(build_wsclean_no_dde_command(options), mpi_nnodes)


def build_wsclean_mpi_facets_command(
    mpi_nnodes: int,
    options: WscleanFacetOptions,
) -> list[str]:
    """Build the MPI facet-corrected WSClean command for one imaging sector."""
    return _mpi_wsclean_command(
        build_wsclean_facets_command(replace(options, num_gridding_threads=None)),
        mpi_nnodes,
    )


def build_wsclean_mpi_screens_command(
    mpi_nnodes: int,
    options: WscleanScreenOptions,
) -> list[str]:
    """Build the MPI screen-corrected WSClean command for one imaging sector."""
    return _mpi_wsclean_command(build_wsclean_screens_command(options), mpi_nnodes)


def build_filter_skymodel_command(
    flat_noise_image: str,
    true_sky_image: str,
    true_sky_skymodel: str,
    apparent_sky_skymodel: str,
    output_root: str,
    vertices_file: str,
    beam_ms: list[str],
    threshisl: float,
    threshpix: float,
    filter_by_mask: bool,
    source_finder: str,
    ncores: int,
    bright_true_sky_skymodel: Optional[str] = None,
) -> list[str]:
    """Build the skymodel-filtering module command for one imaging sector."""
    command = python_module_command(
        FILTER_SKYMODEL_MODULE,
        flat_noise_image,
        true_sky_image,
        true_sky_skymodel,
        apparent_sky_skymodel,
        output_root,
        vertices_file,
        comma_join(beam_ms),
    )
    append_prefixed_value(command, "--bright_true_sky_skymodel=", bright_true_sky_skymodel)
    command.extend(
        [
            f"--threshisl={threshisl}",
            f"--threshpix={threshpix}",
            f"--filter_by_mask={bool_token(filter_by_mask)}",
            f"--source_finder={source_finder}",
            f"--ncores={ncores}",
        ]
    )
    return command


def build_make_catalog_from_image_cube_command(
    cube_image: str,
    cube_beams: str,
    cube_frequencies: str,
    output_catalog: str,
    threshisl: float,
    threshpix: float,
    ncores: int,
) -> list[str]:
    """Build the image-cube catalog generation module command."""
    return python_module_command(
        CUBE_CATALOG_MODULE,
        cube_image,
        cube_beams,
        cube_frequencies,
        output_catalog,
        f"--threshisl={threshisl}",
        f"--threshpix={threshpix}",
        f"--ncores={ncores}",
    )

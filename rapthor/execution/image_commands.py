"""Command builders for the Image execution flow."""

from typing import Optional

from rapthor.execution.commands import (
    append_flag,
    append_option_value,
    append_option_values,
    append_prefixed_value,
    bool_token,
    comma_join,
    normalize_command,
)

ATERM_CONFIG_FILENAME = "aterm_plus_beam.cfg"


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


def build_prepare_imaging_data_command(
    msin: str,
    data_colname: str,
    msout: str,
    starttime: str,
    ntimes: int,
    phasecenter: str,
    freqstep: int,
    timestep: int,
    beamdir: str,
    numthreads: int,
    steps: str,
    maxinterval: Optional[int] = None,
    timebase: Optional[float] = None,
    h5parm: Optional[str] = None,
    fulljones_h5parm: Optional[str] = None,
    normalize_h5parm: Optional[str] = None,
    central_patch_name: Optional[str] = None,
    applycal_steps: Optional[str] = None,
) -> list[str]:
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
        f"msin={msin}",
        f"msin.datacolumn={data_colname}",
        f"msout={msout}",
        f"msin.starttime={starttime}",
        f"msin.ntimes={ntimes}",
        f"shift.phasecenter={_strip_wrapping_shell_quotes(phasecenter)}",
        f"avg.freqstep={freqstep}",
        f"avg.timestep={timestep}",
    ]
    append_prefixed_value(command, "bdaavg.timebase=", timebase)
    append_prefixed_value(command, "bdaavg.maxinterval=", maxinterval)
    command.append(f"applybeam.direction={_strip_wrapping_shell_quotes(beamdir)}")
    append_prefixed_value(command, "applycal.parmdb=", h5parm)
    append_prefixed_value(command, "applycal.fulljones.parmdb=", fulljones_h5parm)
    append_prefixed_value(command, "applycal.normalization.parmdb=", normalize_h5parm)
    if central_patch_name is not None:
        command.append(f"applycal.direction=[{central_patch_name}]")
    command.extend([f"numthreads={numthreads}", f"steps={steps}"])
    append_prefixed_value(command, "applycal.steps=", applycal_steps)
    return command


def build_concat_time_command(
    input_filenames: list[str],
    output_filename: str,
    data_colname: str,
) -> list[str]:
    """Build the `concat_ms.py` command for one imaging sector."""
    return [
        "concat_ms.py",
        *input_filenames,
        f"--msout={output_filename}",
        "--concat_property=time",
        f"--data_colname={data_colname}",
    ]


def build_blank_image_command(
    mask_filename: str,
    wsclean_imsize: list[int],
    vertices_file: str,
    ra: float,
    dec: float,
    cellsize_deg: float,
    image_filename: Optional[str] = None,
    region_file: Optional[str] = None,
) -> list[str]:
    """Build the `blank_image.py` command for one imaging sector."""
    command = ["blank_image.py", mask_filename]
    if image_filename is not None:
        command.append(image_filename)
    command.extend(
        [
            f"--imsize={wsclean_imsize[0]},{wsclean_imsize[1]}",
            f"--vertices_file={vertices_file}",
            f"--reference_ra_deg={ra}",
            f"--reference_dec_deg={dec}",
            f"--cellsize_deg={cellsize_deg}",
        ]
    )
    append_prefixed_value(command, "--region_file=", region_file)
    return command


def build_make_region_file_command(
    skymodel: str,
    ra_mid: float,
    dec_mid: float,
    width_ra: float,
    width_dec: float,
    outfile: str,
    enclose_names: bool = True,
) -> list[str]:
    """Build the `make_region_file.py` command for facet imaging."""
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


def build_compress_sector_images_command(images: list[str]) -> list[str]:
    """Build the `fpack` command for sector image compression."""
    return ["fpack", *images]


def build_make_skymodel_image_command(
    source_catalog: str,
    reference_image: str,
    output_image_name: str,
) -> list[str]:
    """Build the `restore_skymodel.py` command for filtered-model images."""
    return ["restore_skymodel.py", source_catalog, reference_image, output_image_name]


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


def build_make_image_cube_command(input_image_list: list[str], output_image: str) -> list[str]:
    """Build the `make_image_cube.py` command for one Stokes image cube."""
    return ["make_image_cube.py", comma_join(input_image_list), output_image]


def build_make_catalog_from_image_cube_command(
    cube: str,
    cube_beams: str,
    cube_frequencies: str,
    output_catalog: str,
    threshisl: float,
    threshpix: float,
    ncores: int,
) -> list[str]:
    """Build the source-catalog command for a Stokes-I image cube."""
    return [
        "make_catalog_from_image_cube.py",
        cube,
        cube_beams,
        cube_frequencies,
        output_catalog,
        f"--threshisl={threshisl}",
        f"--threshpix={threshpix}",
        f"--ncores={ncores}",
    ]


def build_normalize_flux_scale_command(
    source_catalog: str,
    ms_file: str,
    normalize_h5parm: str,
) -> list[str]:
    """Build the flux-scale normalization command."""
    return ["normalize_flux_scale.py", source_catalog, ms_file, normalize_h5parm]


def build_wsclean_no_dde_command(
    msin: str,
    name: str,
    mask: str,
    wsclean_imsize: list[int],
    wsclean_niter: int,
    wsclean_nmiter: int,
    robust: float,
    min_uv_lambda: float,
    max_uv_lambda: float,
    mgain: float,
    multiscale: bool,
    save_source_list: bool,
    pol: str,
    link_polarizations: object,
    join_polarizations: bool,
    skip_final_iteration: bool,
    cellsize_deg: float,
    channels_out: int,
    deconvolution_channels: int,
    fit_spectral_pol: int,
    taper_arcsec: float,
    local_rms_strength: float,
    local_rms_window: float,
    local_rms_method: str,
    wsclean_mem: float,
    auto_mask: float,
    auto_mask_nmiter: int,
    idg_mode: str,
    num_threads: int,
    num_deconvolution_threads: int,
    dd_psf_grid: list[int],
    apply_time_frequency_smearing: bool,
    temp_dir: str,
) -> list[str]:
    """Build the serial no-DDE WSClean command for one imaging sector."""
    command = [
        "wsclean",
        "-no-update-model-required",
        "-local-rms",
        "-join-channels",
        "-apply-primary-beam",
        "-log-time",
        "-gridder",
        "wgridder",
        "-temp-dir",
        temp_dir,
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
        str(robust),
    ]
    options = [
        ("-name", name),
        ("-fits-mask", mask),
        ("-size", wsclean_imsize),
        ("-niter", wsclean_niter),
        ("-nmiter", wsclean_nmiter),
        ("-minuv-l", min_uv_lambda),
        ("-maxuv-l", max_uv_lambda),
        ("-mgain", mgain),
        ("-pol", pol),
        ("-scale", cellsize_deg),
        ("-channels-out", channels_out),
        ("-deconvolution-channels", deconvolution_channels),
        ("-fit-spectral-pol", fit_spectral_pol),
        ("-taper-gaussian", taper_arcsec),
        ("-local-rms-strength", local_rms_strength),
        ("-local-rms-window", local_rms_window),
        ("-local-rms-method", local_rms_method),
        ("-abs-mem", wsclean_mem),
        ("-auto-mask", auto_mask),
        ("-auto-mask-nmiter", auto_mask_nmiter),
        ("-idg-mode", idg_mode),
        ("-j", num_threads),
        ("-deconvolution-threads", num_deconvolution_threads),
        ("-dd-psf-grid", dd_psf_grid),
    ]
    append_option_values(command, options)
    append_flag(command, "-multiscale", multiscale)
    append_flag(command, "-save-source-list", save_source_list)
    if link_polarizations:
        append_option_value(command, "-link-polarizations", link_polarizations)
    append_flag(command, "-join-polarizations", join_polarizations)
    append_flag(command, "-skip-final-iteration", skip_final_iteration)
    append_flag(command, "-apply-time-frequency-smearing", apply_time_frequency_smearing)
    command.append(msin)
    return command


def build_wsclean_facets_command(
    msin: str,
    name: str,
    mask: str,
    wsclean_imsize: list[int],
    wsclean_niter: int,
    wsclean_nmiter: int,
    robust: float,
    min_uv_lambda: float,
    max_uv_lambda: float,
    mgain: float,
    multiscale: bool,
    scalar_visibilities: bool,
    diagonal_visibilities: bool,
    save_source_list: bool,
    pol: str,
    link_polarizations: object,
    join_polarizations: bool,
    skip_final_iteration: bool,
    cellsize_deg: float,
    channels_out: int,
    deconvolution_channels: int,
    fit_spectral_pol: int,
    taper_arcsec: float,
    local_rms_strength: float,
    local_rms_window: float,
    local_rms_method: str,
    wsclean_mem: float,
    auto_mask: float,
    auto_mask_nmiter: int,
    idg_mode: str,
    num_threads: int,
    num_deconvolution_threads: int,
    dd_psf_grid: list[int],
    h5parm: str,
    soltabs: str,
    region_file: str,
    num_gridding_threads: int,
    apply_time_frequency_smearing: bool,
    shared_facet_reads: bool,
    shared_facet_writes: bool,
    temp_dir: str,
) -> list[str]:
    """Build the serial facet-corrected WSClean command for one imaging sector."""
    command = [
        "wsclean",
        "-no-update-model-required",
        "-local-rms",
        "-join-channels",
        "-apply-facet-beam",
        "-log-time",
        "-gridder",
        "wgridder",
        "-major-iteration-mode",
        "single",
        "-temp-dir",
        temp_dir,
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
        str(robust),
        "-apply-facet-solutions",
        h5parm,
        soltabs,
    ]
    options = [
        ("-name", name),
        ("-fits-mask", mask),
        ("-size", wsclean_imsize),
        ("-niter", wsclean_niter),
        ("-nmiter", wsclean_nmiter),
        ("-minuv-l", min_uv_lambda),
        ("-maxuv-l", max_uv_lambda),
        ("-mgain", mgain),
        ("-pol", pol),
        ("-scale", cellsize_deg),
        ("-channels-out", channels_out),
        ("-deconvolution-channels", deconvolution_channels),
        ("-fit-spectral-pol", fit_spectral_pol),
        ("-taper-gaussian", taper_arcsec),
        ("-local-rms-strength", local_rms_strength),
        ("-local-rms-window", local_rms_window),
        ("-local-rms-method", local_rms_method),
        ("-abs-mem", wsclean_mem),
        ("-auto-mask", auto_mask),
        ("-auto-mask-nmiter", auto_mask_nmiter),
        ("-idg-mode", idg_mode),
        ("-j", num_threads),
        ("-deconvolution-threads", num_deconvolution_threads),
        ("-dd-psf-grid", dd_psf_grid),
        ("-parallel-gridding", num_gridding_threads),
        ("-facet-regions", region_file),
    ]
    append_option_values(command, options)
    append_flag(command, "-multiscale", multiscale)
    append_flag(command, "-scalar-visibilities", scalar_visibilities)
    append_flag(command, "-diagonal-visibilities", diagonal_visibilities)
    append_flag(command, "-save-source-list", save_source_list)
    if link_polarizations:
        append_option_value(command, "-link-polarizations", link_polarizations)
    append_flag(command, "-join-polarizations", join_polarizations)
    append_flag(command, "-skip-final-iteration", skip_final_iteration)
    append_flag(command, "-apply-time-frequency-smearing", apply_time_frequency_smearing)
    append_flag(command, "-shared-facet-reads", shared_facet_reads)
    append_flag(command, "-shared-facet-writes", shared_facet_writes)
    command.append(msin)
    return command


def build_wsclean_screens_command(
    msin: str,
    name: str,
    mask: str,
    wsclean_imsize: list[int],
    wsclean_niter: int,
    wsclean_nmiter: int,
    robust: float,
    min_uv_lambda: float,
    max_uv_lambda: float,
    mgain: float,
    multiscale: bool,
    save_source_list: bool,
    pol: str,
    link_polarizations: object,
    join_polarizations: bool,
    skip_final_iteration: bool,
    cellsize_deg: float,
    channels_out: int,
    deconvolution_channels: int,
    fit_spectral_pol: int,
    taper_arcsec: float,
    local_rms_strength: float,
    local_rms_window: float,
    local_rms_method: str,
    wsclean_mem: float,
    auto_mask: float,
    auto_mask_nmiter: int,
    idg_mode: str,
    num_threads: int,
    num_deconvolution_threads: int,
    dd_psf_grid: list[int],
    interval: list[int],
    apply_time_frequency_smearing: bool,
    temp_dir: str,
    aterm_config: str = ATERM_CONFIG_FILENAME,
) -> list[str]:
    """Build the serial screen-corrected WSClean command for one imaging sector."""
    command = [
        "wsclean",
        "-no-update-model-required",
        "-local-rms",
        "-join-channels",
        "-gridder",
        "idg",
        "-major-iteration-mode",
        "single",
        "-log-time",
        "-temp-dir",
        temp_dir,
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
        aterm_config,
        "-weight",
        "briggs",
        str(robust),
    ]
    options = [
        ("-name", name),
        ("-fits-mask", mask),
        ("-size", wsclean_imsize),
        ("-niter", wsclean_niter),
        ("-nmiter", wsclean_nmiter),
        ("-minuv-l", min_uv_lambda),
        ("-maxuv-l", max_uv_lambda),
        ("-mgain", mgain),
        ("-pol", pol),
        ("-scale", cellsize_deg),
        ("-channels-out", channels_out),
        ("-deconvolution-channels", deconvolution_channels),
        ("-fit-spectral-pol", fit_spectral_pol),
        ("-taper-gaussian", taper_arcsec),
        ("-local-rms-strength", local_rms_strength),
        ("-local-rms-window", local_rms_window),
        ("-local-rms-method", local_rms_method),
        ("-abs-mem", wsclean_mem),
        ("-auto-mask", auto_mask),
        ("-auto-mask-nmiter", auto_mask_nmiter),
        ("-idg-mode", idg_mode),
        ("-j", num_threads),
        ("-deconvolution-threads", num_deconvolution_threads),
        ("-dd-psf-grid", dd_psf_grid),
        ("-interval", interval),
    ]
    append_option_values(command, options)
    append_flag(command, "-multiscale", multiscale)
    append_flag(command, "-save-source-list", save_source_list)
    if link_polarizations:
        append_option_value(command, "-link-polarizations", link_polarizations)
    append_flag(command, "-join-polarizations", join_polarizations)
    append_flag(command, "-skip-final-iteration", skip_final_iteration)
    append_flag(command, "-apply-time-frequency-smearing", apply_time_frequency_smearing)
    command.append(msin)
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


def build_wsclean_mpi_no_dde_command(mpi_nnodes: int, **kwargs) -> list[str]:
    """Build the MPI no-DDE WSClean command for one imaging sector."""
    return _mpi_wsclean_command(build_wsclean_no_dde_command(**kwargs), mpi_nnodes)


def build_wsclean_mpi_facets_command(mpi_nnodes: int, **kwargs) -> list[str]:
    """Build the MPI facet-corrected WSClean command for one imaging sector."""
    kwargs = dict(kwargs)
    kwargs["num_gridding_threads"] = None
    return _mpi_wsclean_command(build_wsclean_facets_command(**kwargs), mpi_nnodes)


def build_wsclean_mpi_screens_command(mpi_nnodes: int, **kwargs) -> list[str]:
    """Build the MPI screen-corrected WSClean command for one imaging sector."""
    return _mpi_wsclean_command(build_wsclean_screens_command(**kwargs), mpi_nnodes)


def build_check_image_beam_command(input_image: str, beam_size_arcsec: float) -> list[str]:
    """Build the `check_image_beam.py` command for one image."""
    return ["check_image_beam.py", input_image, str(beam_size_arcsec)]


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
    """Build the `filter_skymodel.py` command for one imaging sector."""
    command = [
        "filter_skymodel.py",
        flat_noise_image,
        true_sky_image,
        true_sky_skymodel,
        apparent_sky_skymodel,
        output_root,
        vertices_file,
        comma_join(beam_ms),
    ]
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


def build_calculate_image_diagnostics_command(
    flat_noise_image: str,
    flat_noise_rms_image: str,
    true_sky_image: str,
    true_sky_rms_image: str,
    input_catalog: str,
    obs_ms: list[str],
    obs_starttime: list[str],
    obs_ntimes: list[int],
    diagnostics_file: str,
    output_root: str,
    allow_internet_access: bool,
    facet_region_file: Optional[str] = None,
    photometry_skymodel: Optional[str] = None,
    astrometry_skymodel: Optional[str] = None,
) -> list[str]:
    """Build the `calculate_image_diagnostics.py` command for one sector."""
    command = [
        "calculate_image_diagnostics.py",
        flat_noise_image,
        flat_noise_rms_image,
        true_sky_image,
        true_sky_rms_image,
        input_catalog,
        comma_join(obs_ms),
        comma_join(obs_starttime),
        comma_join(obs_ntimes),
        diagnostics_file,
        output_root,
    ]
    append_prefixed_value(command, "--facet_region_file=", facet_region_file or "none")
    append_prefixed_value(command, "--photometry_comparison_skymodel=", photometry_skymodel)
    append_prefixed_value(command, "--astrometry_comparison_skymodel=", astrometry_skymodel)
    append_flag(command, "--allow_internet_access", allow_internet_access)
    return command


def normalized_prepare_imaging_data_command(**kwargs) -> list[str]:
    """Return normalized DP3 prepare-imaging command tokens for fixture comparisons."""
    return normalize_command(build_prepare_imaging_data_command(**kwargs))


def normalized_concat_time_command(**kwargs) -> list[str]:
    """Return normalized concat-time command tokens for fixture comparisons."""
    return normalize_command(build_concat_time_command(**kwargs))


def normalized_blank_image_command(**kwargs) -> list[str]:
    """Return normalized blank-image command tokens for fixture comparisons."""
    return normalize_command(build_blank_image_command(**kwargs))


def normalized_compress_sector_images_command(**kwargs) -> list[str]:
    """Return normalized sector-image compression command tokens for fixture comparisons."""
    return normalize_command(build_compress_sector_images_command(**kwargs))


def normalized_make_skymodel_image_command(**kwargs) -> list[str]:
    """Return normalized make-skymodel-image command tokens for fixture comparisons."""
    return normalize_command(build_make_skymodel_image_command(**kwargs))


def normalized_wsclean_restore_command(**kwargs) -> list[str]:
    """Return normalized WSClean restore command tokens for fixture comparisons."""
    return normalize_command(build_wsclean_restore_command(**kwargs))


def normalized_make_image_cube_command(**kwargs) -> list[str]:
    """Return normalized make-image-cube command tokens for fixture comparisons."""
    return normalize_command(build_make_image_cube_command(**kwargs))


def normalized_make_catalog_from_image_cube_command(**kwargs) -> list[str]:
    """Return normalized image-cube catalog command tokens for fixture comparisons."""
    return normalize_command(build_make_catalog_from_image_cube_command(**kwargs))


def normalized_normalize_flux_scale_command(**kwargs) -> list[str]:
    """Return normalized flux-scale normalization command tokens for fixture comparisons."""
    return normalize_command(build_normalize_flux_scale_command(**kwargs))


def normalized_make_region_file_command(**kwargs) -> list[str]:
    """Return normalized make-region-file command tokens for fixture comparisons."""
    return normalize_command(build_make_region_file_command(**kwargs))


def normalized_wsclean_no_dde_command(**kwargs) -> list[str]:
    """Return normalized no-DDE WSClean command tokens for fixture comparisons."""
    return normalize_command(build_wsclean_no_dde_command(**kwargs))


def normalized_wsclean_mpi_no_dde_command(**kwargs) -> list[str]:
    """Return normalized MPI no-DDE WSClean command tokens for fixture comparisons."""
    return normalize_command(build_wsclean_mpi_no_dde_command(**kwargs))


def normalized_wsclean_facets_command(**kwargs) -> list[str]:
    """Return normalized facet WSClean command tokens for fixture comparisons."""
    return normalize_command(build_wsclean_facets_command(**kwargs))


def normalized_wsclean_mpi_facets_command(**kwargs) -> list[str]:
    """Return normalized MPI facet WSClean command tokens for fixture comparisons."""
    return normalize_command(build_wsclean_mpi_facets_command(**kwargs))


def normalized_wsclean_screens_command(**kwargs) -> list[str]:
    """Return normalized screen WSClean command tokens for fixture comparisons."""
    return normalize_command(build_wsclean_screens_command(**kwargs))


def normalized_wsclean_mpi_screens_command(**kwargs) -> list[str]:
    """Return normalized MPI screen WSClean command tokens for fixture comparisons."""
    return normalize_command(build_wsclean_mpi_screens_command(**kwargs))

"""Prefect flows for imaging."""

import glob
import os
import shutil
from typing import Mapping, Optional

from prefect import flow, task

from rapthor.execution.artifacts import publish_fits_image_artifacts, publish_plot_file_records
from rapthor.execution.commands import normalize_command
from rapthor.execution.config import ExecutionConfig
from rapthor.execution.flows.runtime import run_flow_with_task_runner
from rapthor.execution.outputs import directory_record, file_record, validate_output_record
from rapthor.execution.payloads import assert_serializable_payload
from rapthor.execution.prefect_logging import publish_python_logs_to_prefect
from rapthor.execution.resources import (
    ResourceRequest,
    thread_environment,
    validate_resource_request,
)
from rapthor.execution.shell import ShellCommand, run_shell_command

ATERM_CONFIG_FILENAME = "aterm_plus_beam.cfg"


def _bool_token(value: bool) -> str:
    return "True" if value else "False"


def _path_record_path(record: object, path_class: str) -> str:
    if isinstance(record, Mapping) and record.get("class") == path_class:
        path = record.get("path")
        if isinstance(path, str) and path:
            return path
    raise ValueError(f"Expected a {path_class} output record, got {record!r}")


def _optional_path_record_path(record: object, path_class: str) -> Optional[str]:
    if record is None:
        return None
    return _path_record_path(record, path_class)


def _optional_file_record_path(record: object) -> Optional[str]:
    return _optional_path_record_path(record, "File")


def _directory_record_path(record: object) -> str:
    return _path_record_path(record, "Directory")


def _file_record_path(record: object) -> str:
    return _path_record_path(record, "File")


def _validate_basename(filename: object, name: str) -> str:
    if not isinstance(filename, str) or not filename:
        raise ValueError(f"{name} must be a non-empty string")
    if os.path.isabs(filename) or os.path.basename(filename) != filename:
        raise ValueError(f"{name} must be a basename")
    return filename


def _join_comma(values: list[object]) -> str:
    return ",".join(str(value) for value in values)


def _pol_token(pol: object) -> str:
    if isinstance(pol, str):
        return pol
    if isinstance(pol, list):
        return "".join(str(value) for value in pol)
    raise ValueError("pol must be a string or list")


def _is_stokes_i(pol: str) -> bool:
    return pol.upper() == "I"


def _append_optional_prefixed(command: list[str], prefix: str, value: Optional[object]) -> None:
    if value is not None:
        command.append(f"{prefix}{value}")


def _append_option(command: list[str], option: str, value: object) -> None:
    command.extend([option, str(value)])


def _append_flag(command: list[str], option: str, enabled: bool) -> None:
    if enabled:
        command.append(option)


def _append_options(command: list[str], options: list[tuple[str, object]]) -> None:
    for option, value in options:
        if value is None:
            continue
        if isinstance(value, list):
            command.append(option)
            command.extend(str(item) for item in value)
        else:
            _append_option(command, option, value)


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
    _append_optional_prefixed(command, "bdaavg.timebase=", timebase)
    _append_optional_prefixed(command, "bdaavg.maxinterval=", maxinterval)
    command.append(f"applybeam.direction={_strip_wrapping_shell_quotes(beamdir)}")
    _append_optional_prefixed(command, "applycal.parmdb=", h5parm)
    _append_optional_prefixed(command, "applycal.fulljones.parmdb=", fulljones_h5parm)
    _append_optional_prefixed(command, "applycal.normalization.parmdb=", normalize_h5parm)
    if central_patch_name is not None:
        command.append(f"applycal.direction=[{central_patch_name}]")
    command.extend([f"numthreads={numthreads}", f"steps={steps}"])
    _append_optional_prefixed(command, "applycal.steps=", applycal_steps)
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
    _append_optional_prefixed(command, "--region_file=", region_file)
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
        f"--enclose_names={_bool_token(enclose_names)}",
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
    return ["make_image_cube.py", _join_comma(input_image_list), output_image]


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
    _append_options(command, options)
    _append_flag(command, "-multiscale", multiscale)
    _append_flag(command, "-save-source-list", save_source_list)
    if link_polarizations:
        _append_option(command, "-link-polarizations", link_polarizations)
    _append_flag(command, "-join-polarizations", join_polarizations)
    _append_flag(command, "-skip-final-iteration", skip_final_iteration)
    _append_flag(command, "-apply-time-frequency-smearing", apply_time_frequency_smearing)
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
    _append_options(command, options)
    _append_flag(command, "-multiscale", multiscale)
    _append_flag(command, "-scalar-visibilities", scalar_visibilities)
    _append_flag(command, "-diagonal-visibilities", diagonal_visibilities)
    _append_flag(command, "-save-source-list", save_source_list)
    if link_polarizations:
        _append_option(command, "-link-polarizations", link_polarizations)
    _append_flag(command, "-join-polarizations", join_polarizations)
    _append_flag(command, "-skip-final-iteration", skip_final_iteration)
    _append_flag(command, "-apply-time-frequency-smearing", apply_time_frequency_smearing)
    _append_flag(command, "-shared-facet-reads", shared_facet_reads)
    _append_flag(command, "-shared-facet-writes", shared_facet_writes)
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
    _append_options(command, options)
    _append_flag(command, "-multiscale", multiscale)
    _append_flag(command, "-save-source-list", save_source_list)
    if link_polarizations:
        _append_option(command, "-link-polarizations", link_polarizations)
    _append_flag(command, "-join-polarizations", join_polarizations)
    _append_flag(command, "-skip-final-iteration", skip_final_iteration)
    _append_flag(command, "-apply-time-frequency-smearing", apply_time_frequency_smearing)
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
        _join_comma(beam_ms),
    ]
    _append_optional_prefixed(command, "--bright_true_sky_skymodel=", bright_true_sky_skymodel)
    command.extend(
        [
            f"--threshisl={threshisl}",
            f"--threshpix={threshpix}",
            f"--filter_by_mask={_bool_token(filter_by_mask)}",
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
        _join_comma(obs_ms),
        _join_comma(obs_starttime),
        _join_comma(obs_ntimes),
        diagnostics_file,
        output_root,
    ]
    _append_optional_prefixed(command, "--facet_region_file=", facet_region_file or "none")
    _append_optional_prefixed(command, "--photometry_comparison_skymodel=", photometry_skymodel)
    _append_optional_prefixed(command, "--astrometry_comparison_skymodel=", astrometry_skymodel)
    _append_flag(command, "--allow_internet_access", allow_internet_access)
    return command


def image_payload_from_inputs(
    input_parms: Mapping[str, object],
    pipeline_working_dir: object,
    *,
    apply_screens: bool = False,
    use_facets: bool = False,
    compress_images: bool = False,
    make_image_cube: bool = False,
    normalize_flux_scale: bool = False,
    use_mpi: bool = False,
) -> dict:
    """Create a serializable Image flow payload."""
    if apply_screens and use_facets:
        raise ValueError("apply_screens and use_facets cannot both be enabled")
    if normalize_flux_scale and not make_image_cube:
        raise ValueError("normalize_flux_scale requires make_image_cube=True")

    pol = _pol_token(input_parms["pol"])
    peel_bright_sources = bool(input_parms.get("peel_bright_sources", False))
    bright_skymodel_pb = _optional_file_record_path(input_parms.get("bright_skymodel_pb"))
    if peel_bright_sources and bright_skymodel_pb is None:
        raise ValueError("bright_skymodel_pb must be a File record when peel_bright_sources=True")

    pipeline_dir = str(pipeline_working_dir)
    image_names = input_parms.get("image_name", [])
    if not isinstance(image_names, list):
        raise ValueError("image_name must be a list")
    sector_count = len(image_names)
    save_filtered_model_image = bool(input_parms.get("save_filtered_model_image"))
    per_sector_keys = [
        "obs_filename",
        "prepare_filename",
        "concat_filename",
        "previous_mask_filename",
        "mask_filename",
        "starttime",
        "ntimes",
        "image_freqstep",
        "image_timestep",
        "image_maxinterval",
        "image_timebase",
        "phasecenter",
        "channels_out",
        "deconvolution_channels",
        "fit_spectral_pol",
        "ra",
        "dec",
        "wsclean_imsize",
        "vertices_file",
        "region_file",
        "wsclean_niter",
        "wsclean_nmiter",
        "robust",
        "cellsize_deg",
        "min_uv_lambda",
        "max_uv_lambda",
        "mgain",
        "taper_arcsec",
        "local_rms_strength",
        "local_rms_window",
        "local_rms_method",
        "auto_mask",
        "auto_mask_nmiter",
        "idg_mode",
        "wsclean_mem",
        "threshisl",
        "threshpix",
        "do_multiscale",
        "dd_psf_grid",
    ]
    if use_facets:
        per_sector_keys.extend(
            [
                "ra_mid",
                "dec_mid",
                "width_ra",
                "width_dec",
                "facet_region_file",
            ]
        )
    if save_filtered_model_image:
        per_sector_keys.append("filtered_model_image_name")
    if make_image_cube:
        per_sector_keys.append("image_I_cube_name")
    if normalize_flux_scale:
        per_sector_keys.extend(["output_source_catalog", "output_normalize_h5parm"])
    if use_mpi:
        per_sector_keys.extend(["mpi_nnodes", "mpi_cpus_per_task"])
    for key in per_sector_keys:
        value = input_parms.get(key, [])
        if not isinstance(value, list) or len(value) != sector_count:
            raise ValueError(f"{key} must be a list with one value per sector")

    h5parm = _optional_file_record_path(input_parms.get("h5parm"))
    if (apply_screens or use_facets) and h5parm is None:
        raise ValueError("h5parm must be a File record for screen or facet imaging")
    interval = None
    if apply_screens:
        interval_value = input_parms.get("interval")
        if (
            not isinstance(interval_value, list)
            or len(interval_value) != 2
            or not all(isinstance(value, int) for value in interval_value)
        ):
            raise ValueError("interval must be a two-element integer list when apply_screens=True")
        interval = [int(value) for value in interval_value]
    facet_skymodel = None
    if use_facets:
        facet_skymodel = _file_record_path(input_parms.get("skymodel"))
        for key in [
            "soltabs",
            "parallel_gridding_threads",
            "scalar_visibilities",
            "diagonal_visibilities",
            "shared_facet_rw",
        ]:
            if key not in input_parms:
                raise ValueError(f"{key} is required when use_facets=True")
    fulljones_h5parm = _optional_file_record_path(input_parms.get("fulljones_h5parm"))
    input_normalize_h5parm = _optional_file_record_path(input_parms.get("input_normalize_h5parm"))
    photometry_skymodel = _optional_file_record_path(input_parms.get("photometry_skymodel"))
    astrometry_skymodel = _optional_file_record_path(input_parms.get("astrometry_skymodel"))

    sectors = []
    for sector_index in range(sector_count):
        obs_records = input_parms["obs_filename"][sector_index]
        prepare_filenames = input_parms["prepare_filename"][sector_index]
        starttimes = input_parms["starttime"][sector_index]
        ntimes = input_parms["ntimes"][sector_index]
        freqsteps = input_parms["image_freqstep"][sector_index]
        timesteps = input_parms["image_timestep"][sector_index]
        maxintervals = input_parms["image_maxinterval"][sector_index]
        obs_inputs = [
            obs_records,
            prepare_filenames,
            starttimes,
            ntimes,
            freqsteps,
            timesteps,
            maxintervals,
        ]
        if not all(isinstance(value, list) for value in obs_inputs):
            raise ValueError(f"sector {sector_index} observation inputs must be lists")
        obs_count = len(obs_records)
        if any(len(value) != obs_count for value in obs_inputs):
            raise ValueError(f"sector {sector_index} observation inputs must have the same length")

        prepare_tasks = []
        for obs_index in range(obs_count):
            msout = _validate_basename(
                prepare_filenames[obs_index], f"prepare_filename[{sector_index}][{obs_index}]"
            )
            prepare_tasks.append(
                {
                    "msin": _directory_record_path(obs_records[obs_index]),
                    "msout": msout,
                    "msout_path": os.path.join(pipeline_dir, msout),
                    "starttime": str(starttimes[obs_index]),
                    "ntimes": int(ntimes[obs_index]),
                    "freqstep": int(freqsteps[obs_index]),
                    "timestep": int(timesteps[obs_index]),
                    "maxinterval": (
                        None if maxintervals[obs_index] is None else int(maxintervals[obs_index])
                    ),
                }
            )

        image_name = _validate_basename(image_names[sector_index], f"image_name[{sector_index}]")
        concat_filename = _validate_basename(
            input_parms["concat_filename"][sector_index], f"concat_filename[{sector_index}]"
        )
        mask_filename = _validate_basename(
            input_parms["mask_filename"][sector_index], f"mask_filename[{sector_index}]"
        )
        facet_region_filename = None
        if use_facets:
            facet_region_filename = _validate_basename(
                input_parms["facet_region_file"][sector_index],
                f"facet_region_file[{sector_index}]",
            )
        filtered_model_image_filename = None
        if save_filtered_model_image:
            filtered_model_image_filename = _validate_basename(
                input_parms["filtered_model_image_name"][sector_index],
                f"filtered_model_image_name[{sector_index}]",
            )
        image_i_cube_filename = None
        image_cube_specs = []
        if make_image_cube:
            image_i_cube_filename = _validate_basename(
                input_parms["image_I_cube_name"][sector_index],
                f"image_I_cube_name[{sector_index}]",
            )
            for stokes in pol.upper():
                key = f"image_{stokes}_cube_name"
                if key not in input_parms:
                    continue
                image_cube_filename = _validate_basename(
                    input_parms[key][sector_index],
                    f"{key}[{sector_index}]",
                )
                image_cube_specs.append(
                    {
                        "pol": stokes,
                        "filename": image_cube_filename,
                        "path": os.path.join(pipeline_dir, image_cube_filename),
                    }
                )
            if not image_cube_specs:
                image_cube_specs.append(
                    {
                        "pol": "I",
                        "filename": image_i_cube_filename,
                        "path": os.path.join(pipeline_dir, image_i_cube_filename),
                    }
                )
        output_source_catalog_filename = None
        output_normalize_h5parm_filename = None
        if normalize_flux_scale:
            output_source_catalog_filename = _validate_basename(
                input_parms["output_source_catalog"][sector_index],
                f"output_source_catalog[{sector_index}]",
            )
            output_normalize_h5parm_filename = _validate_basename(
                input_parms["output_normalize_h5parm"][sector_index],
                f"output_normalize_h5parm[{sector_index}]",
            )
        sectors.append(
            {
                "image_name": image_name,
                "apply_screens": apply_screens,
                "use_facets": use_facets,
                "use_mpi": bool(use_mpi),
                "compress_images": bool(compress_images),
                "make_image_cube": bool(make_image_cube),
                "normalize_flux_scale": bool(normalize_flux_scale),
                "peel_bright_sources": peel_bright_sources,
                "save_filtered_model_image": save_filtered_model_image,
                "bright_skymodel_pb": bright_skymodel_pb,
                "data_colname": str(input_parms["data_colname"]),
                "prepare_tasks": prepare_tasks,
                "concat_filename": concat_filename,
                "concat_path": os.path.join(pipeline_dir, concat_filename),
                "previous_mask_filename": _optional_file_record_path(
                    input_parms["previous_mask_filename"][sector_index]
                ),
                "mask_filename": mask_filename,
                "mask_path": os.path.join(pipeline_dir, mask_filename),
                "timebase": input_parms["image_timebase"][sector_index],
                "phasecenter": str(input_parms["phasecenter"][sector_index]),
                "h5parm": h5parm,
                "fulljones_h5parm": fulljones_h5parm,
                "input_normalize_h5parm": input_normalize_h5parm,
                "prepare_data_steps": str(input_parms["prepare_data_steps"]),
                "prepare_data_applycal_steps": input_parms.get("prepare_data_applycal_steps"),
                "central_patch_name": (
                    input_parms.get("central_patch_name", [None] * sector_count)[sector_index]
                    if isinstance(input_parms.get("central_patch_name", []), list)
                    else None
                ),
                "channels_out": int(input_parms["channels_out"][sector_index]),
                "deconvolution_channels": int(input_parms["deconvolution_channels"][sector_index]),
                "fit_spectral_pol": int(input_parms["fit_spectral_pol"][sector_index]),
                "ra": float(input_parms["ra"][sector_index]),
                "dec": float(input_parms["dec"][sector_index]),
                "wsclean_imsize": [
                    int(value) for value in input_parms["wsclean_imsize"][sector_index]
                ],
                "vertices_file": _file_record_path(input_parms["vertices_file"][sector_index]),
                "region_file": _optional_file_record_path(input_parms["region_file"][sector_index]),
                "facet_skymodel": facet_skymodel,
                "facet_region_filename": facet_region_filename,
                "facet_region_path": (
                    None
                    if facet_region_filename is None
                    else os.path.join(pipeline_dir, facet_region_filename)
                ),
                "filtered_model_image_filename": filtered_model_image_filename,
                "filtered_model_image_path": (
                    None
                    if filtered_model_image_filename is None
                    else os.path.join(pipeline_dir, filtered_model_image_filename)
                ),
                "image_I_cube_filename": image_i_cube_filename,
                "image_I_cube_path": (
                    None
                    if image_i_cube_filename is None
                    else os.path.join(pipeline_dir, image_i_cube_filename)
                ),
                "image_cube_specs": image_cube_specs,
                "output_source_catalog_filename": output_source_catalog_filename,
                "output_source_catalog_path": (
                    None
                    if output_source_catalog_filename is None
                    else os.path.join(pipeline_dir, output_source_catalog_filename)
                ),
                "output_normalize_h5parm_filename": output_normalize_h5parm_filename,
                "output_normalize_h5parm_path": (
                    None
                    if output_normalize_h5parm_filename is None
                    else os.path.join(pipeline_dir, output_normalize_h5parm_filename)
                ),
                "ra_mid": (None if not use_facets else float(input_parms["ra_mid"][sector_index])),
                "dec_mid": (
                    None if not use_facets else float(input_parms["dec_mid"][sector_index])
                ),
                "width_ra": (
                    None if not use_facets else float(input_parms["width_ra"][sector_index])
                ),
                "width_dec": (
                    None if not use_facets else float(input_parms["width_dec"][sector_index])
                ),
                "wsclean_niter": int(input_parms["wsclean_niter"][sector_index]),
                "wsclean_nmiter": int(input_parms["wsclean_nmiter"][sector_index]),
                "skip_final_iteration": bool(input_parms["skip_final_iteration"]),
                "robust": float(input_parms["robust"][sector_index]),
                "cellsize_deg": float(input_parms["cellsize_deg"][sector_index]),
                "min_uv_lambda": float(input_parms["min_uv_lambda"][sector_index]),
                "max_uv_lambda": float(input_parms["max_uv_lambda"][sector_index]),
                "mgain": float(input_parms["mgain"][sector_index]),
                "taper_arcsec": float(input_parms["taper_arcsec"][sector_index]),
                "local_rms_strength": float(input_parms["local_rms_strength"][sector_index]),
                "local_rms_window": float(input_parms["local_rms_window"][sector_index]),
                "local_rms_method": str(input_parms["local_rms_method"][sector_index]),
                "auto_mask": float(input_parms["auto_mask"][sector_index]),
                "auto_mask_nmiter": int(input_parms["auto_mask_nmiter"][sector_index]),
                "idg_mode": str(input_parms["idg_mode"][sector_index]),
                "wsclean_mem": float(input_parms["wsclean_mem"][sector_index]),
                "threshisl": float(input_parms["threshisl"][sector_index]),
                "threshpix": float(input_parms["threshpix"][sector_index]),
                "do_multiscale": bool(input_parms["do_multiscale"][sector_index]),
                "dd_psf_grid": [int(value) for value in input_parms["dd_psf_grid"][sector_index]],
                "interval": interval,
                "soltabs": None if not use_facets else str(input_parms["soltabs"]),
                "parallel_gridding_threads": (
                    None if not use_facets else int(input_parms["parallel_gridding_threads"])
                ),
                "scalar_visibilities": (
                    None if not use_facets else bool(input_parms["scalar_visibilities"])
                ),
                "diagonal_visibilities": (
                    None if not use_facets else bool(input_parms["diagonal_visibilities"])
                ),
                "shared_facet_reads": (
                    None if not use_facets else bool(input_parms["shared_facet_rw"])
                ),
                "shared_facet_writes": (
                    None if not use_facets else bool(input_parms["shared_facet_rw"])
                ),
                "pol": pol,
                "save_source_list": bool(input_parms["save_source_list"]),
                "link_polarizations": input_parms["link_polarizations"],
                "join_polarizations": bool(input_parms["join_polarizations"]),
                "filter_by_mask": bool(input_parms["filter_by_mask"]),
                "source_finder": str(input_parms["source_finder"]),
                "apply_time_frequency_smearing": bool(input_parms["apply_time_frequency_smearing"]),
                "max_threads": int(input_parms["max_threads"]),
                "deconvolution_threads": int(input_parms["deconvolution_threads"]),
                "mpi_nnodes": (
                    None if not use_mpi else int(input_parms["mpi_nnodes"][sector_index])
                ),
                "mpi_cpus_per_task": (
                    None if not use_mpi else int(input_parms["mpi_cpus_per_task"][sector_index])
                ),
                "allow_internet_access": bool(input_parms["allow_internet_access"]),
                "photometry_skymodel": photometry_skymodel,
                "astrometry_skymodel": astrometry_skymodel,
                "obs_original_paths": [_directory_record_path(record) for record in obs_records],
                "obs_starttime": [str(value) for value in starttimes],
                "obs_ntimes": [int(value) for value in ntimes],
            }
        )

    stokes_mode = "stokes_i" if _is_stokes_i(pol) else "full_stokes"
    if apply_screens:
        mode = f"screens_{stokes_mode}"
    elif use_facets:
        mode = f"facet_{stokes_mode}"
    else:
        mode = f"no_dde_{stokes_mode}"

    payload = {
        "mode": mode,
        "use_mpi": bool(use_mpi),
        "pipeline_working_dir": pipeline_dir,
        "sectors": sectors,
    }
    return assert_serializable_payload(payload)


def _require_file(path: str, description: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{description} was not created: {path}")
    return file_record(path)


def _require_directory(path: str, description: str) -> dict:
    if not os.path.isdir(path):
        raise FileNotFoundError(f"{description} was not created: {path}")
    return directory_record(path)


def _first_existing_file(patterns: list[str], description: str) -> dict:
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            if os.path.isfile(path):
                return file_record(path)
    raise FileNotFoundError(f"{description} was not created: {', '.join(patterns)}")


def _optional_first_existing_file(patterns: list[str]) -> Optional[dict]:
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            if os.path.isfile(path):
                return file_record(path)
    return None


def _file_records_for_required_patterns(patterns: list[str], description: str) -> list[dict]:
    records = _file_records_for_patterns(patterns)
    if not records:
        raise FileNotFoundError(f"{description} was not created: {', '.join(patterns)}")
    return records


def _file_records_for_patterns(patterns: list[str]) -> list[dict]:
    records = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            if os.path.isfile(path):
                records.append(file_record(path))
    return records


def _compressed_file_record(record: dict, description: str) -> dict:
    return _require_file(f"{record['path']}.fz", description)


def _cleanup_directory(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path)


def _compress_image_records(
    image_name: str,
    sector_images: list[dict],
    extra_images: list[dict],
    pipeline_working_dir: str,
    execution_config: ExecutionConfig,
    shell_operation_cls=None,
) -> tuple[list[dict], list[dict]]:
    command = build_compress_sector_images_command(
        [record["path"] for record in sector_images + extra_images]
    )
    _run_shell(
        command, pipeline_working_dir, execution_config, shell_operation_cls=shell_operation_cls
    )
    compressed_sector_images = [
        _compressed_file_record(sector_images[0], "Compressed WSClean non-PB image"),
        _compressed_file_record(sector_images[1], "Compressed WSClean PB image"),
    ]
    compressed_extra_images = _file_records_for_patterns(
        [
            os.path.join(pipeline_working_dir, f"{image_name}-MFS-[QUV]-image.fits.fz"),
            os.path.join(pipeline_working_dir, f"{image_name}-MFS-[QUV]-image-pb.fits.fz"),
            os.path.join(pipeline_working_dir, f"{image_name}-MFS-*residual.fits.fz"),
            os.path.join(pipeline_working_dir, f"{image_name}-MFS-*model-pb.fits.fz"),
            os.path.join(pipeline_working_dir, f"{image_name}-MFS-*dirty.fits.fz"),
        ]
    )
    return compressed_sector_images, compressed_extra_images


def _channel_image_patterns(image_name: str, stokes: str, pipeline_working_dir: str) -> list[str]:
    if stokes == "I":
        return [
            os.path.join(pipeline_working_dir, f"{image_name}-0???-image-pb.fits"),
            os.path.join(pipeline_working_dir, f"{image_name}-0???-I-image-pb.fits"),
        ]
    return [os.path.join(pipeline_working_dir, f"{image_name}-0???-{stokes}-image-pb.fits")]


def _make_image_cube_records(
    image_name: str,
    image_cube_specs: list[Mapping[str, object]],
    pipeline_working_dir: str,
    execution_config: ExecutionConfig,
    shell_operation_cls=None,
) -> tuple[list[dict], list[dict], list[dict]]:
    image_cubes = []
    image_cube_beams = []
    image_cube_frequencies = []
    for spec in image_cube_specs:
        stokes = str(spec["pol"])
        image_cube_filename = str(spec["filename"])
        channel_images = _file_records_for_required_patterns(
            _channel_image_patterns(image_name, stokes, pipeline_working_dir),
            f"WSClean Stokes-{stokes} channel images",
        )
        command = build_make_image_cube_command(
            [record["path"] for record in channel_images], image_cube_filename
        )
        _run_shell(
            command, pipeline_working_dir, execution_config, shell_operation_cls=shell_operation_cls
        )
        image_cube_path = os.path.join(pipeline_working_dir, image_cube_filename)
        image_cubes.append(_require_file(image_cube_path, f"Stokes-{stokes} image cube"))
        image_cube_beams.append(
            _require_file(f"{image_cube_path}_beams.txt", f"Stokes-{stokes} image cube beams")
        )
        image_cube_frequencies.append(
            _require_file(
                f"{image_cube_path}_frequencies.txt",
                f"Stokes-{stokes} image cube frequencies",
            )
        )
    return image_cubes, image_cube_beams, image_cube_frequencies


def _make_normalization_records(
    image_cube: dict,
    image_cube_beams: dict,
    image_cube_frequencies: dict,
    concat_record: dict,
    sector: Mapping[str, object],
    pipeline_working_dir: str,
    execution_config: ExecutionConfig,
    shell_operation_cls=None,
) -> tuple[dict, dict]:
    catalog_command = build_make_catalog_from_image_cube_command(
        image_cube["path"],
        image_cube_beams["path"],
        image_cube_frequencies["path"],
        str(sector["output_source_catalog_filename"]),
        float(sector["threshisl"]),
        float(sector["threshpix"]),
        int(sector["max_threads"]),
    )
    _run_shell(
        catalog_command,
        pipeline_working_dir,
        execution_config,
        shell_operation_cls=shell_operation_cls,
    )
    source_catalog = _require_file(
        str(sector["output_source_catalog_path"]), "Normalization source catalog"
    )

    normalize_command = build_normalize_flux_scale_command(
        source_catalog["path"],
        concat_record["path"],
        str(sector["output_normalize_h5parm_filename"]),
    )
    _run_shell(
        normalize_command,
        pipeline_working_dir,
        execution_config,
        shell_operation_cls=shell_operation_cls,
    )
    normalize_h5parm = _require_file(
        str(sector["output_normalize_h5parm_path"]), "Flux-scale normalization h5parm"
    )
    return source_catalog, normalize_h5parm


def _restore_bright_source_image(
    image_record: Mapping[str, str],
    bright_skymodel_pb: str,
    pipeline_working_dir: str,
    execution_config: ExecutionConfig,
    numthreads: int,
    description: str,
    shell_operation_cls=None,
) -> dict:
    output_image = os.path.basename(image_record["path"])
    command = build_wsclean_restore_command(
        image_record["path"], bright_skymodel_pb, output_image, numthreads
    )
    _run_shell(
        command, pipeline_working_dir, execution_config, shell_operation_cls=shell_operation_cls
    )
    return _require_file(os.path.join(pipeline_working_dir, output_image), description)


def _write_aterm_config(pipeline_working_dir: str, h5parm: str) -> str:
    config_path = os.path.join(pipeline_working_dir, ATERM_CONFIG_FILENAME)
    os.makedirs(pipeline_working_dir, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as handle:
        handle.write(build_aterm_config_content(h5parm))
    return config_path


def _run_shell(
    command: list[str],
    pipeline_working_dir: str,
    execution_config: ExecutionConfig,
    shell_operation_cls=None,
    environment: Optional[Mapping[str, str]] = None,
) -> None:
    run_shell_command(
        ShellCommand(
            command=command,
            environment={} if environment is None else dict(environment),
            working_directory=pipeline_working_dir,
        ),
        execution_config,
        shell_operation_cls=shell_operation_cls,
    )


def _mpi_environment(
    threads: int,
    processes: int,
    execution_config: ExecutionConfig,
) -> Mapping[str, str]:
    resource_request = validate_resource_request(
        ResourceRequest(
            name="wsclean-mpi",
            threads=threads,
            processes=processes,
            use_mpi=True,
            exclusive=True,
        ),
        execution_config,
    )
    return thread_environment(resource_request)


def _wsclean_threads_for_sector(sector: Mapping[str, object]) -> int:
    if sector["use_mpi"]:
        return int(sector["mpi_cpus_per_task"])
    return int(sector["max_threads"])


def _wsclean_environment_for_sector(
    sector: Mapping[str, object],
    execution_config: ExecutionConfig,
) -> Optional[Mapping[str, str]]:
    if not sector["use_mpi"]:
        return None
    return _mpi_environment(
        _wsclean_threads_for_sector(sector),
        int(sector["mpi_nnodes"]),
        execution_config,
    )


def _build_wsclean_command_for_sector(
    sector: Mapping[str, object],
    concat_record: Mapping[str, str],
    mask_record: Mapping[str, str],
    region_record: Optional[Mapping[str, str]],
    temp_dir: str,
) -> list[str]:
    common_kwargs = {
        "msin": concat_record["path"],
        "name": str(sector["image_name"]),
        "mask": mask_record["path"],
        "wsclean_imsize": list(sector["wsclean_imsize"]),
        "wsclean_niter": int(sector["wsclean_niter"]),
        "wsclean_nmiter": int(sector["wsclean_nmiter"]),
        "robust": float(sector["robust"]),
        "min_uv_lambda": float(sector["min_uv_lambda"]),
        "max_uv_lambda": float(sector["max_uv_lambda"]),
        "mgain": float(sector["mgain"]),
        "multiscale": bool(sector["do_multiscale"]),
        "save_source_list": bool(sector["save_source_list"]),
        "pol": str(sector["pol"]),
        "link_polarizations": sector["link_polarizations"],
        "join_polarizations": bool(sector["join_polarizations"]),
        "skip_final_iteration": bool(sector["skip_final_iteration"]),
        "cellsize_deg": float(sector["cellsize_deg"]),
        "channels_out": int(sector["channels_out"]),
        "deconvolution_channels": int(sector["deconvolution_channels"]),
        "fit_spectral_pol": int(sector["fit_spectral_pol"]),
        "taper_arcsec": float(sector["taper_arcsec"]),
        "local_rms_strength": float(sector["local_rms_strength"]),
        "local_rms_window": float(sector["local_rms_window"]),
        "local_rms_method": str(sector["local_rms_method"]),
        "wsclean_mem": float(sector["wsclean_mem"]),
        "auto_mask": float(sector["auto_mask"]),
        "auto_mask_nmiter": int(sector["auto_mask_nmiter"]),
        "idg_mode": str(sector["idg_mode"]),
        "num_threads": _wsclean_threads_for_sector(sector),
        "num_deconvolution_threads": int(sector["deconvolution_threads"]),
        "dd_psf_grid": list(sector["dd_psf_grid"]),
        "apply_time_frequency_smearing": bool(sector["apply_time_frequency_smearing"]),
        "temp_dir": temp_dir,
    }
    if sector["use_facets"]:
        facet_kwargs = {
            **common_kwargs,
            "scalar_visibilities": bool(sector["scalar_visibilities"]),
            "diagonal_visibilities": bool(sector["diagonal_visibilities"]),
            "h5parm": str(sector["h5parm"]),
            "soltabs": str(sector["soltabs"]),
            "region_file": region_record["path"],
            "num_gridding_threads": int(sector["parallel_gridding_threads"]),
            "shared_facet_reads": bool(sector["shared_facet_reads"]),
            "shared_facet_writes": bool(sector["shared_facet_writes"]),
        }
        if sector["use_mpi"]:
            return build_wsclean_mpi_facets_command(
                mpi_nnodes=int(sector["mpi_nnodes"]), **facet_kwargs
            )
        return build_wsclean_facets_command(**facet_kwargs)
    if sector["apply_screens"]:
        screen_kwargs = {
            **common_kwargs,
            "interval": list(sector["interval"]),
        }
        if sector["use_mpi"]:
            return build_wsclean_mpi_screens_command(
                mpi_nnodes=int(sector["mpi_nnodes"]), **screen_kwargs
            )
        return build_wsclean_screens_command(**screen_kwargs)
    if sector["use_mpi"]:
        return build_wsclean_mpi_no_dde_command(
            mpi_nnodes=int(sector["mpi_nnodes"]), **common_kwargs
        )
    return build_wsclean_no_dde_command(**common_kwargs)


def run_image_sector(
    sector: Mapping[str, object],
    pipeline_working_dir: str,
    execution_config: Optional[ExecutionConfig] = None,
    shell_operation_cls=None,
) -> dict:
    """Run one imaging sector."""
    config = execution_config or ExecutionConfig(task_runner="sync")
    prepared_records = []
    for prepare_task in sector["prepare_tasks"]:
        if not os.path.isdir(str(prepare_task["msout_path"])):
            command = build_prepare_imaging_data_command(
                str(prepare_task["msin"]),
                str(sector["data_colname"]),
                str(prepare_task["msout"]),
                str(prepare_task["starttime"]),
                int(prepare_task["ntimes"]),
                str(sector["phasecenter"]),
                int(prepare_task["freqstep"]),
                int(prepare_task["timestep"]),
                str(sector["phasecenter"]),
                int(sector["max_threads"]),
                str(sector["prepare_data_steps"]),
                maxinterval=prepare_task.get("maxinterval"),
                timebase=sector.get("timebase"),
                h5parm=sector.get("h5parm"),
                fulljones_h5parm=sector.get("fulljones_h5parm"),
                normalize_h5parm=sector.get("input_normalize_h5parm"),
                central_patch_name=sector.get("central_patch_name"),
                applycal_steps=sector.get("prepare_data_applycal_steps"),
            )
            _run_shell(
                command, pipeline_working_dir, config, shell_operation_cls=shell_operation_cls
            )
        prepared_records.append(
            _require_directory(str(prepare_task["msout_path"]), "Prepared imaging MS")
        )

    prepared_paths = [record["path"] for record in prepared_records]
    if not os.path.isdir(str(sector["concat_path"])):
        concat_command = build_concat_time_command(
            prepared_paths, str(sector["concat_filename"]), str(sector["data_colname"])
        )
        _run_shell(
            concat_command, pipeline_working_dir, config, shell_operation_cls=shell_operation_cls
        )
    concat_record = _require_directory(str(sector["concat_path"]), "Concatenated imaging MS")

    if not os.path.isfile(str(sector["mask_path"])):
        mask_command = build_blank_image_command(
            str(sector["mask_filename"]),
            list(sector["wsclean_imsize"]),
            str(sector["vertices_file"]),
            float(sector["ra"]),
            float(sector["dec"]),
            float(sector["cellsize_deg"]),
            image_filename=sector.get("previous_mask_filename"),
            region_file=sector.get("region_file"),
        )
        _run_shell(
            mask_command, pipeline_working_dir, config, shell_operation_cls=shell_operation_cls
        )
    mask_record = _require_file(str(sector["mask_path"]), "Imaging mask")

    region_record = None
    if sector["use_facets"]:
        if not os.path.isfile(str(sector["facet_region_path"])):
            region_command = build_make_region_file_command(
                str(sector["facet_skymodel"]),
                float(sector["ra_mid"]),
                float(sector["dec_mid"]),
                float(sector["width_ra"]),
                float(sector["width_dec"]),
                str(sector["facet_region_filename"]),
            )
            _run_shell(
                region_command,
                pipeline_working_dir,
                config,
                shell_operation_cls=shell_operation_cls,
            )
        region_record = _require_file(str(sector["facet_region_path"]), "Facet region file")

    temp_dir = os.path.join(pipeline_working_dir, f"{sector['image_name']}_wsclean_tmp")
    image_name = str(sector["image_name"])
    nonpb_image_patterns = [
        os.path.join(pipeline_working_dir, f"{image_name}-MFS-image.fits"),
        os.path.join(pipeline_working_dir, f"{image_name}-MFS-I-image.fits"),
    ]
    pb_image_patterns = [
        os.path.join(pipeline_working_dir, f"{image_name}-MFS-image-pb.fits"),
        os.path.join(pipeline_working_dir, f"{image_name}-MFS-I-image-pb.fits"),
    ]
    nonpb_image = _optional_first_existing_file(nonpb_image_patterns)
    pb_image = _optional_first_existing_file(pb_image_patterns)
    wsclean_ran = False
    if nonpb_image is None or pb_image is None:
        if sector["apply_screens"]:
            _write_aterm_config(pipeline_working_dir, str(sector["h5parm"]))
        wsclean_command = _build_wsclean_command_for_sector(
            sector, concat_record, mask_record, region_record, temp_dir
        )
        wsclean_environment = _wsclean_environment_for_sector(sector, config)
        try:
            os.makedirs(temp_dir, exist_ok=True)
            _run_shell(
                wsclean_command,
                pipeline_working_dir,
                config,
                shell_operation_cls=shell_operation_cls,
                environment=wsclean_environment,
            )
            wsclean_ran = True
        finally:
            _cleanup_directory(temp_dir)
        nonpb_image = _first_existing_file(nonpb_image_patterns, "WSClean non-PB image")
        pb_image = _first_existing_file(pb_image_patterns, "WSClean PB image")
    extra_images = _file_records_for_patterns(
        [
            os.path.join(pipeline_working_dir, f"{image_name}-MFS-[QUV]-image.fits"),
            os.path.join(pipeline_working_dir, f"{image_name}-MFS-[QUV]-image-pb.fits"),
            os.path.join(pipeline_working_dir, f"{image_name}-MFS-*residual.fits"),
            os.path.join(pipeline_working_dir, f"{image_name}-MFS-*model-pb.fits"),
            os.path.join(pipeline_working_dir, f"{image_name}-MFS-*dirty.fits"),
        ]
    )
    sector_images = [nonpb_image, pb_image]
    if sector["peel_bright_sources"]:
        pb_image = _restore_bright_source_image(
            pb_image,
            str(sector["bright_skymodel_pb"]),
            pipeline_working_dir,
            config,
            int(sector["max_threads"]),
            "Bright-source restored PB image",
            shell_operation_cls=shell_operation_cls,
        )
        nonpb_image = _restore_bright_source_image(
            nonpb_image,
            str(sector["bright_skymodel_pb"]),
            pipeline_working_dir,
            config,
            int(sector["max_threads"]),
            "Bright-source restored non-PB image",
            shell_operation_cls=shell_operation_cls,
        )
        sector_images = [nonpb_image, pb_image]
    image_cubes = []
    image_cube_beams = []
    image_cube_frequencies = []
    if sector["make_image_cube"]:
        image_cubes, image_cube_beams, image_cube_frequencies = _make_image_cube_records(
            image_name,
            list(sector["image_cube_specs"]),
            pipeline_working_dir,
            config,
            shell_operation_cls=shell_operation_cls,
        )

    skymodel_nonpb = None
    skymodel_pb = None
    if sector["save_source_list"]:
        skymodel_nonpb = _require_file(
            os.path.join(pipeline_working_dir, f"{image_name}-sources.txt"),
            "WSClean apparent-sky source list",
        )
        skymodel_pb = _require_file(
            os.path.join(pipeline_working_dir, f"{image_name}-sources-pb.txt"),
            "WSClean true-sky source list",
        )

    if wsclean_ran:
        for image_record in (pb_image, nonpb_image):
            command = build_check_image_beam_command(
                image_record["path"], float(sector["taper_arcsec"])
            )
            _run_shell(
                command, pipeline_working_dir, config, shell_operation_cls=shell_operation_cls
            )
            _require_file(image_record["path"], "Beam-checked image")

    filter_command = build_filter_skymodel_command(
        nonpb_image["path"],
        pb_image["path"],
        skymodel_pb["path"] if sector["save_source_list"] else "none",
        skymodel_nonpb["path"] if sector["save_source_list"] else "none",
        image_name,
        str(sector["vertices_file"]),
        list(sector["obs_original_paths"]),
        float(sector["threshisl"]),
        float(sector["threshpix"]),
        bool(sector["filter_by_mask"]),
        str(sector["source_finder"]),
        int(sector["max_threads"]),
        bright_true_sky_skymodel=sector.get("bright_skymodel_pb"),
    )
    _run_shell(
        filter_command, pipeline_working_dir, config, shell_operation_cls=shell_operation_cls
    )

    filtered_true_sky = _require_file(
        os.path.join(pipeline_working_dir, f"{image_name}.true_sky.txt"),
        "Filtered true-sky skymodel",
    )
    filtered_apparent_sky = _require_file(
        os.path.join(pipeline_working_dir, f"{image_name}.apparent_sky.txt"),
        "Filtered apparent-sky skymodel",
    )
    diagnostics = _require_file(
        os.path.join(pipeline_working_dir, f"{image_name}.image_diagnostics.json"),
        "Image diagnostics",
    )
    flat_noise_rms = _require_file(
        os.path.join(pipeline_working_dir, f"{image_name}.flat_noise_rms.fits"),
        "Flat-noise RMS image",
    )
    true_sky_rms = _require_file(
        os.path.join(pipeline_working_dir, f"{image_name}.true_sky_rms.fits"),
        "True-sky RMS image",
    )
    source_catalog = _require_file(
        os.path.join(pipeline_working_dir, f"{image_name}.source_catalog.fits"),
        "Source catalog",
    )
    source_filtering_mask_path = os.path.join(
        pipeline_working_dir, f"{os.path.basename(pb_image['path'])}.mask.fits"
    )
    source_filtering_mask = (
        file_record(source_filtering_mask_path)
        if os.path.isfile(source_filtering_mask_path)
        else None
    )

    skymodel_image = None
    if sector["save_filtered_model_image"]:
        skymodel_image_command = build_make_skymodel_image_command(
            filtered_apparent_sky["path"],
            pb_image["path"],
            str(sector["filtered_model_image_filename"]),
        )
        _run_shell(
            skymodel_image_command,
            pipeline_working_dir,
            config,
            shell_operation_cls=shell_operation_cls,
        )
        skymodel_image = _require_file(
            str(sector["filtered_model_image_path"]), "Filtered skymodel image"
        )

    diagnostics_command = build_calculate_image_diagnostics_command(
        nonpb_image["path"],
        flat_noise_rms["path"],
        pb_image["path"],
        true_sky_rms["path"],
        source_catalog["path"],
        list(sector["obs_original_paths"]),
        list(sector["obs_starttime"]),
        list(sector["obs_ntimes"]),
        diagnostics["path"],
        image_name,
        bool(sector["allow_internet_access"]),
        facet_region_file=None if region_record is None else region_record["path"],
        photometry_skymodel=sector.get("photometry_skymodel"),
        astrometry_skymodel=sector.get("astrometry_skymodel"),
    )
    _run_shell(
        diagnostics_command, pipeline_working_dir, config, shell_operation_cls=shell_operation_cls
    )
    diagnostics = _require_file(
        os.path.join(pipeline_working_dir, f"{image_name}.image_diagnostics.json"),
        "Image diagnostics",
    )
    offsets_path = os.path.join(pipeline_working_dir, f"{image_name}.astrometry_offsets.json")
    offsets = file_record(offsets_path) if os.path.isfile(offsets_path) else None
    diagnostic_plots = _file_records_for_patterns(
        [os.path.join(pipeline_working_dir, f"{image_name}*.pdf")]
    )
    publish_plot_file_records([diagnostics], pipeline_working_dir)
    publish_plot_file_records(diagnostic_plots, pipeline_working_dir)

    output_sector_images = sector_images
    output_extra_images = extra_images
    if sector["compress_images"]:
        output_sector_images, output_extra_images = _compress_image_records(
            image_name,
            sector_images,
            extra_images,
            pipeline_working_dir,
            config,
            shell_operation_cls=shell_operation_cls,
        )

    normalization_source_catalog = None
    normalize_h5parm = None
    if sector["normalize_flux_scale"]:
        normalization_source_catalog, normalize_h5parm = _make_normalization_records(
            image_cubes[0],
            image_cube_beams[0],
            image_cube_frequencies[0],
            concat_record,
            sector,
            pipeline_working_dir,
            config,
            shell_operation_cls=shell_operation_cls,
        )

    result = {
        "filtered_skymodel_true_sky": filtered_true_sky,
        "filtered_skymodel_apparent_sky": filtered_apparent_sky,
        "pybdsf_catalog": source_catalog,
        "sector_diagnostics": diagnostics,
        "sector_offsets": offsets,
        "sector_diagnostic_plots": diagnostic_plots,
        "visibilities": prepared_records,
        "sector_I_images": output_sector_images,
        "sector_extra_images": output_extra_images,
        "source_filtering_mask": source_filtering_mask,
        "sector_skymodels": [skymodel_nonpb, skymodel_pb] if sector["save_source_list"] else None,
    }
    if region_record is not None:
        result["sector_region_file"] = region_record
    if skymodel_image is not None:
        result["sector_skymodel_image_fits"] = skymodel_image
    if image_cubes:
        result["sector_image_cubes"] = image_cubes
        result["sector_image_cube_beams"] = image_cube_beams
        result["sector_image_cube_frequencies"] = image_cube_frequencies
    if normalize_h5parm is not None:
        result["sector_source_catalog"] = normalization_source_catalog
        result["sector_normalize_h5parm"] = normalize_h5parm
    fits_records = (
        output_sector_images
        + output_extra_images
        + [flat_noise_rms, true_sky_rms, source_catalog]
        + ([source_filtering_mask] if source_filtering_mask is not None else [])
        + ([skymodel_image] if skymodel_image is not None else [])
        + image_cubes
    )
    publish_fits_image_artifacts(fits_records, pipeline_working_dir)
    return result


@task(name="image_sector")
def image_sector_task(
    sector: Mapping[str, object],
    pipeline_working_dir: str,
    execution_config: Optional[ExecutionConfig] = None,
    shell_operation_cls=None,
) -> dict:
    """Prefect task wrapper for one imaging sector."""
    with publish_python_logs_to_prefect():
        return run_image_sector(
            sector,
            pipeline_working_dir,
            execution_config=execution_config,
            shell_operation_cls=shell_operation_cls,
        )


def _result_from_sector_records(sector_outputs: list[dict]) -> dict:
    result = {
        "filtered_skymodel_true_sky": [
            sector["filtered_skymodel_true_sky"] for sector in sector_outputs
        ],
        "filtered_skymodel_apparent_sky": [
            sector["filtered_skymodel_apparent_sky"] for sector in sector_outputs
        ],
        "pybdsf_catalog": [sector["pybdsf_catalog"] for sector in sector_outputs],
        "sector_diagnostics": [sector["sector_diagnostics"] for sector in sector_outputs],
        "sector_offsets": [sector["sector_offsets"] for sector in sector_outputs],
        "sector_diagnostic_plots": [sector["sector_diagnostic_plots"] for sector in sector_outputs],
        "visibilities": [sector["visibilities"] for sector in sector_outputs],
        "sector_I_images": [sector["sector_I_images"] for sector in sector_outputs],
        "sector_extra_images": [sector["sector_extra_images"] for sector in sector_outputs],
        "source_filtering_mask": [sector["source_filtering_mask"] for sector in sector_outputs],
    }
    if any(sector["sector_skymodels"] is not None for sector in sector_outputs):
        result["sector_skymodels"] = [sector["sector_skymodels"] for sector in sector_outputs]
    if any("sector_region_file" in sector for sector in sector_outputs):
        result["sector_region_file"] = [
            sector.get("sector_region_file") for sector in sector_outputs
        ]
    if any("sector_skymodel_image_fits" in sector for sector in sector_outputs):
        result["sector_skymodel_image_fits"] = [
            sector.get("sector_skymodel_image_fits") for sector in sector_outputs
        ]
    if any("sector_image_cubes" in sector for sector in sector_outputs):
        result["sector_image_cubes"] = [
            sector.get("sector_image_cubes") for sector in sector_outputs
        ]
        result["sector_image_cube_beams"] = [
            sector.get("sector_image_cube_beams") for sector in sector_outputs
        ]
        result["sector_image_cube_frequencies"] = [
            sector.get("sector_image_cube_frequencies") for sector in sector_outputs
        ]
    if any("sector_normalize_h5parm" in sector for sector in sector_outputs):
        result["sector_source_catalog"] = [
            sector.get("sector_source_catalog") for sector in sector_outputs
        ]
        result["sector_normalize_h5parm"] = [
            sector.get("sector_normalize_h5parm") for sector in sector_outputs
        ]
    for value in result.values():
        validate_output_record(value, allow_none=True)
    return result


def _validate_image_payload(payload: Mapping[str, object]) -> tuple[str, list[Mapping]]:
    supported_modes = {
        "facet_full_stokes",
        "facet_stokes_i",
        "no_dde_full_stokes",
        "no_dde_stokes_i",
        "screens_full_stokes",
        "screens_stokes_i",
    }
    if payload.get("mode") not in supported_modes:
        raise ValueError("Only no-DDE, facet, and screen image payloads are supported")
    pipeline_working_dir = str(payload["pipeline_working_dir"])
    sectors = payload.get("sectors", [])
    if not isinstance(sectors, list):
        raise ValueError("sectors must be a list")
    return pipeline_working_dir, sectors


def run_image_flow(
    payload: Mapping[str, object],
    execution_config: Optional[ExecutionConfig] = None,
    shell_operation_cls=None,
) -> dict:
    """Run imaging commands and return finalizer-compatible outputs."""
    assert_serializable_payload(payload)
    config = execution_config or ExecutionConfig(task_runner="sync")
    pipeline_working_dir, sectors = _validate_image_payload(payload)
    sector_outputs = [
        run_image_sector(
            sector,
            pipeline_working_dir,
            execution_config=config,
            shell_operation_cls=shell_operation_cls,
        )
        for sector in sectors
    ]
    return _result_from_sector_records(sector_outputs)


def _run_image_prefect_tasks(
    payload: Mapping[str, object],
    execution_config: Optional[ExecutionConfig] = None,
) -> dict:
    config = execution_config or ExecutionConfig(task_runner="sync")
    pipeline_working_dir, sectors = _validate_image_payload(payload)
    sector_outputs = [
        image_sector_task.submit(
            sector,
            pipeline_working_dir,
            execution_config=config,
        )
        for sector in sectors
    ]
    sector_outputs = [output.result() for output in sector_outputs]
    return _result_from_sector_records(sector_outputs)


@flow(name="image")
def _image_flow(
    payload: Mapping[str, object],
    execution_config: Optional[ExecutionConfig] = None,
):
    """Prefect implementation for imaging."""
    with publish_python_logs_to_prefect():
        return _run_image_prefect_tasks(payload, execution_config=execution_config)


def image_flow(
    payload: Mapping[str, object],
    execution_config: Optional[ExecutionConfig] = None,
):
    """Prefect entry point for imaging."""
    return run_flow_with_task_runner(
        _image_flow,
        payload,
        execution_config=execution_config,
    )


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

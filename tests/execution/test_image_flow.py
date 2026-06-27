import json
import shlex
from copy import deepcopy
from pathlib import Path

import pytest
from prefect.testing.utilities import prefect_test_harness

import rapthor.execution.image.sector as image_sector_module
from rapthor.execution.commands import normalize_command
from rapthor.execution.config import ExecutionConfig
from rapthor.execution.flows.image import (
    image_flow,
    image_sector_task,
    run_image_flow,
)
from rapthor.execution.image.commands import (
    ATERM_CONFIG_FILENAME,
    build_aterm_config_content,
    build_blank_image_command,
    build_calculate_image_diagnostics_command,
    build_check_image_beam_command,
    build_compress_sector_images_command,
    build_concat_time_command,
    build_filter_skymodel_command,
    build_make_catalog_from_image_cube_command,
    build_make_image_cube_command,
    build_make_region_file_command,
    build_make_skymodel_image_command,
    build_normalize_flux_scale_command,
    build_prepare_imaging_data_command,
    build_wsclean_facets_command,
    build_wsclean_mpi_facets_command,
    build_wsclean_mpi_no_dde_command,
    build_wsclean_mpi_screens_command,
    build_wsclean_no_dde_command,
    build_wsclean_restore_command,
    build_wsclean_screens_command,
)
from rapthor.execution.image.payloads import image_payload_from_inputs
from rapthor.lib.records import directory_record, file_record, validate_output_record
from rapthor.operations.image import Image, ImageInitial, ImageNormalize

FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fake_image_shell_operation_cls():
    class FakeImageShellOperation:
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.instances.append(self)

        def run(self):
            tokens = shlex.split(self.kwargs["commands"][0])
            cwd = Path(self.kwargs["working_dir"])
            if tokens[0] == "DP3":
                output_name = next(
                    token.split("=", 1)[1] for token in tokens if token.startswith("msout=")
                )
                (cwd / output_name).mkdir(parents=True, exist_ok=True)
            elif tokens[0] == "concat_ms.py":
                output_name = next(
                    token.split("=", 1)[1] for token in tokens if token.startswith("--msout=")
                )
                (cwd / output_name).mkdir(parents=True, exist_ok=True)
            elif tokens[0] == "blank_image.py":
                (cwd / tokens[1]).write_text("mask")
            elif tokens[0] == "make_region_file.py":
                (cwd / tokens[6]).write_text("region")
            elif tokens[0] in {"wsclean", "mpirun"}:
                if tokens[0] == "wsclean" and "-restore-list" in tokens:
                    (cwd / tokens[-1]).write_text("restored image")
                    return "OK"
                if tokens[0] == "mpirun":
                    tokens = ["wsclean"] + tokens[tokens.index("wsclean-mp") + 1 :]
                image_name = tokens[tokens.index("-name") + 1]
                pol = tokens[tokens.index("-pol") + 1].upper()
                temp_dir = Path(tokens[tokens.index("-temp-dir") + 1])
                if not temp_dir.is_absolute():
                    temp_dir = cwd / temp_dir
                assert temp_dir.is_dir()
                (temp_dir / "wsclean.tmp").write_text("temporary")
                channels_out = int(tokens[tokens.index("-channels-out") + 1])
                if pol == "I":
                    for suffix in [
                        "-MFS-I-image.fits",
                        "-MFS-I-image-pb.fits",
                        "-MFS-I-residual.fits",
                        "-MFS-I-model-pb.fits",
                        "-MFS-I-dirty.fits",
                    ]:
                        (cwd / f"{image_name}{suffix}").write_text("image")
                    for channel in range(channels_out):
                        (cwd / f"{image_name}-{channel:04d}-I-image-pb.fits").write_text("channel")
                else:
                    for suffix in ["-MFS-image.fits", "-MFS-image-pb.fits"]:
                        (cwd / f"{image_name}{suffix}").write_text("image")
                    for stokes in "QUV":
                        for suffix in ["image.fits", "image-pb.fits"]:
                            (cwd / f"{image_name}-MFS-{stokes}-{suffix}").write_text("image")
                    for stokes in "IQUV":
                        for suffix in ["residual.fits", "model-pb.fits", "dirty.fits"]:
                            (cwd / f"{image_name}-MFS-{stokes}-{suffix}").write_text("image")
                    for channel in range(channels_out):
                        (cwd / f"{image_name}-{channel:04d}-image-pb.fits").write_text("channel")
                        for stokes in "QUV":
                            (cwd / f"{image_name}-{channel:04d}-{stokes}-image-pb.fits").write_text(
                                "channel"
                            )
                if "-save-source-list" in tokens:
                    for suffix in ["-sources.txt", "-sources-pb.txt"]:
                        (cwd / f"{image_name}{suffix}").write_text("skymodel")
            elif tokens[0] == "make_image_cube.py":
                output_image = tokens[2]
                (cwd / output_image).write_text("cube")
                (cwd / f"{output_image}_beams.txt").write_text("beams")
                (cwd / f"{output_image}_frequencies.txt").write_text("frequencies")
            elif tokens[0] == "make_catalog_from_image_cube.py":
                (cwd / tokens[4]).write_text("catalog")
            elif tokens[0] == "normalize_flux_scale.py":
                (cwd / tokens[3]).write_text("h5parm")
            elif tokens[0] == "fpack":
                for image in tokens[1:]:
                    Path(f"{image}.fz").write_text("compressed")
            elif tokens[0] == "check_image_beam.py":
                Path(tokens[1]).touch(exist_ok=True)
            elif tokens[0] == "filter_skymodel.py":
                output_root = tokens[5]
                true_sky_image = Path(tokens[2]).name
                for suffix in [
                    ".true_sky.txt",
                    ".apparent_sky.txt",
                    ".flat_noise_rms.fits",
                    ".true_sky_rms.fits",
                    ".source_catalog.fits",
                ]:
                    (cwd / f"{output_root}{suffix}").write_text("filter")
                (cwd / f"{true_sky_image}.mask.fits").write_text("mask")
                (cwd / f"{output_root}.image_diagnostics.json").write_text("{}")
            elif tokens[0] == "restore_skymodel.py":
                (cwd / tokens[3]).write_text("model image")
            elif tokens[0] == "calculate_image_diagnostics.py":
                output_root = tokens[10]
                (cwd / f"{output_root}.image_diagnostics.json").write_text("{}")
                (cwd / f"{output_root}.astrometry_offsets.json").write_text("{}")
                (cwd / f"{output_root}.photometry.pdf").write_text("plot")
            else:
                raise AssertionError(f"Unexpected command: {tokens[0]}")
            return "OK"

    return FakeImageShellOperation


class NoOutputShellOperation:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def run(self):
        return "OK"


class FailingShellOperation:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def run(self):
        raise RuntimeError("image failed")


class ObservationStub:
    def __init__(self, ms_filename: str, starttime: float, numsamples: int):
        self.ms_filename = ms_filename
        self.starttime = starttime
        self.numsamples = numsamples
        self.timepersample = 1.0
        self.channels_are_regular = True
        self.parameters = {}


class SectorStub:
    def __init__(self, observations=None):
        self.name = "sector_1"
        self.diagnostics = []
        self.observations = observations or []
        self.I_mask_file = None
        self.ra = 123.0
        self.dec = 45.0
        self.wsclean_nchannels = 4
        self.wsclean_deconvolution_channels = 2
        self.wsclean_spectral_poly_order = 2
        self.imsize = [1024, 1024]
        self.width_ra = 2.0
        self.width_dec = 2.5
        self.vertices_file = "/data/sector_1.vertices"
        self.region_file = None
        self.wsclean_niter = 1000
        self.wsclean_nmiter = 5
        self.robust = -0.5
        self.cellsize_deg = 0.001
        self.min_uv_lambda = 80.0
        self.max_uv_lambda = 1000000.0
        self.mgain = 0.85
        self.taper_arcsec = 0.0
        self.local_rms_strength = 0.0
        self.local_rms_window = 25.0
        self.local_rms_method = "rms-with-min"
        self.auto_mask = 5.0
        self.auto_mask_nmiter = 1
        self.idg_mode = "cpu"
        self.mem_limit_gb = 8.0
        self.threshisl = 4.0
        self.threshpix = 5.0
        self.multiscale = True
        self.dd_psf_grid = [1, 1]

    def set_imaging_parameters(
        self,
        do_multiscale=False,
        recalculate_imsize=False,
        imaging_parameters=None,
        preapply_dde_solutions=False,
    ):
        self.multiscale = bool(do_multiscale)
        for index, obs in enumerate(self.observations):
            obs.parameters["ms_filename"] = obs.ms_filename
            obs.parameters["ms_prep_filename"] = f"{self.name}_obs_{index}_prep.ms"
            obs.parameters["image_freqstep"] = 4
            obs.parameters["image_timestep"] = 2
            obs.parameters["image_bda_maxinterval"] = 8

    def get_obs_parameters(self, name):
        if name == "ms_filename":
            return [obs.ms_filename for obs in self.observations]
        return [obs.parameters[name] for obs in self.observations]


class FieldStub:
    def __init__(self, tmp_path):
        self.parset = _operation_parset(tmp_path)
        self.observations = [
            ObservationStub("obs_0.ms", 59000.0, 10),
            ObservationStub("obs_1.ms", 59001.0, 12),
        ]
        self.make_image_cube = False
        self.full_field_sector = SectorStub(self.observations)
        self.imaging_sectors = [SectorStub(self.observations)]
        self.normalize_sector = SectorStub(self.observations)
        self.save_supplementary_images = False
        self.save_visibilities = False
        self.image_pol = "I"
        self.lofar_to_true_flux_ratio = None
        self.lofar_to_true_flux_std = None
        self.normalize_flux_scale = True
        self.apply_normalizations = False
        self.image_cube_stokes_list = ["I"]
        self.compress_images = False
        self.compress_selfcal_images = False
        self.apply_screens = False
        self.dde_method = "none"
        self.h5parm_filename = None
        self.dd_h5parm_filename = None
        self.di_h5parm_filename = None
        self.fulljones_h5parm_filename = None
        self.calibration_strategy = {}
        self.normalize_h5parm = None
        self.photometry_skymodel = None
        self.astrometry_skymodel = None
        self.use_mpi = False
        self.do_predict = False
        self.do_multiscale_clean = True
        self.pol_combine_method = "join"
        self.apply_amplitudes = False
        self.apply_fulljones = False
        self.apply_diagonal_solutions = False
        self.peel_bright_sources = False
        self.average_visibilities = True
        self.image_bda_timebase = 10.0
        self.slow_timestep_sec = 0.0
        self.data_colname = "DATA"
        self.skip_final_major_iteration = True
        self.correct_smearing_in_imaging = False
        self.disable_clean = False
        self.ra = 123.0
        self.dec = 45.0
        self.calibration_skymodel_file = "/data/calibration.skymodel"
        self.bright_source_skymodel_file = "/data/bright_sources_pb.txt"
        self.normalization_skymodels = None
        self.normalization_reference_frequencies = None

    def get_calibration_radius(self):
        return 1.0


def _operation_parset(tmp_path):
    return {
        "dir_working": str(tmp_path / "working"),
        "cluster_specific": {
            "debug_workflow": False,
            "keep_temporary_files": False,
            "max_nodes": 1,
            "batch_system": "single_machine",
            "cpus_per_task": 1,
            "mem_per_node_gb": 0,
            "dir_local": None,
            "local_scratch_dir": None,
            "global_scratch_dir": None,
            "use_container": False,
            "container_type": "docker",
            "max_cores": 1,
            "max_threads": 4,
            "deconvolution_threads": 2,
            "parallel_gridding_threads": 3,
            "allow_internet_access": False,
            "prefect_task_runner": "sync",
        },
        "imaging_specific": {
            "use_clean_mask": False,
            "save_filtered_model_image": False,
            "filter_skymodel": False,
            "source_finder": "bdsf",
            "shared_facet_rw": False,
        },
    }


def _image_input_parms():
    return {
        "obs_filename": [
            [
                directory_record("/data/obs_0.ms"),
                directory_record("/data/obs_1.ms"),
            ]
        ],
        "data_colname": "DATA",
        "prepare_filename": [["sector_1_obs_0_prep.ms", "sector_1_obs_1_prep.ms"]],
        "concat_filename": ["sector_1_concat.ms"],
        "previous_mask_filename": [None],
        "mask_filename": ["sector_1_mask.fits"],
        "starttime": [["50000.0", "50010.0"]],
        "ntimes": [[10, 12]],
        "image_freqstep": [[4, 4]],
        "image_timestep": [[2, 2]],
        "image_maxinterval": [[8, 8]],
        "image_timebase": [10.0],
        "phasecenter": ["'[123.0deg, 45.0deg]'"],
        "image_name": ["sector_1"],
        "pol": "I",
        "save_source_list": True,
        "link_polarizations": False,
        "join_polarizations": False,
        "prepare_data_steps": "[applybeam,shift,avg,bdaavg]",
        "prepare_data_applycal_steps": None,
        "h5parm": None,
        "fulljones_h5parm": None,
        "input_normalize_h5parm": None,
        "channels_out": [4],
        "deconvolution_channels": [2],
        "fit_spectral_pol": [2],
        "ra": [123.0],
        "dec": [45.0],
        "wsclean_imsize": [[1024, 1024]],
        "vertices_file": [file_record("/data/sector_1.vertices")],
        "region_file": [None],
        "wsclean_niter": [1000],
        "wsclean_nmiter": [5],
        "skip_final_iteration": True,
        "robust": [-0.5],
        "cellsize_deg": [0.001],
        "min_uv_lambda": [80.0],
        "max_uv_lambda": [1000000.0],
        "mgain": [0.85],
        "taper_arcsec": [0.0],
        "local_rms_strength": [0.0],
        "local_rms_window": [25.0],
        "local_rms_method": ["rms-with-min"],
        "auto_mask": [5.0],
        "auto_mask_nmiter": [1],
        "idg_mode": ["cpu"],
        "wsclean_mem": [8.0],
        "threshisl": [4.0],
        "threshpix": [5.0],
        "filter_by_mask": False,
        "source_finder": "bdsf",
        "do_multiscale": [True],
        "dd_psf_grid": [[1, 1]],
        "apply_time_frequency_smearing": False,
        "interval": [0, 10],
        "max_threads": 4,
        "deconvolution_threads": 2,
        "save_filtered_model_image": False,
        "filtered_model_image_name": ["sector_1-MFS-filtered-model.fits.fz"],
        "allow_internet_access": False,
        "photometry_skymodel": None,
        "astrometry_skymodel": None,
        "peel_bright_sources": False,
        "shared_facet_rw": False,
    }


def _facet_image_input_parms():
    input_parms = _image_input_parms()
    input_parms.update(
        {
            "h5parm": file_record("/data/facet-solutions.h5"),
            "skymodel": file_record("/data/calibration.skymodel"),
            "ra_mid": [123.0],
            "dec_mid": [45.0],
            "width_ra": [2.0],
            "width_dec": [2.5],
            "facet_region_file": ["sector_1_facets_ds9.reg"],
            "soltabs": "phase000",
            "parallel_gridding_threads": 3,
            "scalar_visibilities": True,
            "diagonal_visibilities": False,
            "shared_facet_rw": True,
        }
    )
    return input_parms


def _screens_image_input_parms():
    input_parms = _image_input_parms()
    input_parms.update(
        {
            "h5parm": file_record("/data/screen-solutions.h5"),
            "prepare_data_steps": "[applybeam,shift,avg]",
            "image_maxinterval": [[None, None]],
            "interval": [0, 9],
        }
    )
    return input_parms


def _filtered_model_image_input_parms():
    input_parms = _image_input_parms()
    input_parms["save_filtered_model_image"] = True
    return input_parms


def _bright_peeling_image_input_parms():
    input_parms = _image_input_parms()
    input_parms["peel_bright_sources"] = True
    input_parms["bright_skymodel_pb"] = file_record("/data/bright_sources_pb.txt")
    return input_parms


def _image_cube_input_parms():
    input_parms = _image_input_parms()
    input_parms["image_I_cube_name"] = ["sector_1_I_freq_cube.fits"]
    return input_parms


def _full_stokes_image_input_parms():
    input_parms = _image_input_parms()
    input_parms["pol"] = "IQUV"
    input_parms["save_source_list"] = False
    input_parms["join_polarizations"] = True
    return input_parms


def _linked_full_stokes_image_input_parms():
    input_parms = _full_stokes_image_input_parms()
    input_parms["join_polarizations"] = False
    input_parms["link_polarizations"] = "I"
    return input_parms


def _full_stokes_image_cube_input_parms():
    input_parms = _full_stokes_image_input_parms()
    input_parms["image_I_cube_name"] = ["sector_1_I_freq_cube.fits"]
    input_parms["image_Q_cube_name"] = ["sector_1_Q_freq_cube.fits"]
    input_parms["image_U_cube_name"] = ["sector_1_U_freq_cube.fits"]
    input_parms["image_V_cube_name"] = ["sector_1_V_freq_cube.fits"]
    return input_parms


def _normalize_image_input_parms():
    input_parms = _image_cube_input_parms()
    input_parms["save_source_list"] = False
    input_parms["output_source_catalog"] = ["sector_1_source_catalog.fits"]
    input_parms["output_normalize_h5parm"] = ["sector_1_normalize.h5parm"]
    return input_parms


def _clean_disabled_image_input_parms():
    input_parms = _image_input_parms()
    input_parms["wsclean_niter"] = [0]
    return input_parms


def _mpi_image_input_parms():
    input_parms = _image_input_parms()
    input_parms["mpi_nnodes"] = [2]
    input_parms["mpi_cpus_per_task"] = [3]
    return input_parms


def _mpi_facet_image_input_parms():
    input_parms = _facet_image_input_parms()
    input_parms["mpi_nnodes"] = [2]
    input_parms["mpi_cpus_per_task"] = [3]
    return input_parms


def _mpi_screens_image_input_parms():
    input_parms = _screens_image_input_parms()
    input_parms["mpi_nnodes"] = [2]
    input_parms["mpi_cpus_per_task"] = [3]
    return input_parms


def _two_sector_image_input_parms():
    input_parms = deepcopy(_image_input_parms())
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
        "image_name",
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
        "filtered_model_image_name",
    ]
    for key in per_sector_keys:
        input_parms[key].append(deepcopy(input_parms[key][0]))
    input_parms["prepare_filename"][1] = ["sector_2_obs_0_prep.ms", "sector_2_obs_1_prep.ms"]
    input_parms["concat_filename"][1] = "sector_2_concat.ms"
    input_parms["mask_filename"][1] = "sector_2_mask.fits"
    input_parms["image_name"][1] = "sector_2"
    input_parms["vertices_file"][1] = file_record("/data/sector_2.vertices")
    input_parms["filtered_model_image_name"][1] = "sector_2-MFS-filtered-model.fits.fz"
    return input_parms


def _expected_image_operation_outputs(operation):
    pipeline_dir = Path(operation.pipeline_working_dir)
    return {
        "filtered_skymodel_true_sky": [file_record(pipeline_dir / "sector_1.true_sky.txt")],
        "filtered_skymodel_apparent_sky": [file_record(pipeline_dir / "sector_1.apparent_sky.txt")],
        "pybdsf_catalog": [file_record(pipeline_dir / "sector_1.source_catalog.fits")],
        "sector_diagnostics": [file_record(pipeline_dir / "sector_1.image_diagnostics.json")],
        "sector_offsets": [file_record(pipeline_dir / "sector_1.astrometry_offsets.json")],
        "sector_diagnostic_plots": [[file_record(pipeline_dir / "sector_1.photometry.pdf")]],
        "visibilities": [
            [
                directory_record(pipeline_dir / "sector_1_obs_0_prep.ms"),
                directory_record(pipeline_dir / "sector_1_obs_1_prep.ms"),
            ]
        ],
        "sector_I_images": [
            [
                file_record(pipeline_dir / "sector_1-MFS-I-image.fits"),
                file_record(pipeline_dir / "sector_1-MFS-I-image-pb.fits"),
            ]
        ],
        "sector_extra_images": [
            [
                file_record(pipeline_dir / "sector_1-MFS-I-residual.fits"),
                file_record(pipeline_dir / "sector_1-MFS-I-model-pb.fits"),
                file_record(pipeline_dir / "sector_1-MFS-I-dirty.fits"),
            ]
        ],
        "source_filtering_mask": [
            file_record(pipeline_dir / "sector_1-MFS-I-image-pb.fits.mask.fits")
        ],
        "sector_skymodels": [
            [
                file_record(pipeline_dir / "sector_1-sources.txt"),
                file_record(pipeline_dir / "sector_1-sources-pb.txt"),
            ]
        ],
    }


def _expected_full_stokes_image_operation_outputs(operation):
    pipeline_dir = Path(operation.pipeline_working_dir)
    extra_images = []
    for suffix in [
        "Q-image.fits",
        "U-image.fits",
        "V-image.fits",
        "Q-image-pb.fits",
        "U-image-pb.fits",
        "V-image-pb.fits",
        "I-residual.fits",
        "Q-residual.fits",
        "U-residual.fits",
        "V-residual.fits",
        "I-model-pb.fits",
        "Q-model-pb.fits",
        "U-model-pb.fits",
        "V-model-pb.fits",
        "I-dirty.fits",
        "Q-dirty.fits",
        "U-dirty.fits",
        "V-dirty.fits",
    ]:
        extra_images.append(file_record(pipeline_dir / f"sector_1-MFS-{suffix}"))

    return {
        "filtered_skymodel_true_sky": [file_record(pipeline_dir / "sector_1.true_sky.txt")],
        "filtered_skymodel_apparent_sky": [file_record(pipeline_dir / "sector_1.apparent_sky.txt")],
        "pybdsf_catalog": [file_record(pipeline_dir / "sector_1.source_catalog.fits")],
        "sector_diagnostics": [file_record(pipeline_dir / "sector_1.image_diagnostics.json")],
        "sector_offsets": [file_record(pipeline_dir / "sector_1.astrometry_offsets.json")],
        "sector_diagnostic_plots": [[file_record(pipeline_dir / "sector_1.photometry.pdf")]],
        "visibilities": [
            [
                directory_record(pipeline_dir / "sector_1_obs_0_prep.ms"),
                directory_record(pipeline_dir / "sector_1_obs_1_prep.ms"),
            ]
        ],
        "sector_I_images": [
            [
                file_record(pipeline_dir / "sector_1-MFS-image.fits"),
                file_record(pipeline_dir / "sector_1-MFS-image-pb.fits"),
            ]
        ],
        "sector_extra_images": [extra_images],
        "source_filtering_mask": [
            file_record(pipeline_dir / "sector_1-MFS-image-pb.fits.mask.fits")
        ],
    }


def _expected_compressed_image_operation_outputs(operation):
    pipeline_dir = Path(operation.pipeline_working_dir)
    return {
        "filtered_skymodel_true_sky": [file_record(pipeline_dir / "sector_1.true_sky.txt")],
        "filtered_skymodel_apparent_sky": [file_record(pipeline_dir / "sector_1.apparent_sky.txt")],
        "pybdsf_catalog": [file_record(pipeline_dir / "sector_1.source_catalog.fits")],
        "sector_diagnostics": [file_record(pipeline_dir / "sector_1.image_diagnostics.json")],
        "sector_offsets": [file_record(pipeline_dir / "sector_1.astrometry_offsets.json")],
        "sector_diagnostic_plots": [[file_record(pipeline_dir / "sector_1.photometry.pdf")]],
        "visibilities": [
            [
                directory_record(pipeline_dir / "sector_1_obs_0_prep.ms"),
                directory_record(pipeline_dir / "sector_1_obs_1_prep.ms"),
            ]
        ],
        "sector_I_images": [
            [
                file_record(pipeline_dir / "sector_1-MFS-I-image.fits.fz"),
                file_record(pipeline_dir / "sector_1-MFS-I-image-pb.fits.fz"),
            ]
        ],
        "sector_extra_images": [
            [
                file_record(pipeline_dir / "sector_1-MFS-I-residual.fits.fz"),
                file_record(pipeline_dir / "sector_1-MFS-I-model-pb.fits.fz"),
                file_record(pipeline_dir / "sector_1-MFS-I-dirty.fits.fz"),
            ]
        ],
        "source_filtering_mask": [
            file_record(pipeline_dir / "sector_1-MFS-I-image-pb.fits.mask.fits")
        ],
        "sector_skymodels": [
            [
                file_record(pipeline_dir / "sector_1-sources.txt"),
                file_record(pipeline_dir / "sector_1-sources-pb.txt"),
            ]
        ],
        "sector_skymodel_image_fits": [
            file_record(pipeline_dir / "sector_1-MFS-filtered-model.fits.fz")
        ],
    }


def _expected_facet_image_operation_outputs(operation):
    outputs = _expected_image_operation_outputs(operation)
    outputs["sector_region_file"] = [
        file_record(Path(operation.pipeline_working_dir) / "sector_1_facets_ds9.reg")
    ]
    return outputs


def _expected_image_cube_operation_outputs(operation):
    pipeline_dir = Path(operation.pipeline_working_dir)
    outputs = _expected_image_operation_outputs(operation)
    outputs["sector_image_cubes"] = [[file_record(pipeline_dir / "sector_1_I_freq_cube.fits")]]
    outputs["sector_image_cube_beams"] = [
        [file_record(pipeline_dir / "sector_1_I_freq_cube.fits_beams.txt")]
    ]
    outputs["sector_image_cube_frequencies"] = [
        [file_record(pipeline_dir / "sector_1_I_freq_cube.fits_frequencies.txt")]
    ]
    return outputs


def _expected_normalize_image_operation_outputs(operation):
    pipeline_dir = Path(operation.pipeline_working_dir)
    outputs = _expected_image_cube_operation_outputs(operation)
    outputs.pop("sector_skymodels")
    outputs["sector_source_catalog"] = [file_record(pipeline_dir / "sector_1_source_catalog.fits")]
    outputs["sector_normalize_h5parm"] = [file_record(pipeline_dir / "sector_1_normalize.h5parm")]
    return outputs


def _materialize_image_operation_outputs(value):
    if isinstance(value, dict) and "class" not in value:
        for item in value.values():
            _materialize_image_operation_outputs(item)
        return
    if isinstance(value, list):
        for item in value:
            _materialize_image_operation_outputs(item)
        return
    if value is None:
        return
    if value["class"] == "Directory":
        Path(value["path"]).mkdir(parents=True, exist_ok=True)
        return
    path = Path(value["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}" if path.suffix == ".json" else "image")


def test_image_command_builders_match_reference_fixtures():
    commands = json.loads((FIXTURE_DIR / "command_reference.json").read_text())

    assert (
        normalize_command(
            build_prepare_imaging_data_command(
                msin="obs_0.ms",
                data_colname="DATA",
                msout="sector_1_obs_0_prep.ms",
                starttime="50000.0",
                ntimes=10,
                phasecenter="'[123.0deg, 45.0deg]'",
                freqstep=4,
                timestep=2,
                beamdir="'[123.0deg, 45.0deg]'",
                numthreads=4,
                steps="[applybeam,shift,avg,bdaavg]",
                maxinterval=8,
                timebase=10.0,
            )
        )
        == commands["image"]["prepare_imaging_data"]
    )
    assert (
        normalize_command(
            build_concat_time_command(
                input_filenames=["sector_1_obs_0_prep.ms", "sector_1_obs_1_prep.ms"],
                output_filename="sector_1_concat.ms",
                data_colname="DATA",
            )
        )
        == commands["image"]["concat_time"]
    )
    assert (
        normalize_command(
            build_blank_image_command(
                mask_filename="sector_1_mask.fits",
                wsclean_imsize=[1024, 1024],
                vertices_file="sector_1.vertices",
                ra=123.0,
                dec=45.0,
                cellsize_deg=0.001,
            )
        )
        == commands["image"]["blank_image"]
    )
    assert (
        normalize_command(
            build_compress_sector_images_command(
                images=[
                    "sector_1-MFS-I-image.fits",
                    "sector_1-MFS-I-image-pb.fits",
                    "sector_1-MFS-I-residual.fits",
                    "sector_1-MFS-I-model-pb.fits",
                    "sector_1-MFS-I-dirty.fits",
                ]
            )
        )
        == commands["image"]["compress_sector_images"]
    )
    assert (
        normalize_command(
            build_make_skymodel_image_command(
                source_catalog="sector_1.apparent_sky.txt",
                reference_image="sector_1-MFS-I-image-pb.fits",
                output_image_name="sector_1-MFS-filtered-model.fits.fz",
            )
        )
        == commands["image"]["make_skymodel_image"]
    )
    assert (
        normalize_command(
            build_wsclean_restore_command(
                residual_image="sector_1-MFS-I-image-pb.fits",
                source_list="bright_sources_pb.txt",
                output_image="sector_1-MFS-I-image-pb.fits",
                numthreads=4,
            )
        )
        == commands["image"]["wsclean_restore"]
    )
    assert (
        normalize_command(
            build_make_image_cube_command(
                input_image_list=[
                    "sector_1-0000-I-image-pb.fits",
                    "sector_1-0001-I-image-pb.fits",
                ],
                output_image="sector_1_I_freq_cube.fits",
            )
        )
        == commands["image"]["make_image_cube"]
    )
    assert (
        normalize_command(
            build_make_catalog_from_image_cube_command(
                cube="sector_1_I_freq_cube.fits",
                cube_beams="sector_1_I_freq_cube.fits_beams.txt",
                cube_frequencies="sector_1_I_freq_cube.fits_frequencies.txt",
                output_catalog="sector_1_source_catalog.fits",
                threshisl=4.0,
                threshpix=5.0,
                ncores=4,
            )
        )
        == commands["image"]["make_catalog_from_image_cube"]
    )
    assert (
        normalize_command(
            build_normalize_flux_scale_command(
                source_catalog="sector_1_source_catalog.fits",
                ms_file="sector_1_concat.ms",
                normalize_h5parm="sector_1_normalize.h5parm",
            )
        )
        == commands["image"]["normalize_flux_scale"]
    )
    assert (
        normalize_command(
            build_wsclean_no_dde_command(
                msin="sector_1_concat.ms",
                name="sector_1",
                mask="sector_1_mask.fits",
                wsclean_imsize=[1024, 1024],
                wsclean_niter=1000,
                wsclean_nmiter=5,
                robust=-0.5,
                min_uv_lambda=80.0,
                max_uv_lambda=1000000.0,
                mgain=0.85,
                multiscale=True,
                save_source_list=True,
                pol="I",
                link_polarizations=False,
                join_polarizations=False,
                skip_final_iteration=True,
                cellsize_deg=0.001,
                channels_out=4,
                deconvolution_channels=2,
                fit_spectral_pol=2,
                taper_arcsec=0.0,
                local_rms_strength=0.0,
                local_rms_window=25.0,
                local_rms_method="rms-with-min",
                wsclean_mem=8.0,
                auto_mask=5.0,
                auto_mask_nmiter=1,
                idg_mode="cpu",
                num_threads=4,
                num_deconvolution_threads=2,
                dd_psf_grid=[1, 1],
                apply_time_frequency_smearing=False,
                temp_dir="sector_1_wsclean_tmp",
            )
        )
        == commands["image"]["wsclean_no_dde"]
    )
    assert (
        normalize_command(
            build_make_region_file_command(
                skymodel="calibration.skymodel",
                ra_mid=123.0,
                dec_mid=45.0,
                width_ra=2.0,
                width_dec=2.5,
                outfile="sector_1_facets_ds9.reg",
            )
        )
        == commands["image"]["make_region_file"]
    )
    assert (
        normalize_command(
            build_wsclean_facets_command(
                msin="sector_1_concat.ms",
                name="sector_1",
                mask="sector_1_mask.fits",
                wsclean_imsize=[1024, 1024],
                wsclean_niter=1000,
                wsclean_nmiter=5,
                robust=-0.5,
                min_uv_lambda=80.0,
                max_uv_lambda=1000000.0,
                mgain=0.85,
                multiscale=True,
                scalar_visibilities=True,
                diagonal_visibilities=False,
                save_source_list=True,
                pol="I",
                link_polarizations=False,
                join_polarizations=False,
                skip_final_iteration=True,
                cellsize_deg=0.001,
                channels_out=4,
                deconvolution_channels=2,
                fit_spectral_pol=2,
                taper_arcsec=0.0,
                local_rms_strength=0.0,
                local_rms_window=25.0,
                local_rms_method="rms-with-min",
                wsclean_mem=8.0,
                auto_mask=5.0,
                auto_mask_nmiter=1,
                idg_mode="cpu",
                num_threads=4,
                num_deconvolution_threads=2,
                dd_psf_grid=[1, 1],
                h5parm="facet-solutions.h5",
                soltabs="phase000",
                region_file="sector_1_facets_ds9.reg",
                num_gridding_threads=3,
                apply_time_frequency_smearing=False,
                shared_facet_reads=True,
                shared_facet_writes=True,
                temp_dir="sector_1_wsclean_tmp",
            )
        )
        == commands["image"]["wsclean_facets"]
    )
    assert (
        normalize_command(
            build_wsclean_screens_command(
                msin="sector_1_concat.ms",
                name="sector_1",
                mask="sector_1_mask.fits",
                wsclean_imsize=[1024, 1024],
                wsclean_niter=1000,
                wsclean_nmiter=5,
                robust=-0.5,
                min_uv_lambda=80.0,
                max_uv_lambda=1000000.0,
                mgain=0.85,
                multiscale=True,
                save_source_list=True,
                pol="I",
                link_polarizations=False,
                join_polarizations=False,
                skip_final_iteration=True,
                cellsize_deg=0.001,
                channels_out=4,
                deconvolution_channels=2,
                fit_spectral_pol=2,
                taper_arcsec=0.0,
                local_rms_strength=0.0,
                local_rms_window=25.0,
                local_rms_method="rms-with-min",
                wsclean_mem=8.0,
                auto_mask=5.0,
                auto_mask_nmiter=1,
                idg_mode="cpu",
                num_threads=4,
                num_deconvolution_threads=2,
                dd_psf_grid=[1, 1],
                interval=[0, 9],
                apply_time_frequency_smearing=False,
                temp_dir="sector_1_wsclean_tmp",
            )
        )
        == commands["image"]["wsclean_screens"]
    )


def test_prepare_imaging_data_command_strips_wrapping_shell_quotes_from_directions():
    command = normalize_command(
        build_prepare_imaging_data_command(
            msin="obs_0.ms",
            data_colname="DATA",
            msout="sector_1_obs_0_prep.ms",
            starttime="50000.0",
            ntimes=10,
            phasecenter="'[123.0deg, 45.0deg]'",
            freqstep=4,
            timestep=2,
            beamdir='"[123.0deg, 45.0deg]"',
            numthreads=4,
            steps="[applybeam,shift,avg,bdaavg]",
        )
    )

    assert "shift.phasecenter=[123.0deg, 45.0deg]" in command
    assert "applybeam.direction=[123.0deg, 45.0deg]" in command


def test_wsclean_command_builders_preserve_full_stokes_options():
    command = normalize_command(
        build_wsclean_no_dde_command(
            msin="sector_1_concat.ms",
            name="sector_1",
            mask="sector_1_mask.fits",
            wsclean_imsize=[1024, 1024],
            wsclean_niter=1000,
            wsclean_nmiter=5,
            robust=-0.5,
            min_uv_lambda=80.0,
            max_uv_lambda=1000000.0,
            mgain=0.85,
            multiscale=True,
            save_source_list=False,
            pol="IQUV",
            link_polarizations=False,
            join_polarizations=True,
            skip_final_iteration=True,
            cellsize_deg=0.001,
            channels_out=4,
            deconvolution_channels=2,
            fit_spectral_pol=2,
            taper_arcsec=0.0,
            local_rms_strength=0.0,
            local_rms_window=25.0,
            local_rms_method="rms-with-min",
            wsclean_mem=8.0,
            auto_mask=5.0,
            auto_mask_nmiter=1,
            idg_mode="cpu",
            num_threads=4,
            num_deconvolution_threads=2,
            dd_psf_grid=[1, 1],
            apply_time_frequency_smearing=False,
            temp_dir="sector_1_wsclean_tmp",
        )
    )

    assert command[command.index("-pol") + 1] == "IQUV"
    assert "-join-polarizations" in command
    assert "-link-polarizations" not in command
    assert "-save-source-list" not in command

    linked_command = normalize_command(
        build_wsclean_no_dde_command(
            msin="sector_1_concat.ms",
            name="sector_1",
            mask="sector_1_mask.fits",
            wsclean_imsize=[1024, 1024],
            wsclean_niter=1000,
            wsclean_nmiter=5,
            robust=-0.5,
            min_uv_lambda=80.0,
            max_uv_lambda=1000000.0,
            mgain=0.85,
            multiscale=True,
            save_source_list=False,
            pol="IQUV",
            link_polarizations="I",
            join_polarizations=False,
            skip_final_iteration=True,
            cellsize_deg=0.001,
            channels_out=4,
            deconvolution_channels=2,
            fit_spectral_pol=2,
            taper_arcsec=0.0,
            local_rms_strength=0.0,
            local_rms_window=25.0,
            local_rms_method="rms-with-min",
            wsclean_mem=8.0,
            auto_mask=5.0,
            auto_mask_nmiter=1,
            idg_mode="cpu",
            num_threads=4,
            num_deconvolution_threads=2,
            dd_psf_grid=[1, 1],
            apply_time_frequency_smearing=False,
            temp_dir="sector_1_wsclean_tmp",
        )
    )

    assert linked_command[linked_command.index("-link-polarizations") + 1] == "I"
    assert "-join-polarizations" not in linked_command


def test_wsclean_mpi_command_builders_use_mpirun_launcher():
    common_kwargs = {
        "msin": "sector_1_concat.ms",
        "name": "sector_1",
        "mask": "sector_1_mask.fits",
        "wsclean_imsize": [1024, 1024],
        "wsclean_niter": 1000,
        "wsclean_nmiter": 5,
        "robust": -0.5,
        "min_uv_lambda": 80.0,
        "max_uv_lambda": 1000000.0,
        "mgain": 0.85,
        "multiscale": True,
        "save_source_list": True,
        "pol": "I",
        "link_polarizations": False,
        "join_polarizations": False,
        "skip_final_iteration": True,
        "cellsize_deg": 0.001,
        "channels_out": 4,
        "deconvolution_channels": 2,
        "fit_spectral_pol": 2,
        "taper_arcsec": 0.0,
        "local_rms_strength": 0.0,
        "local_rms_window": 25.0,
        "local_rms_method": "rms-with-min",
        "wsclean_mem": 8.0,
        "auto_mask": 5.0,
        "auto_mask_nmiter": 1,
        "idg_mode": "cpu",
        "num_threads": 3,
        "num_deconvolution_threads": 2,
        "dd_psf_grid": [1, 1],
        "apply_time_frequency_smearing": False,
        "temp_dir": "sector_1_wsclean_tmp",
    }

    command = normalize_command(build_wsclean_mpi_no_dde_command(mpi_nnodes=2, **common_kwargs))
    assert command[:10] == [
        "mpirun",
        "--bind-to",
        "none",
        "-x",
        "OPENBLAS_NUM_THREADS",
        "-npernode",
        "1",
        "-np",
        "2",
        "wsclean-mp",
    ]
    assert command[command.index("-j") + 1] == "3"
    assert "-apply-primary-beam" in command

    facet_command = normalize_command(
        build_wsclean_mpi_facets_command(
            mpi_nnodes=2,
            **{
                **common_kwargs,
                "scalar_visibilities": True,
                "diagonal_visibilities": False,
                "h5parm": "facet-solutions.h5",
                "soltabs": "phase000",
                "region_file": "sector_1_facets_ds9.reg",
                "num_gridding_threads": 7,
                "shared_facet_reads": True,
                "shared_facet_writes": True,
            },
        )
    )
    assert facet_command[:10] == command[:10]
    assert "-apply-facet-solutions" in facet_command
    assert "-facet-regions" in facet_command
    assert "-parallel-gridding" not in facet_command
    assert "-shared-facet-reads" in facet_command
    assert "-shared-facet-writes" in facet_command

    screen_command = normalize_command(
        build_wsclean_mpi_screens_command(
            mpi_nnodes=2,
            **{
                **common_kwargs,
                "interval": [0, 9],
            },
        )
    )
    assert screen_command[:10] == command[:10]
    assert screen_command[screen_command.index("-gridder") + 1] == "idg"
    assert screen_command[screen_command.index("-aterm-config") + 1] == ATERM_CONFIG_FILENAME
    assert "-apply-primary-beam" not in screen_command
    assert "-apply-facet-beam" not in screen_command


def test_image_support_command_builders_create_expected_tokens():
    assert build_aterm_config_content("/data/screen-solutions.h5") == (
        "aterms = [idgcalsolutions, beam]\n"
        "idgcalsolutions.type = h5parm\n"
        "idgcalsolutions.files = [/data/screen-solutions.h5]\n"
        "idgcalsolutions.update_interval = 8\n"
        "beam.differential = true\n"
        "beam.update_interval = 120\n"
        "beam.usechannelfreq = true\n"
    )
    assert build_make_region_file_command(
        "calibration.skymodel",
        123.0,
        45.0,
        2.0,
        2.5,
        "sector_1_facets_ds9.reg",
    ) == [
        "make_region_file.py",
        "calibration.skymodel",
        "123.0",
        "45.0",
        "2.0",
        "2.5",
        "sector_1_facets_ds9.reg",
        "--enclose_names=True",
    ]
    assert build_compress_sector_images_command(
        ["sector_1-MFS-I-image.fits", "sector_1-MFS-I-image-pb.fits"]
    ) == [
        "fpack",
        "sector_1-MFS-I-image.fits",
        "sector_1-MFS-I-image-pb.fits",
    ]
    assert build_make_skymodel_image_command(
        "sector_1.apparent_sky.txt",
        "sector_1-MFS-I-image-pb.fits",
        "sector_1-MFS-filtered-model.fits.fz",
    ) == [
        "restore_skymodel.py",
        "sector_1.apparent_sky.txt",
        "sector_1-MFS-I-image-pb.fits",
        "sector_1-MFS-filtered-model.fits.fz",
    ]
    assert build_wsclean_restore_command(
        "sector_1-MFS-I-image-pb.fits",
        "bright_sources_pb.txt",
        "sector_1-MFS-I-image-pb.fits",
        4,
    ) == [
        "wsclean",
        "-j",
        "4",
        "-restore-list",
        "sector_1-MFS-I-image-pb.fits",
        "bright_sources_pb.txt",
        "sector_1-MFS-I-image-pb.fits",
    ]
    assert build_make_image_cube_command(
        ["sector_1-0000-I-image-pb.fits", "sector_1-0001-I-image-pb.fits"],
        "sector_1_I_freq_cube.fits",
    ) == [
        "make_image_cube.py",
        "sector_1-0000-I-image-pb.fits,sector_1-0001-I-image-pb.fits",
        "sector_1_I_freq_cube.fits",
    ]
    assert build_make_catalog_from_image_cube_command(
        "sector_1_I_freq_cube.fits",
        "sector_1_I_freq_cube.fits_beams.txt",
        "sector_1_I_freq_cube.fits_frequencies.txt",
        "sector_1_source_catalog.fits",
        4.0,
        5.0,
        4,
    ) == [
        "make_catalog_from_image_cube.py",
        "sector_1_I_freq_cube.fits",
        "sector_1_I_freq_cube.fits_beams.txt",
        "sector_1_I_freq_cube.fits_frequencies.txt",
        "sector_1_source_catalog.fits",
        "--threshisl=4.0",
        "--threshpix=5.0",
        "--ncores=4",
    ]
    assert build_normalize_flux_scale_command(
        "sector_1_source_catalog.fits",
        "sector_1_concat.ms",
        "sector_1_normalize.h5parm",
    ) == [
        "normalize_flux_scale.py",
        "sector_1_source_catalog.fits",
        "sector_1_concat.ms",
        "sector_1_normalize.h5parm",
    ]
    assert build_check_image_beam_command("sector_1-MFS-I-image.fits", 0.0) == [
        "check_image_beam.py",
        "sector_1-MFS-I-image.fits",
        "0.0",
    ]
    assert build_filter_skymodel_command(
        "sector_1-MFS-I-image.fits",
        "sector_1-MFS-I-image-pb.fits",
        "sector_1-sources-pb.txt",
        "sector_1-sources.txt",
        "sector_1",
        "sector_1.vertices",
        ["obs_0.ms", "obs_1.ms"],
        4.0,
        5.0,
        False,
        "bdsf",
        4,
    ) == [
        "filter_skymodel.py",
        "sector_1-MFS-I-image.fits",
        "sector_1-MFS-I-image-pb.fits",
        "sector_1-sources-pb.txt",
        "sector_1-sources.txt",
        "sector_1",
        "sector_1.vertices",
        "obs_0.ms,obs_1.ms",
        "--threshisl=4.0",
        "--threshpix=5.0",
        "--filter_by_mask=False",
        "--source_finder=bdsf",
        "--ncores=4",
    ]
    assert "--bright_true_sky_skymodel=bright_sources_pb.txt" in build_filter_skymodel_command(
        "sector_1-MFS-I-image.fits",
        "sector_1-MFS-I-image-pb.fits",
        "sector_1-sources-pb.txt",
        "sector_1-sources.txt",
        "sector_1",
        "sector_1.vertices",
        ["obs_0.ms", "obs_1.ms"],
        4.0,
        5.0,
        False,
        "bdsf",
        4,
        bright_true_sky_skymodel="bright_sources_pb.txt",
    )
    assert build_calculate_image_diagnostics_command(
        "sector_1-MFS-I-image.fits",
        "sector_1.flat_noise_rms.fits",
        "sector_1-MFS-I-image-pb.fits",
        "sector_1.true_sky_rms.fits",
        "sector_1.source_catalog.fits",
        ["obs_0.ms", "obs_1.ms"],
        ["50000.0", "50010.0"],
        [10, 12],
        "sector_1.image_diagnostics.json",
        "sector_1",
        False,
    ) == [
        "calculate_image_diagnostics.py",
        "sector_1-MFS-I-image.fits",
        "sector_1.flat_noise_rms.fits",
        "sector_1-MFS-I-image-pb.fits",
        "sector_1.true_sky_rms.fits",
        "sector_1.source_catalog.fits",
        "obs_0.ms,obs_1.ms",
        "50000.0,50010.0",
        "10,12",
        "sector_1.image_diagnostics.json",
        "sector_1",
        "--facet_region_file=none",
    ]


def test_image_payload_from_inputs_builds_serializable_no_dde_payload(tmp_path):
    payload = image_payload_from_inputs(_image_input_parms(), tmp_path)

    assert payload["mode"] == "no_dde_stokes_i"
    assert payload["pipeline_working_dir"] == str(tmp_path)
    assert len(payload["sectors"]) == 1
    sector = payload["sectors"][0]
    assert sector["image_name"] == "sector_1"
    assert sector["concat_path"] == str(tmp_path / "sector_1_concat.ms")
    assert sector["mask_path"] == str(tmp_path / "sector_1_mask.fits")
    assert sector["prepare_tasks"][0] == {
        "msin": "/data/obs_0.ms",
        "msout": "sector_1_obs_0_prep.ms",
        "msout_path": str(tmp_path / "sector_1_obs_0_prep.ms"),
        "starttime": "50000.0",
        "ntimes": 10,
        "freqstep": 4,
        "timestep": 2,
        "maxinterval": 8,
    }


def test_image_payload_from_inputs_builds_serializable_facet_payload(tmp_path):
    payload = image_payload_from_inputs(_facet_image_input_parms(), tmp_path, use_facets=True)

    assert payload["mode"] == "facet_stokes_i"
    sector = payload["sectors"][0]
    assert sector["use_facets"] is True
    assert sector["h5parm"] == "/data/facet-solutions.h5"
    assert sector["facet_skymodel"] == "/data/calibration.skymodel"
    assert sector["facet_region_filename"] == "sector_1_facets_ds9.reg"
    assert sector["facet_region_path"] == str(tmp_path / "sector_1_facets_ds9.reg")
    assert sector["ra_mid"] == 123.0
    assert sector["dec_mid"] == 45.0
    assert sector["width_ra"] == 2.0
    assert sector["width_dec"] == 2.5
    assert sector["soltabs"] == "phase000"
    assert sector["parallel_gridding_threads"] == 3
    assert sector["scalar_visibilities"] is True
    assert sector["diagonal_visibilities"] is False
    assert sector["shared_facet_reads"] is True
    assert sector["shared_facet_writes"] is True


def test_image_payload_from_inputs_builds_serializable_screen_payload(tmp_path):
    payload = image_payload_from_inputs(_screens_image_input_parms(), tmp_path, apply_screens=True)

    assert payload["mode"] == "screens_stokes_i"
    sector = payload["sectors"][0]
    assert sector["apply_screens"] is True
    assert sector["use_facets"] is False
    assert sector["h5parm"] == "/data/screen-solutions.h5"
    assert sector["interval"] == [0, 9]
    assert sector["prepare_tasks"][0]["maxinterval"] is None


def test_image_payload_from_inputs_builds_full_stokes_payload(tmp_path):
    payload = image_payload_from_inputs(_full_stokes_image_input_parms(), tmp_path)

    assert payload["mode"] == "no_dde_full_stokes"
    sector = payload["sectors"][0]
    assert sector["pol"] == "IQUV"
    assert sector["save_source_list"] is False
    assert sector["join_polarizations"] is True
    assert sector["link_polarizations"] is False


def test_image_payload_from_inputs_builds_mpi_payload(tmp_path):
    payload = image_payload_from_inputs(_mpi_image_input_parms(), tmp_path, use_mpi=True)

    assert payload["mode"] == "no_dde_stokes_i"
    assert payload["use_mpi"] is True
    sector = payload["sectors"][0]
    assert sector["use_mpi"] is True
    assert sector["mpi_nnodes"] == 2
    assert sector["mpi_cpus_per_task"] == 3


def test_image_payload_from_inputs_builds_compressed_payload(tmp_path):
    payload = image_payload_from_inputs(_image_input_parms(), tmp_path, compress_images=True)

    sector = payload["sectors"][0]
    assert payload["mode"] == "no_dde_stokes_i"
    assert sector["compress_images"] is True
    assert sector["save_filtered_model_image"] is False


def test_image_payload_from_inputs_builds_filtered_model_payload(tmp_path):
    payload = image_payload_from_inputs(_filtered_model_image_input_parms(), tmp_path)

    sector = payload["sectors"][0]
    assert sector["save_filtered_model_image"] is True
    assert sector["filtered_model_image_filename"] == "sector_1-MFS-filtered-model.fits.fz"
    assert sector["filtered_model_image_path"] == str(
        tmp_path / "sector_1-MFS-filtered-model.fits.fz"
    )


def test_image_payload_from_inputs_builds_bright_peeling_payload(tmp_path):
    payload = image_payload_from_inputs(_bright_peeling_image_input_parms(), tmp_path)

    sector = payload["sectors"][0]
    assert sector["peel_bright_sources"] is True
    assert sector["bright_skymodel_pb"] == "/data/bright_sources_pb.txt"


def test_image_payload_from_inputs_builds_image_cube_payload(tmp_path):
    payload = image_payload_from_inputs(_image_cube_input_parms(), tmp_path, make_image_cube=True)

    sector = payload["sectors"][0]
    assert sector["make_image_cube"] is True
    assert sector["normalize_flux_scale"] is False
    assert sector["image_I_cube_filename"] == "sector_1_I_freq_cube.fits"
    assert sector["image_I_cube_path"] == str(tmp_path / "sector_1_I_freq_cube.fits")
    assert sector["image_cube_specs"] == [
        {
            "pol": "I",
            "filename": "sector_1_I_freq_cube.fits",
            "path": str(tmp_path / "sector_1_I_freq_cube.fits"),
        }
    ]


def test_image_payload_from_inputs_builds_full_stokes_image_cube_payload(tmp_path):
    payload = image_payload_from_inputs(
        _full_stokes_image_cube_input_parms(), tmp_path, make_image_cube=True
    )

    assert payload["mode"] == "no_dde_full_stokes"
    assert payload["sectors"][0]["image_cube_specs"] == [
        {
            "pol": "I",
            "filename": "sector_1_I_freq_cube.fits",
            "path": str(tmp_path / "sector_1_I_freq_cube.fits"),
        },
        {
            "pol": "Q",
            "filename": "sector_1_Q_freq_cube.fits",
            "path": str(tmp_path / "sector_1_Q_freq_cube.fits"),
        },
        {
            "pol": "U",
            "filename": "sector_1_U_freq_cube.fits",
            "path": str(tmp_path / "sector_1_U_freq_cube.fits"),
        },
        {
            "pol": "V",
            "filename": "sector_1_V_freq_cube.fits",
            "path": str(tmp_path / "sector_1_V_freq_cube.fits"),
        },
    ]


def test_image_payload_from_inputs_builds_normalization_payload(tmp_path):
    payload = image_payload_from_inputs(
        _normalize_image_input_parms(),
        tmp_path,
        make_image_cube=True,
        normalize_flux_scale=True,
    )

    sector = payload["sectors"][0]
    assert sector["make_image_cube"] is True
    assert sector["normalize_flux_scale"] is True
    assert sector["save_source_list"] is False
    assert sector["output_source_catalog_filename"] == "sector_1_source_catalog.fits"
    assert sector["output_source_catalog_path"] == str(tmp_path / "sector_1_source_catalog.fits")
    assert sector["output_normalize_h5parm_filename"] == "sector_1_normalize.h5parm"
    assert sector["output_normalize_h5parm_path"] == str(tmp_path / "sector_1_normalize.h5parm")


def test_image_payload_from_inputs_rejects_unsupported_modes(tmp_path):
    with pytest.raises(ValueError, match="cannot both"):
        image_payload_from_inputs(
            _facet_image_input_parms(), tmp_path, apply_screens=True, use_facets=True
        )

    input_parms = _image_input_parms()
    input_parms["pol"] = {"invalid": "pol"}
    with pytest.raises(ValueError, match="pol must be"):
        image_payload_from_inputs(input_parms, tmp_path)

    with pytest.raises(ValueError, match="mpi_nnodes"):
        image_payload_from_inputs(_image_input_parms(), tmp_path, use_mpi=True)

    with pytest.raises(ValueError, match="requires make_image_cube"):
        image_payload_from_inputs(
            _normalize_image_input_parms(), tmp_path, normalize_flux_scale=True
        )

    input_parms = _image_input_parms()
    input_parms["peel_bright_sources"] = True
    with pytest.raises(ValueError, match="bright_skymodel_pb"):
        image_payload_from_inputs(input_parms, tmp_path)


def test_image_payload_from_inputs_requires_facet_inputs(tmp_path):
    input_parms = _facet_image_input_parms()
    input_parms["h5parm"] = None
    with pytest.raises(ValueError, match="h5parm"):
        image_payload_from_inputs(input_parms, tmp_path, use_facets=True)

    input_parms = _facet_image_input_parms()
    del input_parms["soltabs"]
    with pytest.raises(ValueError, match="soltabs"):
        image_payload_from_inputs(input_parms, tmp_path, use_facets=True)


def test_image_payload_from_inputs_requires_screen_inputs(tmp_path):
    input_parms = _screens_image_input_parms()
    input_parms["h5parm"] = None
    with pytest.raises(ValueError, match="h5parm"):
        image_payload_from_inputs(input_parms, tmp_path, apply_screens=True)

    input_parms = _screens_image_input_parms()
    input_parms["interval"] = [0]
    with pytest.raises(ValueError, match="interval"):
        image_payload_from_inputs(input_parms, tmp_path, apply_screens=True)


def test_image_payload_from_inputs_requires_filtered_model_filename(tmp_path):
    input_parms = _filtered_model_image_input_parms()
    input_parms["filtered_model_image_name"] = ["nested/sector_1-filtered.fits.fz"]
    with pytest.raises(ValueError, match="filtered_model_image_name"):
        image_payload_from_inputs(input_parms, tmp_path)


def test_run_image_flow_executes_no_dde_commands_and_returns_records(
    tmp_path, monkeypatch, fake_image_shell_operation_cls
):
    published = []
    fits_published = []

    def fake_publish_plot_file_records(records, root_dir):
        published.append(([Path(record["path"]).name for record in records], root_dir))
        return []

    def fake_publish_fits_image_artifacts(records, root_dir):
        fits_published.append(([Path(record["path"]).name for record in records], root_dir))
        return []

    monkeypatch.setattr(
        image_sector_module,
        "publish_plot_file_records",
        fake_publish_plot_file_records,
    )
    monkeypatch.setattr(
        image_sector_module,
        "publish_fits_image_artifacts",
        fake_publish_fits_image_artifacts,
    )

    outputs = run_image_flow(
        image_payload_from_inputs(_image_input_parms(), tmp_path),
        execution_config=ExecutionConfig(task_runner="sync"),
        shell_operation_cls=fake_image_shell_operation_cls,
    )

    assert outputs["sector_I_images"] == [
        [
            file_record(tmp_path / "sector_1-MFS-I-image.fits"),
            file_record(tmp_path / "sector_1-MFS-I-image-pb.fits"),
        ]
    ]
    assert outputs["visibilities"] == [
        [
            directory_record(tmp_path / "sector_1_obs_0_prep.ms"),
            directory_record(tmp_path / "sector_1_obs_1_prep.ms"),
        ]
    ]
    assert outputs["filtered_skymodel_true_sky"] == [
        file_record(tmp_path / "sector_1.true_sky.txt")
    ]
    assert outputs["filtered_skymodel_apparent_sky"] == [
        file_record(tmp_path / "sector_1.apparent_sky.txt")
    ]
    assert outputs["pybdsf_catalog"] == [file_record(tmp_path / "sector_1.source_catalog.fits")]
    assert outputs["sector_diagnostics"] == [
        file_record(tmp_path / "sector_1.image_diagnostics.json")
    ]
    assert (
        ["sector_1.image_diagnostics.json"],
        str(tmp_path),
    ) in published
    assert (["sector_1.photometry.pdf"], str(tmp_path)) in published
    assert len(fits_published) == 1
    assert fits_published[0][1] == str(tmp_path)
    assert "sector_1-MFS-I-image.fits" in fits_published[0][0]
    assert "sector_1-MFS-I-image-pb.fits" in fits_published[0][0]
    assert "sector_1.flat_noise_rms.fits" in fits_published[0][0]
    assert "sector_1.true_sky_rms.fits" in fits_published[0][0]
    validate_output_record(outputs["sector_I_images"])
    command_names = [
        shlex.split(instance.kwargs["commands"][0])[0]
        for instance in fake_image_shell_operation_cls.instances
    ]
    assert command_names == [
        "DP3",
        "DP3",
        "concat_ms.py",
        "blank_image.py",
        "wsclean",
        "check_image_beam.py",
        "check_image_beam.py",
        "filter_skymodel.py",
        "calculate_image_diagnostics.py",
    ]


def test_run_image_flow_rejects_invalid_prepare_task_payload(
    tmp_path, fake_image_shell_operation_cls
):
    payload = image_payload_from_inputs(_image_input_parms(), tmp_path)
    payload["sectors"][0]["prepare_tasks"] = ["not-a-task"]

    with pytest.raises(ValueError, match="prepare_tasks"):
        run_image_flow(
            payload,
            execution_config=ExecutionConfig(task_runner="sync"),
            shell_operation_cls=fake_image_shell_operation_cls,
        )

    assert fake_image_shell_operation_cls.instances == []


def test_run_image_flow_allows_missing_source_filtering_mask(
    tmp_path, fake_image_shell_operation_cls
):
    class NoFilterMaskShellOperation(fake_image_shell_operation_cls):
        instances = []

        def run(self):
            tokens = shlex.split(self.kwargs["commands"][0])
            cwd = Path(self.kwargs["working_dir"])
            if tokens[0] == "filter_skymodel.py":
                output_root = tokens[5]
                for suffix in [
                    ".true_sky.txt",
                    ".apparent_sky.txt",
                    ".flat_noise_rms.fits",
                    ".true_sky_rms.fits",
                    ".source_catalog.fits",
                ]:
                    (cwd / f"{output_root}{suffix}").write_text("filter")
                (cwd / f"{output_root}.image_diagnostics.json").write_text("{}")
                return "OK"
            return super().run()

    outputs = run_image_flow(
        image_payload_from_inputs(_image_input_parms(), tmp_path),
        execution_config=ExecutionConfig(task_runner="sync"),
        shell_operation_cls=NoFilterMaskShellOperation,
    )

    assert outputs["source_filtering_mask"] == [None]
    validate_output_record(outputs["source_filtering_mask"], allow_none=True)


def test_run_image_flow_reuses_existing_wsclean_products_on_restart(
    tmp_path, fake_image_shell_operation_cls
):
    for ms_name in ["sector_1_obs_0_prep.ms", "sector_1_obs_1_prep.ms", "sector_1_concat.ms"]:
        (tmp_path / ms_name).mkdir()
    for filename in [
        "sector_1_mask.fits",
        "sector_1-MFS-I-image.fits",
        "sector_1-MFS-I-image-pb.fits",
        "sector_1-sources.txt",
        "sector_1-sources-pb.txt",
    ]:
        (tmp_path / filename).write_text("existing")

    outputs = run_image_flow(
        image_payload_from_inputs(_image_input_parms(), tmp_path),
        execution_config=ExecutionConfig(task_runner="sync"),
        shell_operation_cls=fake_image_shell_operation_cls,
    )

    assert outputs["sector_I_images"] == [
        [
            file_record(tmp_path / "sector_1-MFS-I-image.fits"),
            file_record(tmp_path / "sector_1-MFS-I-image-pb.fits"),
        ]
    ]
    command_names = [
        shlex.split(instance.kwargs["commands"][0])[0]
        for instance in fake_image_shell_operation_cls.instances
    ]
    assert command_names == [
        "filter_skymodel.py",
        "calculate_image_diagnostics.py",
    ]


def test_run_image_flow_restores_bright_sources_before_filtering(
    tmp_path, fake_image_shell_operation_cls
):
    outputs = run_image_flow(
        image_payload_from_inputs(_bright_peeling_image_input_parms(), tmp_path),
        execution_config=ExecutionConfig(task_runner="sync"),
        shell_operation_cls=fake_image_shell_operation_cls,
    )

    assert outputs["sector_I_images"] == [
        [
            file_record(tmp_path / "sector_1-MFS-I-image.fits"),
            file_record(tmp_path / "sector_1-MFS-I-image-pb.fits"),
        ]
    ]
    commands = [
        shlex.split(instance.kwargs["commands"][0])
        for instance in fake_image_shell_operation_cls.instances
    ]
    command_names = [command[0] for command in commands]
    assert command_names == [
        "DP3",
        "DP3",
        "concat_ms.py",
        "blank_image.py",
        "wsclean",
        "wsclean",
        "wsclean",
        "check_image_beam.py",
        "check_image_beam.py",
        "filter_skymodel.py",
        "calculate_image_diagnostics.py",
    ]
    restore_commands = [
        command for command in commands if command[0] == "wsclean" and "-restore-list" in command
    ]
    assert restore_commands == [
        [
            "wsclean",
            "-j",
            "4",
            "-restore-list",
            str(tmp_path / "sector_1-MFS-I-image-pb.fits"),
            "/data/bright_sources_pb.txt",
            "sector_1-MFS-I-image-pb.fits",
        ],
        [
            "wsclean",
            "-j",
            "4",
            "-restore-list",
            str(tmp_path / "sector_1-MFS-I-image.fits"),
            "/data/bright_sources_pb.txt",
            "sector_1-MFS-I-image.fits",
        ],
    ]
    filter_command = next(command for command in commands if command[0] == "filter_skymodel.py")
    assert "--bright_true_sky_skymodel=/data/bright_sources_pb.txt" in filter_command


def test_run_image_flow_executes_facet_commands_and_returns_region_file(
    tmp_path, fake_image_shell_operation_cls
):
    outputs = run_image_flow(
        image_payload_from_inputs(_facet_image_input_parms(), tmp_path, use_facets=True),
        execution_config=ExecutionConfig(task_runner="sync"),
        shell_operation_cls=fake_image_shell_operation_cls,
    )

    assert outputs["sector_region_file"] == [file_record(tmp_path / "sector_1_facets_ds9.reg")]
    assert outputs["sector_I_images"] == [
        [
            file_record(tmp_path / "sector_1-MFS-I-image.fits"),
            file_record(tmp_path / "sector_1-MFS-I-image-pb.fits"),
        ]
    ]
    command_names = [
        shlex.split(instance.kwargs["commands"][0])[0]
        for instance in fake_image_shell_operation_cls.instances
    ]
    assert command_names == [
        "DP3",
        "DP3",
        "concat_ms.py",
        "blank_image.py",
        "make_region_file.py",
        "wsclean",
        "check_image_beam.py",
        "check_image_beam.py",
        "filter_skymodel.py",
        "calculate_image_diagnostics.py",
    ]
    facet_command = next(
        shlex.split(instance.kwargs["commands"][0])
        for instance in fake_image_shell_operation_cls.instances
        if shlex.split(instance.kwargs["commands"][0])[0] == "wsclean"
    )
    assert "-apply-facet-beam" in facet_command
    assert "-apply-facet-solutions" in facet_command
    assert "-scalar-visibilities" in facet_command
    assert "-shared-facet-reads" in facet_command
    assert "-shared-facet-writes" in facet_command
    assert "-diagonal-visibilities" not in facet_command


def test_run_image_flow_executes_screen_commands_and_writes_aterm_config(
    tmp_path, fake_image_shell_operation_cls
):
    outputs = run_image_flow(
        image_payload_from_inputs(_screens_image_input_parms(), tmp_path, apply_screens=True),
        execution_config=ExecutionConfig(task_runner="sync"),
        shell_operation_cls=fake_image_shell_operation_cls,
    )

    assert outputs["sector_I_images"] == [
        [
            file_record(tmp_path / "sector_1-MFS-I-image.fits"),
            file_record(tmp_path / "sector_1-MFS-I-image-pb.fits"),
        ]
    ]
    aterm_config = tmp_path / ATERM_CONFIG_FILENAME
    assert aterm_config.read_text() == build_aterm_config_content("/data/screen-solutions.h5")
    command_names = [
        shlex.split(instance.kwargs["commands"][0])[0]
        for instance in fake_image_shell_operation_cls.instances
    ]
    assert command_names == [
        "DP3",
        "DP3",
        "concat_ms.py",
        "blank_image.py",
        "wsclean",
        "check_image_beam.py",
        "check_image_beam.py",
        "filter_skymodel.py",
        "calculate_image_diagnostics.py",
    ]
    screen_command = next(
        shlex.split(instance.kwargs["commands"][0])
        for instance in fake_image_shell_operation_cls.instances
        if shlex.split(instance.kwargs["commands"][0])[0] == "wsclean"
    )
    assert screen_command[screen_command.index("-gridder") + 1] == "idg"
    assert screen_command[screen_command.index("-aterm-config") + 1] == ATERM_CONFIG_FILENAME
    assert screen_command[screen_command.index("-interval") + 1 :][0:2] == ["0", "9"]
    assert "-apply-primary-beam" not in screen_command
    assert "-apply-facet-beam" not in screen_command


def test_run_image_flow_supports_full_stokes_no_dde(tmp_path, fake_image_shell_operation_cls):
    outputs = run_image_flow(
        image_payload_from_inputs(_full_stokes_image_input_parms(), tmp_path),
        execution_config=ExecutionConfig(task_runner="sync"),
        shell_operation_cls=fake_image_shell_operation_cls,
    )

    assert outputs["sector_I_images"] == [
        [
            file_record(tmp_path / "sector_1-MFS-image.fits"),
            file_record(tmp_path / "sector_1-MFS-image-pb.fits"),
        ]
    ]
    extra_images = outputs["sector_extra_images"][0]
    assert file_record(tmp_path / "sector_1-MFS-Q-image.fits") in extra_images
    assert file_record(tmp_path / "sector_1-MFS-U-image-pb.fits") in extra_images
    assert file_record(tmp_path / "sector_1-MFS-V-dirty.fits") in extra_images
    assert outputs["source_filtering_mask"] == [
        file_record(tmp_path / "sector_1-MFS-image-pb.fits.mask.fits")
    ]
    assert "sector_skymodels" not in outputs
    wsclean_command = next(
        shlex.split(instance.kwargs["commands"][0])
        for instance in fake_image_shell_operation_cls.instances
        if shlex.split(instance.kwargs["commands"][0])[0] == "wsclean"
    )
    assert wsclean_command[wsclean_command.index("-pol") + 1] == "IQUV"
    assert "-join-polarizations" in wsclean_command
    assert "-save-source-list" not in wsclean_command


def test_run_image_flow_supports_linked_full_stokes(tmp_path, fake_image_shell_operation_cls):
    run_image_flow(
        image_payload_from_inputs(_linked_full_stokes_image_input_parms(), tmp_path),
        execution_config=ExecutionConfig(task_runner="sync"),
        shell_operation_cls=fake_image_shell_operation_cls,
    )

    wsclean_command = next(
        shlex.split(instance.kwargs["commands"][0])
        for instance in fake_image_shell_operation_cls.instances
        if shlex.split(instance.kwargs["commands"][0])[0] == "wsclean"
    )
    assert wsclean_command[wsclean_command.index("-link-polarizations") + 1] == "I"
    assert "-join-polarizations" not in wsclean_command


def test_run_image_flow_supports_mpi_no_dde(tmp_path, fake_image_shell_operation_cls):
    outputs = run_image_flow(
        image_payload_from_inputs(_mpi_image_input_parms(), tmp_path, use_mpi=True),
        execution_config=ExecutionConfig(task_runner="sync", max_nodes=2, cpus_per_task=3),
        shell_operation_cls=fake_image_shell_operation_cls,
    )

    assert outputs["sector_I_images"] == [
        [
            file_record(tmp_path / "sector_1-MFS-I-image.fits"),
            file_record(tmp_path / "sector_1-MFS-I-image-pb.fits"),
        ]
    ]
    mpi_instance = next(
        instance
        for instance in fake_image_shell_operation_cls.instances
        if shlex.split(instance.kwargs["commands"][0])[0] == "mpirun"
    )
    mpi_command = shlex.split(mpi_instance.kwargs["commands"][0])
    assert mpi_command[:10] == [
        "mpirun",
        "--bind-to",
        "none",
        "-x",
        "OPENBLAS_NUM_THREADS",
        "-npernode",
        "1",
        "-np",
        "2",
        "wsclean-mp",
    ]
    assert mpi_command[mpi_command.index("-j") + 1] == "3"
    assert mpi_instance.kwargs["env"] == {
        "OMP_NUM_THREADS": "3",
        "OPENBLAS_NUM_THREADS": "3",
    }


def test_run_image_flow_rejects_oversubscribed_mpi_wsclean(
    tmp_path, fake_image_shell_operation_cls
):
    with pytest.raises(ValueError, match="requests 2 MPI processes"):
        run_image_flow(
            image_payload_from_inputs(_mpi_image_input_parms(), tmp_path, use_mpi=True),
            execution_config=ExecutionConfig(task_runner="sync", max_nodes=1, cpus_per_task=3),
            shell_operation_cls=fake_image_shell_operation_cls,
        )

    command_names = [
        shlex.split(instance.kwargs["commands"][0])[0]
        for instance in fake_image_shell_operation_cls.instances
    ]
    assert "mpirun" not in command_names


def test_run_image_flow_supports_mpi_facets(tmp_path, fake_image_shell_operation_cls):
    outputs = run_image_flow(
        image_payload_from_inputs(
            _mpi_facet_image_input_parms(), tmp_path, use_facets=True, use_mpi=True
        ),
        execution_config=ExecutionConfig(task_runner="sync", max_nodes=2, cpus_per_task=3),
        shell_operation_cls=fake_image_shell_operation_cls,
    )

    assert outputs["sector_region_file"] == [file_record(tmp_path / "sector_1_facets_ds9.reg")]
    mpi_command = next(
        shlex.split(instance.kwargs["commands"][0])
        for instance in fake_image_shell_operation_cls.instances
        if shlex.split(instance.kwargs["commands"][0])[0] == "mpirun"
    )
    assert "wsclean-mp" in mpi_command
    assert "-apply-facet-solutions" in mpi_command
    assert "-facet-regions" in mpi_command
    assert "-parallel-gridding" not in mpi_command
    assert "-shared-facet-reads" in mpi_command
    assert "-shared-facet-writes" in mpi_command


def test_run_image_flow_supports_mpi_screens(tmp_path, fake_image_shell_operation_cls):
    outputs = run_image_flow(
        image_payload_from_inputs(
            _mpi_screens_image_input_parms(), tmp_path, apply_screens=True, use_mpi=True
        ),
        execution_config=ExecutionConfig(task_runner="sync", max_nodes=2, cpus_per_task=3),
        shell_operation_cls=fake_image_shell_operation_cls,
    )

    assert outputs["sector_I_images"] == [
        [
            file_record(tmp_path / "sector_1-MFS-I-image.fits"),
            file_record(tmp_path / "sector_1-MFS-I-image-pb.fits"),
        ]
    ]
    mpi_command = next(
        shlex.split(instance.kwargs["commands"][0])
        for instance in fake_image_shell_operation_cls.instances
        if shlex.split(instance.kwargs["commands"][0])[0] == "mpirun"
    )
    assert "wsclean-mp" in mpi_command
    assert mpi_command[mpi_command.index("-gridder") + 1] == "idg"
    assert mpi_command[mpi_command.index("-aterm-config") + 1] == ATERM_CONFIG_FILENAME


def test_run_image_flow_returns_compressed_image_outputs(tmp_path, fake_image_shell_operation_cls):
    outputs = run_image_flow(
        image_payload_from_inputs(_image_input_parms(), tmp_path, compress_images=True),
        execution_config=ExecutionConfig(task_runner="sync"),
        shell_operation_cls=fake_image_shell_operation_cls,
    )

    assert outputs["sector_I_images"] == [
        [
            file_record(tmp_path / "sector_1-MFS-I-image.fits.fz"),
            file_record(tmp_path / "sector_1-MFS-I-image-pb.fits.fz"),
        ]
    ]
    assert outputs["sector_extra_images"] == [
        [
            file_record(tmp_path / "sector_1-MFS-I-residual.fits.fz"),
            file_record(tmp_path / "sector_1-MFS-I-model-pb.fits.fz"),
            file_record(tmp_path / "sector_1-MFS-I-dirty.fits.fz"),
        ]
    ]
    command_names = [
        shlex.split(instance.kwargs["commands"][0])[0]
        for instance in fake_image_shell_operation_cls.instances
    ]
    assert command_names == [
        "DP3",
        "DP3",
        "concat_ms.py",
        "blank_image.py",
        "wsclean",
        "check_image_beam.py",
        "check_image_beam.py",
        "filter_skymodel.py",
        "calculate_image_diagnostics.py",
        "fpack",
    ]


def test_run_image_flow_returns_filtered_model_image(tmp_path, fake_image_shell_operation_cls):
    outputs = run_image_flow(
        image_payload_from_inputs(_filtered_model_image_input_parms(), tmp_path),
        execution_config=ExecutionConfig(task_runner="sync"),
        shell_operation_cls=fake_image_shell_operation_cls,
    )

    assert outputs["sector_skymodel_image_fits"] == [
        file_record(tmp_path / "sector_1-MFS-filtered-model.fits.fz")
    ]
    command_names = [
        shlex.split(instance.kwargs["commands"][0])[0]
        for instance in fake_image_shell_operation_cls.instances
    ]
    assert command_names == [
        "DP3",
        "DP3",
        "concat_ms.py",
        "blank_image.py",
        "wsclean",
        "check_image_beam.py",
        "check_image_beam.py",
        "filter_skymodel.py",
        "restore_skymodel.py",
        "calculate_image_diagnostics.py",
    ]


def test_run_image_flow_supports_clean_disabled_stokes_i(tmp_path, fake_image_shell_operation_cls):
    outputs = run_image_flow(
        image_payload_from_inputs(_clean_disabled_image_input_parms(), tmp_path),
        execution_config=ExecutionConfig(task_runner="sync"),
        shell_operation_cls=fake_image_shell_operation_cls,
    )

    assert outputs["sector_I_images"] == [
        [
            file_record(tmp_path / "sector_1-MFS-I-image.fits"),
            file_record(tmp_path / "sector_1-MFS-I-image-pb.fits"),
        ]
    ]
    wsclean_command = next(
        shlex.split(instance.kwargs["commands"][0])
        for instance in fake_image_shell_operation_cls.instances
        if shlex.split(instance.kwargs["commands"][0])[0] == "wsclean"
    )
    assert wsclean_command[wsclean_command.index("-niter") + 1] == "0"


def test_run_image_flow_cleans_isolated_wsclean_temp_dirs(tmp_path, fake_image_shell_operation_cls):
    outputs = run_image_flow(
        image_payload_from_inputs(_two_sector_image_input_parms(), tmp_path),
        execution_config=ExecutionConfig(task_runner="sync"),
        shell_operation_cls=fake_image_shell_operation_cls,
    )

    assert len(outputs["sector_I_images"]) == 2
    temp_dirs = [
        Path(command[command.index("-temp-dir") + 1])
        for command in [
            shlex.split(instance.kwargs["commands"][0])
            for instance in fake_image_shell_operation_cls.instances
            if shlex.split(instance.kwargs["commands"][0])[0] == "wsclean"
        ]
    ]
    assert temp_dirs == [
        tmp_path / "sector_1_wsclean_tmp",
        tmp_path / "sector_2_wsclean_tmp",
    ]
    assert len(set(temp_dirs)) == 2
    assert all(not temp_dir.exists() for temp_dir in temp_dirs)


def test_run_image_flow_cleans_wsclean_temp_dir_on_failure(
    tmp_path, fake_image_shell_operation_cls
):
    class FailingWscleanShellOperation(fake_image_shell_operation_cls):
        instances = []

        def run(self):
            tokens = shlex.split(self.kwargs["commands"][0])
            if tokens[0] == "wsclean":
                temp_dir = Path(tokens[tokens.index("-temp-dir") + 1])
                temp_dir.mkdir(parents=True, exist_ok=True)
                (temp_dir / "wsclean.tmp").write_text("temporary")
                raise RuntimeError("wsclean failed")
            return super().run()

    with pytest.raises(RuntimeError, match="wsclean failed"):
        run_image_flow(
            image_payload_from_inputs(_image_input_parms(), tmp_path),
            execution_config=ExecutionConfig(task_runner="sync"),
            shell_operation_cls=FailingWscleanShellOperation,
        )

    assert not (tmp_path / "sector_1_wsclean_tmp").exists()


def test_run_image_flow_returns_image_cube_outputs(tmp_path, fake_image_shell_operation_cls):
    outputs = run_image_flow(
        image_payload_from_inputs(_image_cube_input_parms(), tmp_path, make_image_cube=True),
        execution_config=ExecutionConfig(task_runner="sync"),
        shell_operation_cls=fake_image_shell_operation_cls,
    )

    assert outputs["sector_image_cubes"] == [[file_record(tmp_path / "sector_1_I_freq_cube.fits")]]
    assert outputs["sector_image_cube_beams"] == [
        [file_record(tmp_path / "sector_1_I_freq_cube.fits_beams.txt")]
    ]
    assert outputs["sector_image_cube_frequencies"] == [
        [file_record(tmp_path / "sector_1_I_freq_cube.fits_frequencies.txt")]
    ]
    command_names = [
        shlex.split(instance.kwargs["commands"][0])[0]
        for instance in fake_image_shell_operation_cls.instances
    ]
    assert command_names == [
        "DP3",
        "DP3",
        "concat_ms.py",
        "blank_image.py",
        "wsclean",
        "make_image_cube.py",
        "check_image_beam.py",
        "check_image_beam.py",
        "filter_skymodel.py",
        "calculate_image_diagnostics.py",
    ]


def test_run_image_flow_returns_full_stokes_image_cube_outputs(
    tmp_path, fake_image_shell_operation_cls
):
    outputs = run_image_flow(
        image_payload_from_inputs(
            _full_stokes_image_cube_input_parms(), tmp_path, make_image_cube=True
        ),
        execution_config=ExecutionConfig(task_runner="sync"),
        shell_operation_cls=fake_image_shell_operation_cls,
    )

    assert outputs["sector_image_cubes"] == [
        [
            file_record(tmp_path / "sector_1_I_freq_cube.fits"),
            file_record(tmp_path / "sector_1_Q_freq_cube.fits"),
            file_record(tmp_path / "sector_1_U_freq_cube.fits"),
            file_record(tmp_path / "sector_1_V_freq_cube.fits"),
        ]
    ]
    make_cube_commands = [
        shlex.split(instance.kwargs["commands"][0])
        for instance in fake_image_shell_operation_cls.instances
        if shlex.split(instance.kwargs["commands"][0])[0] == "make_image_cube.py"
    ]
    assert [command[2] for command in make_cube_commands] == [
        "sector_1_I_freq_cube.fits",
        "sector_1_Q_freq_cube.fits",
        "sector_1_U_freq_cube.fits",
        "sector_1_V_freq_cube.fits",
    ]
    assert make_cube_commands[0][1].split(",") == [
        str(tmp_path / "sector_1-0000-image-pb.fits"),
        str(tmp_path / "sector_1-0001-image-pb.fits"),
        str(tmp_path / "sector_1-0002-image-pb.fits"),
        str(tmp_path / "sector_1-0003-image-pb.fits"),
    ]
    assert make_cube_commands[1][1].split(",") == [
        str(tmp_path / "sector_1-0000-Q-image-pb.fits"),
        str(tmp_path / "sector_1-0001-Q-image-pb.fits"),
        str(tmp_path / "sector_1-0002-Q-image-pb.fits"),
        str(tmp_path / "sector_1-0003-Q-image-pb.fits"),
    ]


def test_run_image_flow_returns_normalization_outputs(tmp_path, fake_image_shell_operation_cls):
    outputs = run_image_flow(
        image_payload_from_inputs(
            _normalize_image_input_parms(),
            tmp_path,
            make_image_cube=True,
            normalize_flux_scale=True,
        ),
        execution_config=ExecutionConfig(task_runner="sync"),
        shell_operation_cls=fake_image_shell_operation_cls,
    )

    assert outputs["sector_source_catalog"] == [
        file_record(tmp_path / "sector_1_source_catalog.fits")
    ]
    assert outputs["sector_normalize_h5parm"] == [
        file_record(tmp_path / "sector_1_normalize.h5parm")
    ]
    assert "sector_skymodels" not in outputs
    command_names = [
        shlex.split(instance.kwargs["commands"][0])[0]
        for instance in fake_image_shell_operation_cls.instances
    ]
    assert command_names == [
        "DP3",
        "DP3",
        "concat_ms.py",
        "blank_image.py",
        "wsclean",
        "make_image_cube.py",
        "check_image_beam.py",
        "check_image_beam.py",
        "filter_skymodel.py",
        "calculate_image_diagnostics.py",
        "make_catalog_from_image_cube.py",
        "normalize_flux_scale.py",
    ]


def test_image_sector_task_wraps_runner(tmp_path, fake_image_shell_operation_cls):
    payload = image_payload_from_inputs(_image_input_parms(), tmp_path)

    task_fn = getattr(image_sector_task, "fn", image_sector_task)
    output = task_fn(
        payload["sectors"][0],
        str(tmp_path),
        execution_config=ExecutionConfig(task_runner="sync"),
        shell_operation_cls=fake_image_shell_operation_cls,
    )

    assert output["sector_I_images"] == [
        file_record(tmp_path / "sector_1-MFS-I-image.fits"),
        file_record(tmp_path / "sector_1-MFS-I-image-pb.fits"),
    ]


def test_image_prefect_flow_entrypoint_runs_with_mocked_shell(
    tmp_path, monkeypatch, fake_image_shell_operation_cls
):
    monkeypatch.setattr(
        "rapthor.execution.shell._load_shell_operation_cls",
        lambda: fake_image_shell_operation_cls,
    )

    with prefect_test_harness(server_startup_timeout=None):
        outputs = image_flow(
            image_payload_from_inputs(_image_input_parms(), tmp_path),
            execution_config=ExecutionConfig(task_runner="sync"),
        )

    assert outputs["sector_I_images"][0][0] == file_record(tmp_path / "sector_1-MFS-I-image.fits")
    assert len(fake_image_shell_operation_cls.instances) == 9


def test_run_image_flow_fails_when_expected_output_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="Prepared imaging MS"):
        run_image_flow(
            image_payload_from_inputs(_image_input_parms(), tmp_path),
            execution_config=ExecutionConfig(task_runner="sync"),
            shell_operation_cls=NoOutputShellOperation,
        )


def test_image_reference_output_fixture_matches_output_contract():
    outputs = json.loads((FIXTURE_DIR / "output_reference.json").read_text())

    for value in outputs["image_no_dde"].values():
        validate_output_record(value, allow_none=True)
    for value in outputs["image_facets"].values():
        validate_output_record(value, allow_none=True)
    for value in outputs["image_screens"].values():
        validate_output_record(value, allow_none=True)
    for value in outputs["image_compressed"].values():
        validate_output_record(value, allow_none=True)
    for value in outputs["image_filtered_model"].values():
        validate_output_record(value, allow_none=True)
    for value in outputs["image_full_stokes"].values():
        validate_output_record(value, allow_none=True)
    for value in outputs["image_cube"].values():
        validate_output_record(value, allow_none=True)
    for value in outputs["image_normalize"].values():
        validate_output_record(value, allow_none=True)


def test_image_initial_finalizer_accepts_prefect_outputs(tmp_path, fake_image_shell_operation_cls):
    field = FieldStub(tmp_path)
    operation = ImageInitial(field)
    outputs = run_image_flow(
        image_payload_from_inputs(_image_input_parms(), operation.pipeline_working_dir),
        execution_config=ExecutionConfig(task_runner="sync"),
        shell_operation_cls=fake_image_shell_operation_cls,
    )

    operation.outputs = outputs
    operation.finalize()

    skymodel_dir = Path(field.parset["dir_working"]) / "skymodels" / "initial_image"
    image_dir = Path(field.parset["dir_working"]) / "images" / "initial_image"
    assert field.full_field_sector.image_skymodel_file_true_sky == str(
        skymodel_dir / "sector_1.true_sky.txt"
    )
    assert (skymodel_dir / "sector_1.true_sky.txt").is_file()
    assert (image_dir / "sector_1-MFS-I-image.fits").is_file()
    assert field.full_field_sector.diagnostics == [{}]
    assert field.lofar_to_true_flux_ratio == 1.0
    assert field.lofar_to_true_flux_std == 0.0
    assert Path(operation.done_file).is_file()


def test_image_finalizer_accepts_prefect_outputs_for_selfcal(
    tmp_path, fake_image_shell_operation_cls
):
    field = FieldStub(tmp_path)
    field.parset["imaging_specific"]["save_filtered_model_image"] = True
    field.save_supplementary_images = True
    field.save_visibilities = True
    operation = Image(field, index=1)
    outputs = run_image_flow(
        image_payload_from_inputs(
            _filtered_model_image_input_parms(),
            operation.pipeline_working_dir,
            compress_images=True,
        ),
        execution_config=ExecutionConfig(task_runner="sync"),
        shell_operation_cls=fake_image_shell_operation_cls,
    )

    operation.outputs = outputs
    operation.finalize()

    sector = field.imaging_sectors[0]
    pipeline_dir = Path(operation.pipeline_working_dir)
    skymodel_dir = Path(field.parset["dir_working"]) / "skymodels" / "image_1"
    visibility_dir = Path(field.parset["dir_working"]) / "visibilities" / "image_1" / sector.name
    diagnostics_dir = Path(field.parset["dir_working"]) / "plots" / "image_1"

    assert sector.I_image_file_apparent_sky == str(pipeline_dir / "sector_1-MFS-I-image.fits.fz")
    assert sector.I_image_file_true_sky == str(pipeline_dir / "sector_1-MFS-I-image-pb.fits.fz")
    assert sector.I_model_file_true_sky == str(pipeline_dir / "sector_1-MFS-I-model-pb.fits.fz")
    assert sector.I_residual_file_apparent_sky == str(
        pipeline_dir / "sector_1-MFS-I-residual.fits.fz"
    )
    assert sector.I_dirty_file_apparent_sky == str(pipeline_dir / "sector_1-MFS-I-dirty.fits.fz")
    assert sector.filtering_mask_file == str(
        pipeline_dir / "sector_1-MFS-I-image-pb.fits.mask.fits"
    )
    assert sector.filtered_model_file_apparent_sky == str(
        pipeline_dir / "sector_1-MFS-filtered-model.fits.fz"
    )
    assert sector.image_skymodel_file_true_sky == str(skymodel_dir / "sector_1.true_sky.txt")
    assert sector.image_skymodel_file_apparent_sky == str(
        skymodel_dir / "sector_1.apparent_sky.txt"
    )
    assert (skymodel_dir / "sector_1.true_sky.txt").is_file()
    assert (skymodel_dir / "sector_1.apparent_sky.txt").is_file()
    assert (skymodel_dir / "sector_1.source_catalog.fits").is_file()
    assert (visibility_dir / "sector_1_obs_0_prep.ms").is_dir()
    assert (visibility_dir / "sector_1_obs_1_prep.ms").is_dir()
    assert (diagnostics_dir / "sector_1.image_diagnostics.json").is_file()
    assert sector.diagnostics == [{"cycle_number": 1}]
    assert field.lofar_to_true_flux_ratio == 1.0
    assert field.lofar_to_true_flux_std == 0.0
    assert Path(operation.done_file).is_file()


def test_image_operation_run_uses_prefect_flow(
    tmp_path, monkeypatch, fake_image_shell_operation_cls
):
    monkeypatch.setattr(
        "rapthor.execution.shell._load_shell_operation_cls",
        lambda: fake_image_shell_operation_cls,
    )
    field = FieldStub(tmp_path)
    operation = Image(field, index=1)

    with prefect_test_harness(server_startup_timeout=None):
        operation.run()

    expected_outputs = _expected_image_operation_outputs(operation)
    sector = field.imaging_sectors[0]
    skymodel_dir = Path(field.parset["dir_working"]) / "skymodels" / "image_1"
    diagnostics_dir = Path(field.parset["dir_working"]) / "plots" / "image_1"

    assert operation.outputs == expected_outputs
    assert json.loads(Path(operation.outputs_file).read_text()) == expected_outputs
    assert Path(operation.done_file).is_file()
    assert Path(operation.pipeline_inputs_file).is_file()
    assert sector.I_image_file_apparent_sky == str(
        Path(operation.pipeline_working_dir) / "sector_1-MFS-I-image.fits"
    )
    assert sector.I_image_file_true_sky == str(
        Path(operation.pipeline_working_dir) / "sector_1-MFS-I-image-pb.fits"
    )
    assert sector.image_skymodel_file_true_sky == str(skymodel_dir / "sector_1.true_sky.txt")
    assert (skymodel_dir / "sector_1.true_sky.txt").is_file()
    assert (diagnostics_dir / "sector_1.image_diagnostics.json").is_file()
    assert sector.diagnostics == [{"cycle_number": 1}]
    assert field.lofar_to_true_flux_ratio == 1.0
    assert field.lofar_to_true_flux_std == 0.0
    assert len(fake_image_shell_operation_cls.instances) == 9


def test_bright_peeling_image_operation_run_uses_prefect_flow(
    tmp_path, monkeypatch, fake_image_shell_operation_cls
):
    monkeypatch.setattr(
        "rapthor.execution.shell._load_shell_operation_cls",
        lambda: fake_image_shell_operation_cls,
    )
    field = FieldStub(tmp_path)
    field.peel_bright_sources = True
    operation = Image(field, index=1)

    with prefect_test_harness(server_startup_timeout=None):
        operation.run()

    expected_outputs = _expected_image_operation_outputs(operation)
    commands = [
        shlex.split(instance.kwargs["commands"][0])
        for instance in fake_image_shell_operation_cls.instances
    ]
    restore_commands = [
        command for command in commands if command[0] == "wsclean" and "-restore-list" in command
    ]
    filter_command = next(command for command in commands if command[0] == "filter_skymodel.py")

    assert operation.outputs == expected_outputs
    assert Path(operation.done_file).is_file()
    assert len(restore_commands) == 2
    assert all("/data/bright_sources_pb.txt" in command for command in restore_commands)
    assert "--bright_true_sky_skymodel=/data/bright_sources_pb.txt" in filter_command


def test_image_operation_run_reuses_prefect_outputs_when_done(
    tmp_path, monkeypatch, fake_image_shell_operation_cls
):
    monkeypatch.setattr(
        "rapthor.execution.shell._load_shell_operation_cls",
        lambda: fake_image_shell_operation_cls,
    )
    field = FieldStub(tmp_path)
    operation = Image(field, index=1)
    expected_outputs = _expected_image_operation_outputs(operation)
    _materialize_image_operation_outputs(expected_outputs)
    Path(operation.done_file).touch()
    Path(operation.outputs_file).write_text(json.dumps(expected_outputs))

    operation.run()

    sector = field.imaging_sectors[0]
    skymodel_dir = Path(field.parset["dir_working"]) / "skymodels" / "image_1"
    assert operation.outputs == expected_outputs
    assert Path(operation.done_file).is_file()
    assert sector.I_image_file_apparent_sky == str(
        Path(operation.pipeline_working_dir) / "sector_1-MFS-I-image.fits"
    )
    assert sector.image_skymodel_file_true_sky == str(skymodel_dir / "sector_1.true_sky.txt")
    assert (skymodel_dir / "sector_1.true_sky.txt").is_file()
    assert sector.diagnostics == [{"cycle_number": 1}]
    assert fake_image_shell_operation_cls.instances == []


@pytest.mark.parametrize(
    "shell_operation_cls, expected_message",
    [
        pytest.param(FailingShellOperation, "image failed", id="shell-failure"),
        pytest.param(
            NoOutputShellOperation,
            "Prepared imaging MS",
            id="missing-prepared-ms-output",
        ),
    ],
)
def test_image_operation_run_failure_does_not_mark_done(
    tmp_path, monkeypatch, shell_operation_cls, expected_message
):
    monkeypatch.setattr(
        "rapthor.execution.shell._load_shell_operation_cls",
        lambda: shell_operation_cls,
    )
    field = FieldStub(tmp_path)
    operation = Image(field, index=1)
    sector = field.imaging_sectors[0]

    with (
        prefect_test_harness(server_startup_timeout=None),
        pytest.raises((FileNotFoundError, RuntimeError), match=expected_message),
    ):
        operation.run()

    assert Path(operation.pipeline_inputs_file).is_file()
    assert not Path(operation.done_file).exists()
    assert not Path(operation.outputs_file).exists()
    assert operation.outputs == {}
    assert not hasattr(sector, "I_image_file_apparent_sky")
    assert not hasattr(sector, "image_skymodel_file_true_sky")
    assert sector.diagnostics == []
    assert field.lofar_to_true_flux_ratio is None
    assert field.lofar_to_true_flux_std is None


def test_full_stokes_image_operation_run_uses_prefect_flow(
    tmp_path, monkeypatch, fake_image_shell_operation_cls
):
    monkeypatch.setattr(
        "rapthor.execution.shell._load_shell_operation_cls",
        lambda: fake_image_shell_operation_cls,
    )
    field = FieldStub(tmp_path)
    field.image_pol = "IQUV"
    operation = Image(field, index=1)

    with prefect_test_harness(server_startup_timeout=None):
        operation.run()

    expected_outputs = _expected_full_stokes_image_operation_outputs(operation)
    sector = field.imaging_sectors[0]

    assert operation.outputs == expected_outputs
    assert json.loads(Path(operation.outputs_file).read_text()) == expected_outputs
    assert Path(operation.done_file).is_file()
    assert Path(operation.pipeline_inputs_file).is_file()
    assert sector.I_image_file_apparent_sky == str(
        Path(operation.pipeline_working_dir) / "sector_1-MFS-image.fits"
    )
    assert sector.I_image_file_true_sky == str(
        Path(operation.pipeline_working_dir) / "sector_1-MFS-image-pb.fits"
    )
    assert sector.Q_image_file_apparent_sky == str(
        Path(operation.pipeline_working_dir) / "sector_1-MFS-Q-image.fits"
    )
    assert sector.U_model_file_true_sky == str(
        Path(operation.pipeline_working_dir) / "sector_1-MFS-U-model-pb.fits"
    )
    assert sector.V_dirty_file_apparent_sky == str(
        Path(operation.pipeline_working_dir) / "sector_1-MFS-V-dirty.fits"
    )
    assert not hasattr(sector, "image_skymodel_file_true_sky")
    assert not hasattr(sector, "image_skymodel_file_apparent_sky")
    assert sector.diagnostics == [{"cycle_number": 1}]
    assert field.lofar_to_true_flux_ratio == 1.0
    assert field.lofar_to_true_flux_std == 0.0
    assert len(fake_image_shell_operation_cls.instances) == 9


def test_compressed_image_operation_run_uses_prefect_flow(
    tmp_path, monkeypatch, fake_image_shell_operation_cls
):
    monkeypatch.setattr(
        "rapthor.execution.shell._load_shell_operation_cls",
        lambda: fake_image_shell_operation_cls,
    )
    field = FieldStub(tmp_path)
    field.compress_images = True
    field.parset["imaging_specific"]["save_filtered_model_image"] = True
    field.save_supplementary_images = True
    field.save_visibilities = True
    operation = Image(field, index=1)

    with prefect_test_harness(server_startup_timeout=None):
        operation.run()

    expected_outputs = _expected_compressed_image_operation_outputs(operation)
    sector = field.imaging_sectors[0]
    pipeline_dir = Path(operation.pipeline_working_dir)
    skymodel_dir = Path(field.parset["dir_working"]) / "skymodels" / "image_1"
    visibility_dir = Path(field.parset["dir_working"]) / "visibilities" / "image_1" / sector.name
    diagnostics_dir = Path(field.parset["dir_working"]) / "plots" / "image_1"
    command_names = [
        shlex.split(instance.kwargs["commands"][0])[0]
        for instance in fake_image_shell_operation_cls.instances
    ]

    assert operation.outputs == expected_outputs
    assert json.loads(Path(operation.outputs_file).read_text()) == expected_outputs
    assert Path(operation.done_file).is_file()
    assert Path(operation.pipeline_inputs_file).is_file()
    assert sector.I_image_file_apparent_sky == str(pipeline_dir / "sector_1-MFS-I-image.fits.fz")
    assert sector.I_image_file_true_sky == str(pipeline_dir / "sector_1-MFS-I-image-pb.fits.fz")
    assert sector.I_model_file_true_sky == str(pipeline_dir / "sector_1-MFS-I-model-pb.fits.fz")
    assert sector.I_residual_file_apparent_sky == str(
        pipeline_dir / "sector_1-MFS-I-residual.fits.fz"
    )
    assert sector.I_dirty_file_apparent_sky == str(pipeline_dir / "sector_1-MFS-I-dirty.fits.fz")
    assert sector.filtering_mask_file == str(
        pipeline_dir / "sector_1-MFS-I-image-pb.fits.mask.fits"
    )
    assert sector.filtered_model_file_apparent_sky == str(
        pipeline_dir / "sector_1-MFS-filtered-model.fits.fz"
    )
    assert sector.image_skymodel_file_true_sky == str(skymodel_dir / "sector_1.true_sky.txt")
    assert sector.image_skymodel_file_apparent_sky == str(
        skymodel_dir / "sector_1.apparent_sky.txt"
    )
    assert (skymodel_dir / "sector_1.true_sky.txt").is_file()
    assert (skymodel_dir / "sector_1.apparent_sky.txt").is_file()
    assert (skymodel_dir / "sector_1.source_catalog.fits").is_file()
    assert (visibility_dir / "sector_1_obs_0_prep.ms").is_dir()
    assert (visibility_dir / "sector_1_obs_1_prep.ms").is_dir()
    assert (diagnostics_dir / "sector_1.image_diagnostics.json").is_file()
    assert sector.diagnostics == [{"cycle_number": 1}]
    assert field.lofar_to_true_flux_ratio == 1.0
    assert field.lofar_to_true_flux_std == 0.0
    assert "restore_skymodel.py" in command_names
    assert "fpack" in command_names


def test_clean_disabled_image_operation_run_uses_prefect_flow(
    tmp_path, monkeypatch, fake_image_shell_operation_cls
):
    monkeypatch.setattr(
        "rapthor.execution.shell._load_shell_operation_cls",
        lambda: fake_image_shell_operation_cls,
    )
    field = FieldStub(tmp_path)
    field.disable_clean = True
    operation = Image(field, index=1)

    with prefect_test_harness(server_startup_timeout=None):
        operation.run()

    expected_outputs = _expected_image_operation_outputs(operation)
    sector = field.imaging_sectors[0]
    wsclean_command = next(
        shlex.split(instance.kwargs["commands"][0])
        for instance in fake_image_shell_operation_cls.instances
        if shlex.split(instance.kwargs["commands"][0])[0] == "wsclean"
    )

    assert operation.outputs == expected_outputs
    assert json.loads(Path(operation.outputs_file).read_text()) == expected_outputs
    assert Path(operation.done_file).is_file()
    assert Path(operation.pipeline_inputs_file).is_file()
    assert wsclean_command[wsclean_command.index("-niter") + 1] == "0"
    assert sector.I_image_file_apparent_sky == str(
        Path(operation.pipeline_working_dir) / "sector_1-MFS-I-image.fits"
    )
    assert sector.I_image_file_true_sky == str(
        Path(operation.pipeline_working_dir) / "sector_1-MFS-I-image-pb.fits"
    )
    assert sector.diagnostics == [{"cycle_number": 1}]
    assert field.lofar_to_true_flux_ratio == 1.0
    assert field.lofar_to_true_flux_std == 0.0


def test_facet_image_operation_run_uses_prefect_flow(
    tmp_path, monkeypatch, fake_image_shell_operation_cls
):
    monkeypatch.setattr(
        "rapthor.execution.shell._load_shell_operation_cls",
        lambda: fake_image_shell_operation_cls,
    )
    field = FieldStub(tmp_path)
    field.dde_method = "full"
    field.dd_h5parm_filename = "/data/facet-solutions.h5"
    field.calibration_strategy = {"dd": ["fast_phase"]}
    field.parset["imaging_specific"]["shared_facet_rw"] = True
    operation = Image(field, index=1)

    with prefect_test_harness(server_startup_timeout=None):
        operation.run()

    expected_outputs = _expected_facet_image_operation_outputs(operation)
    region_dir = Path(field.parset["dir_working"]) / "regions" / "image_1"
    make_region_command = next(
        shlex.split(instance.kwargs["commands"][0])
        for instance in fake_image_shell_operation_cls.instances
        if shlex.split(instance.kwargs["commands"][0])[0] == "make_region_file.py"
    )
    wsclean_command = next(
        shlex.split(instance.kwargs["commands"][0])
        for instance in fake_image_shell_operation_cls.instances
        if shlex.split(instance.kwargs["commands"][0])[0] == "wsclean"
    )

    assert operation.outputs == expected_outputs
    assert Path(operation.done_file).is_file()
    assert make_region_command[1] == "/data/calibration.skymodel"
    assert "-apply-facet-beam" in wsclean_command
    assert "-apply-facet-solutions" in wsclean_command
    assert "/data/facet-solutions.h5" in wsclean_command
    assert "-facet-regions" in wsclean_command
    assert "-shared-facet-reads" in wsclean_command
    assert "-shared-facet-writes" in wsclean_command
    assert (region_dir / "sector_1_facets_ds9.reg").is_file()


def test_screen_image_operation_run_uses_prefect_flow(
    tmp_path, monkeypatch, fake_image_shell_operation_cls
):
    monkeypatch.setattr(
        "rapthor.execution.shell._load_shell_operation_cls",
        lambda: fake_image_shell_operation_cls,
    )
    field = FieldStub(tmp_path)
    field.apply_screens = True
    field.dd_h5parm_filename = "/data/screen-solutions.h5"
    field.calibration_strategy = {"dd": ["fast_phase"]}
    operation = Image(field, index=1)

    with prefect_test_harness(server_startup_timeout=None):
        operation.run()

    expected_outputs = _expected_image_operation_outputs(operation)
    wsclean_command = next(
        shlex.split(instance.kwargs["commands"][0])
        for instance in fake_image_shell_operation_cls.instances
        if shlex.split(instance.kwargs["commands"][0])[0] == "wsclean"
    )
    prepare_commands = [
        shlex.split(instance.kwargs["commands"][0])
        for instance in fake_image_shell_operation_cls.instances
        if shlex.split(instance.kwargs["commands"][0])[0] == "DP3"
    ]

    assert operation.outputs == expected_outputs
    assert Path(operation.done_file).is_file()
    assert "-aterm-config" in wsclean_command
    assert wsclean_command[wsclean_command.index("-aterm-config") + 1] == ATERM_CONFIG_FILENAME
    assert (Path(operation.pipeline_working_dir) / ATERM_CONFIG_FILENAME).is_file()
    assert any("steps=[applybeam,shift,applycal,avg]" in command for command in prepare_commands)


def test_image_cube_operation_run_uses_prefect_flow(
    tmp_path, monkeypatch, fake_image_shell_operation_cls
):
    monkeypatch.setattr(
        "rapthor.execution.shell._load_shell_operation_cls",
        lambda: fake_image_shell_operation_cls,
    )
    field = FieldStub(tmp_path)
    field.make_image_cube = True
    operation = Image(field, index=1)

    with prefect_test_harness(server_startup_timeout=None):
        operation.run()

    expected_outputs = _expected_image_cube_operation_outputs(operation)
    image_dir = Path(field.parset["dir_working"]) / "images" / "image_1"
    make_cube_command = next(
        shlex.split(instance.kwargs["commands"][0])
        for instance in fake_image_shell_operation_cls.instances
        if shlex.split(instance.kwargs["commands"][0])[0] == "make_image_cube.py"
    )

    assert operation.outputs == expected_outputs
    assert Path(operation.done_file).is_file()
    assert make_cube_command[2] == "sector_1_I_freq_cube.fits"
    assert (image_dir / "sector_1_I_freq_cube.fits").is_file()
    assert (image_dir / "sector_1_I_freq_cube.fits_beams.txt").is_file()
    assert (image_dir / "sector_1_I_freq_cube.fits_frequencies.txt").is_file()


def test_normalize_image_operation_run_uses_prefect_flow(
    tmp_path, monkeypatch, fake_image_shell_operation_cls
):
    monkeypatch.setattr(
        "rapthor.execution.shell._load_shell_operation_cls",
        lambda: fake_image_shell_operation_cls,
    )
    field = FieldStub(tmp_path)
    operation = ImageNormalize(field, index=1)

    with prefect_test_harness(server_startup_timeout=None):
        operation.run()

    expected_outputs = _expected_normalize_image_operation_outputs(operation)
    solutions_dir = Path(field.parset["dir_working"]) / "solutions" / "normalize_1"
    image_dir = Path(field.parset["dir_working"]) / "images" / "normalize_1"
    command_names = [
        shlex.split(instance.kwargs["commands"][0])[0]
        for instance in fake_image_shell_operation_cls.instances
    ]

    assert operation.outputs == expected_outputs
    assert json.loads(Path(operation.outputs_file).read_text()) == expected_outputs
    assert Path(operation.done_file).is_file()
    assert "make_image_cube.py" in command_names
    assert "make_catalog_from_image_cube.py" in command_names
    assert "normalize_flux_scale.py" in command_names
    assert field.normalize_h5parm == str(solutions_dir / "sector_1_normalize.h5parm")
    assert (solutions_dir / "sector_1_normalize.h5parm").is_file()
    assert (image_dir / "sector_1_I_freq_cube.fits").is_file()
    assert field.normalize_flux_scale is False
    assert field.apply_normalizations is True


def test_mpi_image_operation_run_uses_prefect_flow(
    tmp_path, monkeypatch, fake_image_shell_operation_cls
):
    monkeypatch.setattr(
        "rapthor.execution.shell._load_shell_operation_cls",
        lambda: fake_image_shell_operation_cls,
    )
    field = FieldStub(tmp_path)
    field.use_mpi = True
    field.parset["cluster_specific"]["max_nodes"] = 3
    field.parset["cluster_specific"]["cpus_per_task"] = 2
    operation = Image(field, index=1)

    with prefect_test_harness(server_startup_timeout=None):
        operation.run()

    expected_outputs = _expected_image_operation_outputs(operation)
    mpi_command = next(
        shlex.split(instance.kwargs["commands"][0])
        for instance in fake_image_shell_operation_cls.instances
        if shlex.split(instance.kwargs["commands"][0])[0] == "mpirun"
    )

    assert operation.outputs == expected_outputs
    assert Path(operation.done_file).is_file()
    assert "wsclean-mp" in mpi_command
    assert mpi_command[mpi_command.index("-np") + 1] == "2"


def test_previous_mask_image_operation_run_uses_prefect_flow(
    tmp_path, monkeypatch, fake_image_shell_operation_cls
):
    monkeypatch.setattr(
        "rapthor.execution.shell._load_shell_operation_cls",
        lambda: fake_image_shell_operation_cls,
    )
    previous_mask = tmp_path / "previous-mask.fits"
    previous_mask.write_text("previous mask")
    field = FieldStub(tmp_path)
    field.parset["imaging_specific"]["use_clean_mask"] = True
    field.imaging_sectors[0].I_mask_file = str(previous_mask)
    operation = Image(field, index=1)

    with prefect_test_harness(server_startup_timeout=None):
        operation.run()

    expected_outputs = _expected_image_operation_outputs(operation)
    sector = field.imaging_sectors[0]
    blank_command = next(
        shlex.split(instance.kwargs["commands"][0])
        for instance in fake_image_shell_operation_cls.instances
        if shlex.split(instance.kwargs["commands"][0])[0] == "blank_image.py"
    )
    mask_dir = Path(field.parset["dir_working"]) / "images" / "image_1" / "sector_1"

    assert operation.outputs == expected_outputs
    assert Path(operation.done_file).is_file()
    assert blank_command[2] == str(previous_mask)
    assert sector.I_mask_file == str(mask_dir / "sector_1-MFS-I-image-pb.fits.mask.fits")
    assert (mask_dir / "sector_1-MFS-I-image-pb.fits.mask.fits").is_file()


def test_image_finalizer_accepts_prefect_outputs_for_full_stokes(
    tmp_path, fake_image_shell_operation_cls
):
    field = FieldStub(tmp_path)
    field.image_pol = "IQUV"
    operation = Image(field, index=1)
    outputs = run_image_flow(
        image_payload_from_inputs(
            _full_stokes_image_input_parms(),
            operation.pipeline_working_dir,
        ),
        execution_config=ExecutionConfig(task_runner="sync"),
        shell_operation_cls=fake_image_shell_operation_cls,
    )

    operation.outputs = outputs
    operation.finalize()

    sector = field.imaging_sectors[0]
    pipeline_dir = Path(operation.pipeline_working_dir)
    skymodel_dir = Path(field.parset["dir_working"]) / "skymodels" / "image_1"
    assert sector.I_image_file_apparent_sky == str(pipeline_dir / "sector_1-MFS-image.fits")
    assert sector.I_image_file_true_sky == str(pipeline_dir / "sector_1-MFS-image-pb.fits")
    assert sector.Q_image_file_apparent_sky == str(pipeline_dir / "sector_1-MFS-Q-image.fits")
    assert sector.Q_image_file_true_sky == str(pipeline_dir / "sector_1-MFS-Q-image-pb.fits")
    assert sector.U_model_file_true_sky == str(pipeline_dir / "sector_1-MFS-U-model-pb.fits")
    assert sector.V_dirty_file_apparent_sky == str(pipeline_dir / "sector_1-MFS-V-dirty.fits")
    assert not hasattr(sector, "image_skymodel_file_true_sky")
    assert not hasattr(sector, "image_skymodel_file_apparent_sky")
    assert (skymodel_dir / "sector_1.source_catalog.fits").is_file()
    assert Path(operation.done_file).is_file()


def test_image_normalize_finalizer_accepts_prefect_outputs(
    tmp_path, fake_image_shell_operation_cls
):
    field = FieldStub(tmp_path)
    operation = ImageNormalize(field, index=1)
    operation.outputs = run_image_flow(
        image_payload_from_inputs(
            _normalize_image_input_parms(),
            operation.pipeline_working_dir,
            make_image_cube=True,
            normalize_flux_scale=True,
        ),
        execution_config=ExecutionConfig(task_runner="sync"),
        shell_operation_cls=fake_image_shell_operation_cls,
    )

    operation.finalize()

    solutions_dir = Path(field.parset["dir_working"]) / "solutions" / "normalize_1"
    image_dir = Path(field.parset["dir_working"]) / "images" / "normalize_1"
    assert field.normalize_h5parm == str(solutions_dir / "sector_1_normalize.h5parm")
    assert (solutions_dir / "sector_1_normalize.h5parm").is_file()
    assert (image_dir / "sector_1_I_freq_cube.fits").is_file()
    assert (image_dir / "sector_1_I_freq_cube.fits_beams.txt").is_file()
    assert (image_dir / "sector_1_I_freq_cube.fits_frequencies.txt").is_file()
    assert field.normalize_flux_scale is False
    assert field.apply_normalizations is True
    assert Path(operation.done_file).is_file()

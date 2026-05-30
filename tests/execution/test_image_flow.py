import json
import shlex
from pathlib import Path

import pytest
from prefect.testing.utilities import prefect_test_harness

from rapthor.execution.config import ExecutionConfig
from rapthor.execution.flows.image import (
    ATERM_CONFIG_FILENAME,
    build_aterm_config_content,
    build_calculate_image_diagnostics_command,
    build_check_image_beam_command,
    build_compress_sector_images_command,
    build_filter_skymodel_command,
    build_make_region_file_command,
    build_make_skymodel_image_command,
    image_flow,
    image_payload_from_inputs,
    image_sector_task,
    normalized_blank_image_command,
    normalized_compress_sector_images_command,
    normalized_concat_time_command,
    normalized_make_region_file_command,
    normalized_make_skymodel_image_command,
    normalized_prepare_imaging_data_command,
    normalized_wsclean_facets_command,
    normalized_wsclean_no_dde_command,
    normalized_wsclean_screens_command,
    run_image_flow,
)
from rapthor.execution.outputs import directory_record, file_record, validate_output_record
from rapthor.operations.image import ImageInitial

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
            cwd = Path(self.kwargs["cwd"])
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
            elif tokens[0] == "wsclean":
                image_name = tokens[tokens.index("-name") + 1]
                for suffix in [
                    "-MFS-I-image.fits",
                    "-MFS-I-image-pb.fits",
                    "-MFS-I-residual.fits",
                    "-MFS-I-model-pb.fits",
                    "-MFS-I-dirty.fits",
                    "-sources.txt",
                    "-sources-pb.txt",
                ]:
                    (cwd / f"{image_name}{suffix}").write_text("image")
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


class SectorStub:
    def __init__(self):
        self.name = "sector_1"
        self.diagnostics = []


class FieldStub:
    def __init__(self, tmp_path):
        self.parset = _operation_parset(tmp_path)
        self.make_image_cube = False
        self.full_field_sector = SectorStub()
        self.save_supplementary_images = False
        self.lofar_to_true_flux_ratio = None
        self.lofar_to_true_flux_std = None


def _operation_parset(tmp_path):
    return {
        "dir_working": str(tmp_path / "working"),
        "cluster_specific": {
            "cwl_runner": "toil",
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
        },
        "imaging_specific": {
            "use_clean_mask": False,
            "save_filtered_model_image": False,
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


def test_image_command_builders_match_reference_fixtures():
    commands = json.loads((FIXTURE_DIR / "cwl_reference_commands.json").read_text())

    assert (
        normalized_prepare_imaging_data_command(
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
        == commands["image"]["prepare_imaging_data"]
    )
    assert (
        normalized_concat_time_command(
            input_filenames=["sector_1_obs_0_prep.ms", "sector_1_obs_1_prep.ms"],
            output_filename="sector_1_concat.ms",
            data_colname="DATA",
        )
        == commands["image"]["concat_time"]
    )
    assert (
        normalized_blank_image_command(
            mask_filename="sector_1_mask.fits",
            wsclean_imsize=[1024, 1024],
            vertices_file="sector_1.vertices",
            ra=123.0,
            dec=45.0,
            cellsize_deg=0.001,
        )
        == commands["image"]["blank_image"]
    )
    assert (
        normalized_compress_sector_images_command(
            images=[
                "sector_1-MFS-I-image.fits",
                "sector_1-MFS-I-image-pb.fits",
                "sector_1-MFS-I-residual.fits",
                "sector_1-MFS-I-model-pb.fits",
                "sector_1-MFS-I-dirty.fits",
            ]
        )
        == commands["image"]["compress_sector_images"]
    )
    assert (
        normalized_make_skymodel_image_command(
            source_catalog="sector_1.apparent_sky.txt",
            reference_image="sector_1-MFS-I-image-pb.fits",
            output_image_name="sector_1-MFS-filtered-model.fits.fz",
        )
        == commands["image"]["make_skymodel_image"]
    )
    assert (
        normalized_wsclean_no_dde_command(
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
        == commands["image"]["wsclean_no_dde"]
    )
    assert (
        normalized_make_region_file_command(
            skymodel="calibration.skymodel",
            ra_mid=123.0,
            dec_mid=45.0,
            width_ra=2.0,
            width_dec=2.5,
            outfile="sector_1_facets_ds9.reg",
        )
        == commands["image"]["make_region_file"]
    )
    assert (
        normalized_wsclean_facets_command(
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
        == commands["image"]["wsclean_facets"]
    )
    assert (
        normalized_wsclean_screens_command(
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
        == commands["image"]["wsclean_screens"]
    )


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


def test_image_payload_from_inputs_rejects_unsupported_modes(tmp_path):
    with pytest.raises(ValueError, match="cannot both"):
        image_payload_from_inputs(
            _facet_image_input_parms(), tmp_path, apply_screens=True, use_facets=True
        )

    input_parms = _image_input_parms()
    input_parms["pol"] = "IQUV"
    with pytest.raises(NotImplementedError, match="Stokes-I"):
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
    tmp_path, fake_image_shell_operation_cls
):
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
    outputs = json.loads((FIXTURE_DIR / "cwl_reference_outputs.json").read_text())

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

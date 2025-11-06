"""
This module contains tests for the different derived `CWLRunner` classes in
`rapthor.lib.cwlrunner`.

Currently, the tests only cover if the correct scratch directories are passed
on the command-line to the different CWL runners.
"""

import os

import pytest
from rapthor.lib.cwlrunner import create_cwl_runner
from rapthor.lib.parset import Parset
from rapthor.operations.image import Image


class Sector:
    """
    Mock class that provides the minimal number of attributes needed to
    mimick the real `Sector` class in the module `rapthor.lib.sector`.
    """

    def __init__(self, name, ra=0.0, dec=0.0, width_ra=0.0, width_dec=0.0, field=None):
        self.name = name
        self.ra = ra
        self.dec = dec
        self.width_ra = width_ra
        self.width_dec = width_dec
        self.field = field

        self.auto_mask = None
        self.auto_mask_nmiter = 1
        self.central_patch = None
        self.I_mask_file = None
        self.imsize = None
        self.multiscale = None
        self.region_file = None
        self.threshisl = None
        self.threshpix = None
        self.vertices_file = "vertices.npy"
        self.wsclean_deconvolution_channels = None
        self.wsclean_nchannels = None
        self.wsclean_niter = None
        self.wsclean_nmiter = None
        self.wsclean_spectral_poly_order = None

        imaging_parameters = self.field.parset["imaging_specific"]
        self.cellsize_arcsec = imaging_parameters["cellsize_arcsec"]
        self.cellsize_deg = self.cellsize_arcsec / 3600.0
        self.robust = imaging_parameters["robust"]
        self.taper_arcsec = imaging_parameters["taper_arcsec"]
        self.local_rms_strength = imaging_parameters["local_rms_strength"]
        self.local_rms_window = imaging_parameters["local_rms_window"]
        self.local_rms_method = imaging_parameters["local_rms_method"]
        self.min_uv_lambda = imaging_parameters["min_uv_lambda"]
        self.max_uv_lambda = imaging_parameters["max_uv_lambda"]
        self.mgain = imaging_parameters["mgain"]
        self.idg_mode = imaging_parameters["idg_mode"]
        self.mem_limit_gb = imaging_parameters["mem_gb"]
        self.dd_psf_grid = imaging_parameters["dd_psf_grid"]

    def set_imaging_parameters(
        self, do_multiscale=False, recalculate_imsize=False, imaging_parameters=None,
        preapply_dde_solutions=False
    ):
        pass

    def get_obs_parameters(self, parameter):
        obs_parameters = {
            "ms_filename": "filename.ms",
            "ms_prep_filename": None,
            "image_freqstep": None,
            "image_timestep": None,
            "image_bda_maxinterval": None,
        }
        return obs_parameters[parameter]


class Observation:
    """
    Mock class that provides the minimal number of attributes needed to
    mimick the real `Observation` class in the module `rapthor.lib.observation`.
    """

    def __init__(self):
        self.ms_imaging_filename = "filename.ms"
        self.starttime = 5086015124.0
        self.numsamples = 1
        self.channels_are_regular = True
        self.timepersample = 1


class Field:
    """
    Mock class that provides the minimal number of attributes needed to
    mimick the real `Field` class in the module `rapthor.lib.field`.
    """

    def __init__(self, parset):
        self.parset = parset

        self.data_colname = 'DATA'
        self.apply_amplitudes = False
        self.apply_fulljones = False
        self.apply_screens = False
        self.calibration_skymodel_file = "calibration_skymodel.txt"
        self.dec = 0
        self.do_predict = False
        self.use_image_based_predict = False
        self.fulljones_h5parm_filename = parset["input_fulljones_h5parm"]
        self.h5parm_filename = parset["input_h5parm"]
        self.image_pol = "I"
        self.imaging_sectors = [Sector("sector_1", field=self)]
        self.observations = [Observation()]
        self.peel_bright_sources = False
        self.ra = 0
        self.correct_smearing_in_calibration = True

        imaging_parameters = self.parset["imaging_specific"]
        self.compress_images = True
        self.dde_method = imaging_parameters["dde_method"]
        self.use_mpi = imaging_parameters["use_mpi"]
        self.do_multiscale_clean = imaging_parameters["do_multiscale_clean"]
        self.pol_combine_method = imaging_parameters["pol_combine_method"]
        self.apply_normalizations = False
        self.auto_mask_nmiter = 1
        self.skip_final_major_iteration = True
        self.image_bda_timebase = 0
        self.slow_timestep_sec = 1
        self.apply_time_frequency_smearing = True
        self.correct_smearing_in_imaging = True
        self.make_image_cube = False

    def get_calibration_radius(self):
        return 5.0


@pytest.fixture(params=("single_machine", "slurm", "slurm_static"))
def batch_system(request):
    return request.param


@pytest.fixture(params=(None, "dir/local"))
def dir_local(tmp_path, request):
    return str(tmp_path / request.param) if request.param else None


@pytest.fixture(params=(None, "global/scratch"))
def global_scratch_dir(tmp_path, request):
    return str(tmp_path / request.param) if request.param else None


@pytest.fixture(params=(None, "local/scratch"))
def local_scratch_dir(tmp_path, request):
    return str(tmp_path / request.param) if request.param else None


@pytest.fixture(params=(False, True))
def use_mpi(request):
    return request.param


@pytest.fixture
def parset(
    tmp_path,
    batch_system,
    cwl_runner,
    dir_local,
    global_scratch_dir,
    local_scratch_dir,
    use_mpi,
):
    """
    Fixture that generates a default parset, with some settings adjusted
    """
    parset = Parset().as_parset_dict()
    parset["dir_working"] = str(tmp_path)
    parset["input_h5parm"] = str(tmp_path / "h5parm.h5")
    parset["cluster_specific"]["batch_system"] = batch_system
    parset["cluster_specific"]["cwl_runner"] = cwl_runner
    parset["cluster_specific"]["dir_local"] = dir_local
    parset["cluster_specific"]["global_scratch_dir"] = global_scratch_dir
    parset["cluster_specific"]["local_scratch_dir"] = local_scratch_dir
    parset["cluster_specific"]["max_nodes"] = 1
    parset["cluster_specific"]["cpus_per_task"] = 4
    parset["imaging_specific"]["use_mpi"] = use_mpi
    return parset


@pytest.fixture
def runner(parset):
    """
    Fixture that generates a `CWLRunner` instance of the correct type.
    For now, we only need a runner for one type of `Operation`: `Image`.
    """
    field = Field(parset)
    operation = Image(field, index=1)
    operation.setup()
    runner = create_cwl_runner(parset["cluster_specific"]["cwl_runner"], operation)
    runner.setup()
    yield runner
    runner.teardown()


@pytest.mark.parametrize("cwl_runner", ("cwltool", "toil"))
class TestCWLRunner:
    def test_mpi_config_file(self, runner):
        """
        Test if the MPI configuration file is present and has the correct content.
        """
        if runner.operation.use_mpi:
            mpi_config_file = runner.args[runner.args.index("--mpi-config-file") + 1]
            assert os.path.isfile(mpi_config_file)
            with open(mpi_config_file, "r", encoding="utf-8") as f:
                content = f.read()
            if runner.operation.batch_system == "slurm":
                assert "runner: 'mpi_runner.sh'" in content
                assert "nproc_flag: '-N'" in content
                assert "extra_flags: ['--cpus-per-task=4', 'mpirun', '-pernode', '--bind-to', 'none', '-x', 'OPENBLAS_NUM_THREADS']" in content
            elif runner.operation.batch_system in ("single_machine", "slurm_static"):
                assert "runner: 'mpirun'" in content
                assert "nproc_flag: '-np'" in content
                assert "extra_flags: ['-pernode', '--bind-to', 'none', '-x', 'OPENBLAS_NUM_THREADS']" in content
        else:
            assert "--mpi-config-file" not in runner.args, "MPI config file should not be present when not using MPI"


@pytest.mark.parametrize("cwl_runner", ("cwltool",))
class TestCWLToolRunner:
    def test_tmpdir_prefix(
        tmp_path,
        dir_local,
        global_scratch_dir,
        local_scratch_dir,
        use_mpi,
        parset,
        runner,
    ):
        """
        Test if command-line option `--tmpdir-prefix` is present when expected,
        and if the value is correct.
        """
        try:
            prefix = runner.args[runner.args.index("--tmpdir-prefix") + 1]
        except ValueError:
            pass
        else:
            scratch_dir = os.path.dirname(prefix)
            if use_mpi:
                if global_scratch_dir:
                    assert scratch_dir == global_scratch_dir
                else:
                    assert scratch_dir.startswith(parset["dir_working"])
            else:
                assert local_scratch_dir or dir_local
                if local_scratch_dir:
                    assert scratch_dir == local_scratch_dir
                elif dir_local:
                    assert scratch_dir == dir_local

    def test_tmp_outdir_prefix(
        tmp_path,
        global_scratch_dir,
        runner,
    ):
        """
        Test if command-line option `--tmp-outdir-prefix` is present when expected,
        and if the value is correct.
        """
        try:
            prefix = runner.args[runner.args.index("--tmp-outdir-prefix") + 1]
        except ValueError:
            pass
        else:
            if global_scratch_dir:
                assert os.path.dirname(prefix) == global_scratch_dir
            else:
                assert False


@pytest.mark.parametrize("cwl_runner", ("toil",))
class TestToilRunner:
    def test_tmpdir_prefix(
        tmp_path,
        dir_local,
        batch_system,
        global_scratch_dir,
        local_scratch_dir,
        use_mpi,
        parset,
        runner,
    ):
        """
        Test if command-line option `--tmpdir-prefix` is present when expected,
        and if the value is correct.
        """
        try:
            prefix = runner.args[runner.args.index("--tmpdir-prefix") + 1]
        except ValueError:
            pass
        else:
            scratch_dir = os.path.dirname(prefix)
            if use_mpi:
                if global_scratch_dir:
                    assert scratch_dir == global_scratch_dir
                else:
                    assert scratch_dir.startswith(parset["dir_working"])
            else:
                assert local_scratch_dir or dir_local
                if local_scratch_dir:
                    assert scratch_dir == local_scratch_dir
                elif dir_local:
                    assert scratch_dir == dir_local
            if batch_system == "single_machine":
                assert os.path.isdir(scratch_dir)

    def test_tmp_outdir_prefix(
        tmp_path,
        batch_system,
        global_scratch_dir,
        parset,
        runner,
    ):
        """
        Test if command-line option `--tmp-outdir-prefix` is present when expected,
        and if the value is correct.
        """
        try:
            prefix = runner.args[runner.args.index("--tmp-outdir-prefix") + 1]
        except ValueError:
            pass
        else:
            scratch_dir = os.path.dirname(prefix)
            if global_scratch_dir:
                assert scratch_dir == global_scratch_dir
            elif batch_system == "slurm":
                assert scratch_dir.startswith(parset["dir_working"])
            else:
                assert False
            assert os.path.isdir(scratch_dir)

    def test_workdir(
        tmp_path,
        batch_system,
        global_scratch_dir,
        parset,
        runner,
    ):
        """
        Test if command-line option `--workDir` is present when using Slurm,
        and if the value is correct.
        """
        try:
            workdir = runner.args[runner.args.index("--workDir") + 1]
        except ValueError:
            assert batch_system != "slurm"
        else:
            if global_scratch_dir:
                assert os.path.dirname(workdir) == global_scratch_dir
            else:
                assert workdir.startswith(parset["dir_working"])
            assert os.path.isdir(workdir)

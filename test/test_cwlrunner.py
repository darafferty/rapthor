"""
This module contains tests for the different derived `CWLRunner` classes in
`rapthor.lib.cwlrunner`.

Currently, the tests only cover if the correct scratch directories are passed
on the command-line to the different CWL runners.
"""

import os
import pytest
from rapthor.lib.parset import Parset
from rapthor.operations.image import Image
from rapthor.lib.cwlrunner import create_cwl_runner


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
        self.central_patch = None
        self.I_mask_file = None
        self.imsize = None
        self.multiscale = None
        self.region_file = None
        self.threshisl = None
        self.threshpix = None
        self.vertices_file = "vertices.pkl"
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
        self.min_uv_lambda = imaging_parameters["min_uv_lambda"]
        self.max_uv_lambda = imaging_parameters["max_uv_lambda"]
        self.mgain = imaging_parameters["mgain"]
        self.idg_mode = imaging_parameters["idg_mode"]
        self.mem_limit_gb = imaging_parameters["mem_gb"]
        self.dd_psf_grid = imaging_parameters["dd_psf_grid"]

    def set_imaging_parameters(
        self, do_multiscale=False, recalculate_imsize=False, imaging_parameters=None
    ):
        pass

    def get_obs_parameters(self, parameter):
        obs_parameters = {
            "ms_filename": "filename.ms",
            "ms_prep_filename": None,
            "image_freqstep": None,
            "image_timestep": None,
        }
        return obs_parameters[parameter]


class Field:
    """
    Mock class that provides the minimal number of attributes needed to
    mimick the real `Field` class in the module `rapthor.lib.field`.
    """

    def __init__(self, parset):
        self.parset = parset

        self.apply_amplitudes = False
        self.apply_fulljones = False
        self.apply_screens = False
        self.calibration_skymodel_file = "calibration_skymodel.txt"
        self.dec = 0
        self.do_predict = False
        self.fulljones_h5parm_filename = parset["input_fulljones_h5parm"]
        self.h5parm_filename = parset["input_h5parm"]
        self.image_pol = "I"
        self.imaging_sectors = [Sector("sector_1", field=self)]
        self.observations = []
        self.peel_bright_sources = False
        self.ra = 0

        imaging_parameters = self.parset["imaging_specific"]
        self.dde_method = imaging_parameters["dde_method"]
        self.use_mpi = imaging_parameters["use_mpi"]
        self.do_multiscale_clean = imaging_parameters["do_multiscale_clean"]
        self.pol_combine_method = imaging_parameters["pol_combine_method"]
        self.apply_normalizations = False

    def get_calibration_radius(self):
        return 5.0

@pytest.fixture
def dir_postfix():
    return f"rapthor.{os.getpid()}"

@pytest.fixture(params=("single_machine", "slurm"))
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
    Fixture that generates a default parset, with some settings adjusted.
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
    return runner


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
            if use_mpi:
                if global_scratch_dir:
                    assert os.path.dirname(prefix) == global_scratch_dir
                else:
                    assert prefix.startswith(parset["dir_working"])
            else:
                assert local_scratch_dir or dir_local
                if local_scratch_dir:
                    assert os.path.dirname(prefix) == local_scratch_dir
                elif dir_local:
                    assert os.path.dirname(prefix) == dir_local

    def test_tmp_outdir_prefix(
        tmp_path,
        global_scratch_dir,
        runner,
    ):
        """
        Test if command-line option `--tmp-outdir-prefix` is present when
        expected, and if the value is correct.
        """
        try:
            prefix = runner.args[runner.args.index("--tmp-outdir-prefix") + 1]
        except ValueError:
            pass
        else:
            if global_scratch_dir:
                assert os.path.dirname(prefix) == global_scratch_dir
            else:
                assert prefix is None


@pytest.mark.parametrize("cwl_runner", ("toil",))
class TestToilRunner:
    def test_tmpdir_prefix(
        tmp_path,
        batch_system,
        dir_local,
        global_scratch_dir,
        local_scratch_dir,
        dir_postfix,
        use_mpi,
        parset,
        runner,
    ):
        """
        Test if command-line option `--tmpdir-prefix` is present when
        expected, and if the value is correct. Note that when running on
        a single machine, an extra sub-directory is created, so we need to
        check for this in the tests.
        """
        try:
            path = runner.args[runner.args.index("--tmpdir-prefix") + 1]
        except ValueError:
            pass
        else:
            prefix = os.path.dirname(path)
            # When running on single machine, prefix contains an extra subdirectory
            if batch_system == "single_machine":
                prefix, postfix = os.path.split(prefix)
            if use_mpi:
                # MPI requires temporary files to be globally available
                if global_scratch_dir:
                    assert prefix == global_scratch_dir
                    # Check if subdirectory's name is as expected 
                    if batch_system == "single_machine":
                        assert postfix == dir_postfix
                else:
                    assert prefix.startswith(parset["dir_working"])
            else:
                assert local_scratch_dir or dir_local
                if local_scratch_dir:
                    assert prefix == local_scratch_dir
                elif dir_local:
                    assert prefix == dir_local
                if batch_system == "single_machine":
                    assert postfix == dir_postfix

    def test_tmp_outdir_prefix(
        tmp_path,
        batch_system,
        global_scratch_dir,
        dir_postfix,
        parset,
        runner,
    ):
        """
        Test if command-line option `--tmp-outdir-prefix` is present when
        expected, and if the value is correct.
        """
        try:
            path = runner.args[runner.args.index("--tmp-outdir-prefix") + 1]
        except ValueError:
            pass
        else:
            prefix, postfix = os.path.split(os.path.dirname(path))
            if global_scratch_dir:
                assert prefix == global_scratch_dir
                assert postfix == dir_postfix
            elif batch_system == "slurm":
                assert prefix.startswith(parset["dir_working"])
                assert postfix == "tmp-out"
            else:
                assert prefix is None

    def test_workdir(
        tmp_path,
        batch_system,
        global_scratch_dir,
        dir_postfix,
        parset,
        runner,
    ):
        """
        Test if command-line option `--workDir` is present when using Slurm,
        and if the value is correct.
        """
        try:
            path = runner.args[runner.args.index("--workDir") + 1]
        except ValueError:
            assert batch_system != "slurm"
        else:
            prefix, postfix = os.path.split(os.path.dirname(path))
            if global_scratch_dir:
                assert prefix == global_scratch_dir
                assert postfix == dir_postfix
            else:
                assert prefix.startswith(parset["dir_working"])
                assert postfix == "tmp-out"

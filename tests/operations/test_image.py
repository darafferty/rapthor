"""
Test cases for the `rapthor.operations.image` module.
"""

import pytest

from pathlib import Path
from tests.cwl.cwl_mock import mocked_cwl_execution
from rapthor.lib.strategy import set_selfcal_strategy

from tests.cwl.cwl_cmdline import generate_command_line
import yaml
from rapthor.operations.image import Image, ImageInitial, ImageNormalize

PATH_TO_OPERATION_STEPS = Path(__file__).parents[2] / "rapthor" / "pipeline" / "steps"


def _mock_cwl_execute(monkeypatch, expected_outputs):
    """
    Monkeypatch the CWL runner's execute method to return expected outputs.
    This allows us to test the image operations without actually running CWL workflows.
    """
    monkeypatch.setattr(
        "rapthor.lib.cwlrunner.BaseCWLRunner.execute",
        lambda self, args, env: mocked_cwl_execution(
            self,
            args,
            env,
            expected_outputs=expected_outputs,
        ),
        raising=False,
    )


def _prepare_field_for_image(field, h5parm_filename="nonexisting_h5parm_file.h5"):
    """
    Prepare the field object for an image operation by setting necessary parameters
    and scanning observations. This is a common setup for image-related tests.
    """
    field.parset["regroup_input_skymodel"] = False
    field.h5parm_filename = str(h5parm_filename)
    field.scan_observations()
    steps = set_selfcal_strategy(field)
    field.update(steps[0], index=1, final=False)
    field.do_predict = False
    field.image_pol = "I"
    field.skip_final_major_iteration = True


def _prepare_field_for_initial_image(field):
    """
    Prepare the field object for an initial image operation.
    """
    field.do_predict = False
    field.scan_observations()
    field.define_full_field_sector()
    field.image_pol = "I"


def _prepare_field_for_normalize_image(field):
    """
    Prepare the field object for a normalize image operation.
    """
    field.do_predict = False
    field.scan_observations()
    field.define_normalize_sector()
    field.image_pol = "I"
    field.apply_screens = False
    field.skip_final_major_iteration = False


def _initialize_operation(operation, do_predict=None, apply_none=None, use_facets=None):
    """
    Set parameters for the given operation based on the provided arguments.
    This allows us to customize the operation setup for different test cases.
    """
    if do_predict is not None:
        operation.do_predict = do_predict
    if apply_none is not None:
        operation.apply_none = apply_none
    if use_facets is not None:
        operation.use_facets = use_facets
    operation.set_parset_parameters()
    operation.set_input_parameters()
    return operation


@pytest.fixture
def h5parm_file(tmp_path):
    """
    Fixture to create a temporary fake h5parm file for testing.
    """
    h5parm = tmp_path / "h5parm_file.h5"
    h5parm.touch()
    return h5parm


@pytest.fixture
def image(field, monkeypatch, expected_image_output):
    """
    Create an instance of the Image operation.
    """
    _prepare_field_for_image(field)
    _mock_cwl_execute(monkeypatch, expected_image_output)
    return _initialize_operation(Image(field=field, index=1), do_predict=False)


@pytest.fixture
def image_last_cycle(field, monkeypatch, expected_image_output_last_cycle):
    """
    Create an instance of the Image operation for the last cycle.
    """
    _prepare_field_for_image(field)
    _mock_cwl_execute(monkeypatch, expected_image_output_last_cycle)
    return _initialize_operation(Image(field=field, index=1), do_predict=False)


@pytest.fixture
def image_initial(field, monkeypatch, expected_image_output):
    """
    Create an instance of the ImageInitial operation.
    """
    _mock_cwl_execute(monkeypatch, expected_image_output)
    _prepare_field_for_initial_image(field)
    return _initialize_operation(ImageInitial(field))


@pytest.fixture
def image_normalize(field, monkeypatch, expected_image_output):
    """
    Create an instance of the ImageNormalize operation.
    """
    _mock_cwl_execute(monkeypatch, expected_image_output)
    _prepare_field_for_normalize_image(field)
    return _initialize_operation(ImageNormalize(field, index=1), do_predict=False)


class TestImage:
    def test_set_parset_parameters(self, field):
        _prepare_field_for_image(field)
        image = Image(field=field, index=1)
        image.set_parset_parameters()
        assert image.parset_parms["use_mpi"] == field.parset["imaging_specific"]["use_mpi"]
        assert image.parset_parms["rapthor_pipeline_dir"] is not None
        assert image.parset_parms["pipeline_working_dir"] is not None

    def test_set_input_parameters(self, field):
        _prepare_field_for_image(field)
        image = Image(field=field, index=1)
        image.set_parset_parameters()
        image.set_input_parameters()
        assert image.input_parms["obs_filename"] is not None
        assert image.input_parms["image_name"] is not None

    def test_finalize(self, image):
        image.run()
        image.finalize()

    def test_run(self, image):
        image.run()
        assert image.is_done()

    @pytest.mark.parametrize("shared_facet_rw", [True, False])
    @pytest.mark.parametrize("use_facets", [True, False])
    @pytest.mark.parametrize("use_mpi", [True, False])
    def test_setting_shared_facet_rw(
        self, field, h5parm_file, shared_facet_rw, use_facets, use_mpi
    ):
        field.parset["imaging_specific"]["use_mpi"] = use_mpi
        field.parset["imaging_specific"]["shared_facet_rw"] = shared_facet_rw
        _prepare_field_for_image(field, h5parm_filename=h5parm_file)
        image = _initialize_operation(
            Image(field, index=1),
            do_predict=False,
            use_facets=use_facets,
        )
        if use_facets:
            assert image.input_parms["shared_facet_rw"] == shared_facet_rw
        else:
            assert not image.input_parms["shared_facet_rw"]

    @pytest.mark.parametrize("use_mpi", [True, False])
    @pytest.mark.parametrize("shared_facet_rw", [True, False])
    def test_shared_facet_rw_in_rendered_workflow(
        self, field, h5parm_file, use_mpi, shared_facet_rw
    ):
        """
        Test that the shared_facet_rw parameter is correctly included in the
        rendered CWL workflow for the image step.
        """
        field.parset["imaging_specific"]["use_mpi"] = use_mpi
        field.parset["imaging_specific"]["shared_facet_rw"] = shared_facet_rw
        _prepare_field_for_image(field, h5parm_filename=h5parm_file)
        image = _initialize_operation(Image(field, index=1), do_predict=False, use_facets=True)
        image.setup()  # renders subpipeline_parset.cwl

        with open(image.subpipeline_parset_file) as f:
            wf = yaml.safe_load(f)

        image_step = next(s for s in wf["steps"] if s["id"] == "image")
        image_step_inputs = {entry["id"]: entry.get("source") for entry in image_step["in"]}

        assert image_step_inputs.get("name") == "image_name"
        assert image_step_inputs.get("shared_facet_reads") == "shared_facet_rw"
        assert image_step_inputs.get("shared_facet_writes") == "shared_facet_rw"

    @pytest.mark.parametrize(
        "cwl_workflow",
        [
            "wsclean_image_facets.cwl",
            "wsclean_mpi_image_facets.cwl",
        ],
    )
    @pytest.mark.parametrize("shared_facet_rw", [True, False])
    def test_wsclean_image_facets_shared_facet_rw_in_final_cli(
        self, tmp_path, cwl_workflow, shared_facet_rw
    ):
        """Verify shared facet read/write options are present in the final WSClean command.

        This test directly passes the correct input names (shared-facet-reads, shared-facet-writes)
        to the CWL step. The bug is upstream in the workflow template that needs to map
        shared_facet_rw to these two separate inputs.
        """

        msdir = tmp_path / "input.ms"
        msdir.mkdir()
        mask = tmp_path / "mask.fits"
        mask.touch()
        h5parm = tmp_path / "solutions.h5"
        h5parm.touch()
        region_file = tmp_path / "facets.reg"
        region_file.touch()

        inputs = {
            "msin": {"class": "Directory", "location": str(msdir)},
            "name": "test-image",
            "mask": {"class": "File", "location": str(mask)},
            "wsclean_imsize": [1024, 1024],
            "wsclean_niter": 1000,
            "wsclean_nmiter": 5,
            "robust": -0.5,
            "min_uv_lambda": 100.0,
            "max_uv_lambda": 100000.0,
            "mgain": 0.8,
            "multiscale": True,
            "scalar_visibilities": False,
            "diagonal_visibilities": True,
            "save_source_list": False,
            "pol": "I",
            "join_polarizations": False,
            "skip_final_iteration": False,
            "cellsize_deg": 0.001,
            "channels_out": 4,
            "deconvolution_channels": 2,
            "fit_spectral_pol": 2,
            "taper_arcsec": 0.0,
            "local_rms_strength": 0.0,
            "local_rms_window": 25.0,
            "local_rms_method": "rms-with-min",
            "wsclean_mem": 8.0,
            "auto_mask": 3.0,
            "auto_mask_nmiter": 2,
            "idg_mode": "cpu",
            "num_threads": 4,
            "num_deconvolution_threads": 2,
            "dd_psf_grid": [2, 2],
            "h5parm": {"class": "File", "location": str(h5parm)},
            "soltabs": "phase000",
            "region_file": {"class": "File", "location": str(region_file)},
            "nnodes": 2,
            "num_gridding_threads": 4,
            "apply_time_frequency_smearing": False,
            # shared_facet_rw is propagated at workflow level to these two CWL inputs.
            "shared_facet_reads": shared_facet_rw,
            "shared_facet_writes": shared_facet_rw,
        }

        cwl_workflow_path = PATH_TO_OPERATION_STEPS / cwl_workflow
        cmd = generate_command_line(
            cwl_workflow_path,
            inputs,
            enable_ext=("mpi" in cwl_workflow_path.name),
        )

        assert cmd is not None
        assert cmd[0] == ("wsclean-mp" if "mpi" in cwl_workflow_path.name else "wsclean")
        assert "-name" in cmd
        assert "test-image" in cmd

        # When shared_facet_rw is True, these flags should appear
        has_reads = "-shared-facet-reads" in cmd
        has_writes = "-shared-facet-writes" in cmd

        assert has_writes is shared_facet_rw
        assert has_reads is shared_facet_rw

    def test_save_model_image(self, field):
        # This is the required setup to configure an Image operation
        # avoiding any other setting will make it throw an expeception
        # refactoring of the fild and image classes seems advisable here
        field.parset["imaging_specific"]["save_filtered_model_image"] = True
        _prepare_field_for_image(field)
        image = _initialize_operation(Image(field, index=1), do_predict=False, apply_none=True)

        assert image.input_parms["save_filtered_model_image"]

    def test_diagnostic_skymodels(self, field):
        """Test to check that the paths to the comparison sky models are set"""
        field.photometry_skymodel = "path/to/photometry_skymodel.txt"
        field.astrometry_skymodel = "path/to/astrometry_skymodel.txt"
        _prepare_field_for_image(field)
        image = _initialize_operation(Image(field, index=1), do_predict=False, apply_none=True)

        assert image.input_parms["photometry_skymodel"]["path"] == "path/to/photometry_skymodel.txt"
        assert image.input_parms["astrometry_skymodel"]["path"] == "path/to/astrometry_skymodel.txt"

    @pytest.mark.parametrize("allow_internet_access", [True, False])
    def test_allow_internet_access(
        self, field, allow_internet_access, monkeypatch, expected_image_output
    ):
        """Test to check that the allow_internet_access flag is set"""
        field.parset["cluster_specific"]["allow_internet_access"] = allow_internet_access
        _prepare_field_for_image(field)
        image = _initialize_operation(Image(field, index=1), do_predict=False, apply_none=True)

        assert image.input_parms["allow_internet_access"] is allow_internet_access
        assert image.parset_parms["allow_internet_access"] is allow_internet_access
        assert image.allow_internet_access is allow_internet_access

        _mock_cwl_execute(monkeypatch, expected_image_output)
        image.run()
        assert image.is_done()

    def test_finalize_without_diagnostic_plots(self, image):
        image.run()
        assert image.is_done()
        image.outputs["sector_diagnostic_plots"][0] = None  # Simulate missing diagnostic plots
        # Handles missing diagnostic plots gracefully without raising an exception
        image.finalize()

    @pytest.mark.parametrize("save_visibilities", [True, False])
    def test_finalize_save_visibilities(self, image, save_visibilities):
        image.field.save_visibilities = save_visibilities
        image.run()
        assert image.is_done()

    def test_sector_extra_images_on_last_cycle(self, image_last_cycle):
        image_last_cycle.run()
        assert image_last_cycle.is_done()
        # Check that the expected I, Q, U, V images are in the outputs
        sector_1 = image_last_cycle.field.imaging_sectors[0]
        assert sector_1.name == "sector_1", (
            f"Expected sector name 'sector_1', got '{sector_1.name}'"
        )
        for pol in ["I", "Q", "U", "V"]:
            assert hasattr(sector_1, f"{pol}_image_file_true_sky"), (
                f"Expected {pol}_image_file_true_sky to be set in sector_1"
            )

    def test_sector_save_supplementary_images_null_mask(self, image):
        image.field.save_supplementary_images = True
        image.set_input_parameters()
        image.run()
        assert image.is_done()
        # Simulate a null mask output and check that it is handled gracefully
        image.outputs["source_filtering_mask"] = [None]
        sector_1 = image.field.imaging_sectors[0]
        assert hasattr(sector_1, "filtering_mask_file"), (
            "Expected filtering_mask_file to be set in sector_1"
        )

    def test_sector_save_supplementary_images(self, image):
        image.field.save_supplementary_images = True
        image.set_input_parameters()
        image.run()
        assert image.is_done()
        sector_1 = image.field.imaging_sectors[0]
        assert hasattr(sector_1, "filtering_mask_file"), (
            "Expected filtering_mask_file to be set in sector_1"
        )
        assert isinstance(sector_1.filtering_mask_file, (str, Path)), (
            f"Expected filtering_mask_file to be a string, got {type(sector_1.filtering_mask_file)}"
        )

    def test_find_in_file_list(self):
        # Test the find_in_file_list method with a sample file list
        file_list = [
            "sector_1-MFS-I-image-pb.fits",
            "sector_1-MFS-I-image.fits",
            "sector_1-MFS-Q-image-pb.fits",
            "sector_1-MFS-Q-image.fits",
            "sector_1-MFS-U-image-pb.fits",
            "sector_1-MFS-U-image.fits",
            "sector_1-MFS-V-image-pb.fits",
            "sector_1-MFS-V-image.fits",
        ]
        type_path_map = Image.find_in_file_list(file_list)
        expected_map = {
            "image_file_true_sky": [
                "sector_1-MFS-I-image-pb.fits",
                "sector_1-MFS-Q-image-pb.fits",
                "sector_1-MFS-U-image-pb.fits",
                "sector_1-MFS-V-image-pb.fits",
            ],
            "image_file_apparent_sky": [
                "sector_1-MFS-I-image.fits",
                "sector_1-MFS-Q-image.fits",
                "sector_1-MFS-U-image.fits",
                "sector_1-MFS-V-image.fits",
            ],
        }
        assert type_path_map == expected_map, f"Expected {expected_map}, got {type_path_map}"

    @pytest.mark.parametrize("pol", ["I", "Q", "U", "V", "X"])
    def test_derive_pol_from_filename(self, pol):
        filename = f"sector_1-MFS-{pol}-image-pb.fits"
        derived_pol = Image.derive_pol_from_filename(filename)
        expected_pol = pol if pol in "IQUV" else "I"
        assert derived_pol == expected_pol, (
            f"Expected polarization '{expected_pol}', got '{derived_pol}'"
        )

    def test_image_operation_sets_mask_file(
        self, field, h5parm_file, monkeypatch, expected_image_output
    ):
        """
        Test that running an image operation with mocked CWL execution
        sets sector.I_mask_file to the correct value when use_clean_mask
        is set to True in the parset.
        """
        field.parset["imaging_specific"]["use_clean_mask"] = True
        _prepare_field_for_image(field, h5parm_filename=h5parm_file)
        _mock_cwl_execute(monkeypatch, expected_image_output)
        image = _initialize_operation(Image(field=field, index=1), do_predict=False)
        image.run()

        # Check that the sector's mask_filename is set
        sector = field.imaging_sectors[0]
        assert sector.I_mask_file is not None
        assert "mask.fits" in sector.I_mask_file
        assert "sector_1" in sector.I_mask_file

    @pytest.mark.parametrize("use_clean_mask", [True, False])
    def test_image_with_previous_mask(
        self, field, h5parm_file, monkeypatch, expected_image_output, use_clean_mask
    ):
        """
        Test that a second image operation uses the mask from the first one.
        The first image operation mocks CWL execution, and the second one checks
        that previous_mask_filename is set correctly in input_parms.
        """
        field.parset["imaging_specific"]["use_clean_mask"] = use_clean_mask
        _prepare_field_for_image(field, h5parm_filename=h5parm_file)
        _mock_cwl_execute(monkeypatch, expected_image_output)

        # First image operation
        image1 = _initialize_operation(Image(field=field, index=1), do_predict=False)
        image1.run()

        # Get the mask file from the first operation
        sector = field.imaging_sectors[0]
        first_mask_file = sector.I_mask_file
        if use_clean_mask:
            assert first_mask_file is not None
        else:
            assert first_mask_file is None

        # Second image operation - simulate reusing the mask from first operation
        # Create a new Image operation with index=2 but don't call field.update()
        image2 = Image(field=field, index=2)
        image2.set_parset_parameters()
        image2.set_input_parameters()

        # Check that previous_mask_filename is set to the mask from the first operation
        assert image2.input_parms["previous_mask_filename"] is not None

        # previous_mask_filename should be a list with one element for one sector
        previous_masks = image2.input_parms["previous_mask_filename"]
        assert isinstance(previous_masks, list)
        assert len(previous_masks) == 1
        # The first mask should be set to the CWL file format of the first operation's mask
        if use_clean_mask:
            assert previous_masks[0] is not None
        else:
            assert previous_masks[0] is None


class TestImageInitial:
    def test_set_parset_parameters(self, image_initial):
        # image_initial.set_parset_parameters()
        pass

    def test_set_input_parameters(self, image_initial):
        # image_initial.set_input_parameters()
        pass

    def test_run(self, image_initial):
        image_initial.run()
        assert image_initial.is_done()

    @pytest.mark.parametrize("dde_method", ["single", "full"])
    def test_initial_image_with_dde_method_single_does_not_raise(self, field, dde_method):
        """
        Regression test for AttributeError when generate_initial_image=True.
        """
        # Set dde_method to 'single' to trigger the bug condition
        field.parset["imaging_specific"]["dde_method"] = dde_method
        field.dde_method = dde_method  # Ensure the field attribute is also set
        _prepare_field_for_initial_image(field)

        image_initial = _initialize_operation(ImageInitial(field))

        # This should NOT raise AttributeError: 'Sector' object has no attribute 'central_patch'
        # The bug causes this to fail because preapply_dde_solutions is incorrectly True
        # Verify apply_none is True and preapply_dde_solutions is False
        assert image_initial.apply_none is True, "apply_none should be True for ImageInitial"
        assert image_initial.preapply_dde_solutions is False, (
            "preapply_dde_solutions should be False for ImageInitial even with dde_method='single'"
        )

    def test_initial_image_save_model_image(self, field):
        field.parset["imaging_specific"]["save_filtered_model_image"] = True
        _prepare_field_for_initial_image(field)
        image_initial = _initialize_operation(ImageInitial(field))

        assert image_initial.input_parms["save_filtered_model_image"]

    def test_finalize_without_diagnostic_plots(self, image_initial):
        image_initial.run()
        assert image_initial.is_done()
        # Simulate missing diagnostic plots
        image_initial.outputs["sector_diagnostic_plots"][0] = None
        # Handles missing diagnostic plots gracefully without raising
        # an exception
        image_initial.finalize()


class TestImageNormalize:
    def test_set_parset_parameters(self, field):
        _prepare_field_for_normalize_image(field)
        image_normalize = ImageNormalize(field=field, index=1)
        image_normalize.set_parset_parameters()
        assert (
            image_normalize.parset_parms["use_mpi"] == field.parset["imaging_specific"]["use_mpi"]
        )
        assert image_normalize.parset_parms["rapthor_pipeline_dir"] is not None
        assert image_normalize.parset_parms["pipeline_working_dir"] is not None
        assert image_normalize.parset_parms["normalize_flux_scale"] is True
        assert image_normalize.parset_parms["image_cube_stokes_list"] == ["I"]

    def test_set_input_parameters(self, field):
        _prepare_field_for_normalize_image(field)
        image_normalize = ImageNormalize(field=field, index=1)
        image_normalize.set_parset_parameters()
        image_normalize.set_input_parameters()
        assert image_normalize.field.normalize_sector.auto_mask == 5.0
        assert image_normalize.field.normalize_sector.auto_mask_nmiter == 2.0
        assert image_normalize.field.normalize_sector.threshisl == 4.0
        assert image_normalize.field.normalize_sector.threshpix == 5.0
        assert image_normalize.field.normalize_sector.max_nmiter == 8
        assert image_normalize.field.normalize_sector.max_wsclean_nchannels == 8
        assert image_normalize.field.normalize_sector.channel_width_hz == 4e6
        assert image_normalize.field.normalize_sector.channel_width_hz == 4e6

        assert image_normalize.apply_normalizations is False
        assert image_normalize.do_predict is False
        assert image_normalize.do_multiscale_clean is False
        assert image_normalize.imaging_parameters["cellsize_arcsec"] == 6.0
        assert image_normalize.imaging_parameters["robust"] == -0.5
        assert image_normalize.imaging_parameters["taper_arcsec"] == 24.0

    def test_finalize(self, image_normalize):
        image_normalize.run()
        image_normalize.finalize()

    def test_run(self, image_normalize):
        image_normalize.run()
        assert image_normalize.is_done()

    def test_save_model_image(self, field):
        field.parset["imaging_specific"]["save_filtered_model_image"] = True
        _prepare_field_for_normalize_image(field)
        image_norm = _initialize_operation(ImageNormalize(field, index=1), do_predict=False)
        assert image_norm.input_parms["save_filtered_model_image"]

    def test_run_with_execute_mock(self, field):
        field.parset["imaging_specific"]["save_filtered_model_image"] = True
        _prepare_field_for_normalize_image(field)

    def test_normalization_skymodel(self, field):
        """Test to check that the paths to the normalization sky models are set"""
        field.normalization_skymodel = "path/to/normalization_skymodel.txt"
        _prepare_field_for_normalize_image(field)
        image_norm = _initialize_operation(ImageNormalize(field, index=1))

        assert (
            image_norm.input_parms["normalization_skymodel"]["path"]
            == "path/to/normalization_skymodel.txt"
        )

    @pytest.mark.parametrize("allow_internet_access", [True, False])
    def test_allow_internet_access(
        self, field, allow_internet_access, monkeypatch, expected_image_output
    ):
        """Test to check that the allow_internet_access flag is set for ImageNormalize"""
        field.parset["cluster_specific"]["allow_internet_access"] = allow_internet_access
        _prepare_field_for_normalize_image(field)
        image_norm = _initialize_operation(ImageNormalize(field, index=1))

        assert image_norm.input_parms["allow_internet_access"] is allow_internet_access
        assert image_norm.parset_parms["allow_internet_access"] is allow_internet_access
        assert image_norm.allow_internet_access is allow_internet_access

        _mock_cwl_execute(monkeypatch, expected_image_output)
        image_norm.run()
        assert image_norm.is_done()


def test_report_sector_diagnostics(sector_name=None, diagnostics_dict=None, log=None):
    # report_sector_diagnostics(sector_name, diagnostics_dict, log)
    pass

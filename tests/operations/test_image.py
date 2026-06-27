"""
Test cases for the `rapthor.operations.image` module.
"""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from rapthor.execution.outputs import file_record
from rapthor.lib.strategy import set_selfcal_strategy
from rapthor.operations.image import (
    Image,
    ImageInitial,
    ImageNormalize,
    report_sector_diagnostics,
)
from rapthor.operations.image_plan import (
    build_image_applycal_steps,
    build_image_facet_solution_controls,
    build_image_prepare_data_steps,
    build_image_wsclean_control_inputs,
)


def _mock_image_flow(monkeypatch, expected_outputs):
    """
    Monkeypatch the image Prefect flow to return expected output records.
    This allows operation finalizer tests to run without external imaging tools.
    """

    def materialize(pipeline_dir, value):
        if isinstance(value, list):
            return [materialize(pipeline_dir, item) for item in value]
        path = pipeline_dir / value
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}" if path.suffix == ".json" else "image")
        return file_record(path)

    def materialize_file(path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}" if path.suffix == ".json" else "image")
        return file_record(path)

    def fake_image_flow(payload, **kwargs):
        pipeline_dir = Path(payload["pipeline_working_dir"])
        sectors = payload["sectors"]
        outputs = {key: materialize(pipeline_dir, value) for key, value in expected_outputs.items()}
        outputs.setdefault("sector_diagnostic_plots", [[] for _ in sectors])
        outputs.setdefault("visibilities", [[] for _ in sectors])

        if any(sector["save_source_list"] for sector in sectors):
            outputs.setdefault(
                "sector_skymodels",
                [
                    [
                        materialize_file(pipeline_dir / f"{sector['image_name']}-sources.txt"),
                        materialize_file(pipeline_dir / f"{sector['image_name']}-sources-pb.txt"),
                    ]
                    for sector in sectors
                ],
            )
        if any(sector["save_filtered_model_image"] for sector in sectors):
            outputs["sector_skymodel_image_fits"] = [
                materialize_file(sector["filtered_model_image_path"]) for sector in sectors
            ]
        if any(sector["use_facets"] for sector in sectors):
            outputs["sector_region_file"] = [
                materialize_file(sector["facet_region_path"]) for sector in sectors
            ]
        if any(sector["make_image_cube"] for sector in sectors):
            outputs["sector_image_cubes"] = [
                [materialize_file(cube_spec["path"]) for cube_spec in sector["image_cube_specs"]]
                for sector in sectors
            ]
            outputs["sector_image_cube_beams"] = [
                [
                    materialize_file(f"{cube_spec['path']}_beams.txt")
                    for cube_spec in sector["image_cube_specs"]
                ]
                for sector in sectors
            ]
            outputs["sector_image_cube_frequencies"] = [
                [
                    materialize_file(f"{cube_spec['path']}_frequencies.txt")
                    for cube_spec in sector["image_cube_specs"]
                ]
                for sector in sectors
            ]
        if any(sector["normalize_flux_scale"] for sector in sectors):
            outputs["sector_source_catalog"] = [
                materialize_file(sector["output_source_catalog_path"]) for sector in sectors
            ]
            outputs["sector_normalize_h5parm"] = [
                materialize_file(sector["output_normalize_h5parm_path"]) for sector in sectors
            ]
        return outputs

    monkeypatch.setattr("rapthor.operations.image.image_flow", fake_image_flow)


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


def _initialize_operation(
    operation,
    do_predict=None,
    apply_none=None,
    use_facets=None,
):
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
    _mock_image_flow(monkeypatch, expected_image_output)
    return _initialize_operation(Image(field=field, index=1), do_predict=False)


@pytest.fixture
def image_last_cycle(field, monkeypatch, expected_image_output_last_cycle):
    """
    Create an instance of the Image operation for the last cycle.
    """
    _prepare_field_for_image(field)
    _mock_image_flow(monkeypatch, expected_image_output_last_cycle)
    return _initialize_operation(Image(field=field, index=1), do_predict=False)


@pytest.fixture
def image_initial(field, monkeypatch, expected_image_output):
    """
    Create an instance of the ImageInitial operation.
    """
    _mock_image_flow(monkeypatch, expected_image_output)
    _prepare_field_for_initial_image(field)
    return _initialize_operation(ImageInitial(field))


@pytest.fixture
def image_normalize(field, monkeypatch, expected_image_output):
    """
    Create an instance of the ImageNormalize operation.
    """
    _mock_image_flow(monkeypatch, expected_image_output)
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

    def test_finalize_uses_matching_diagnostics_for_each_sector(self, field, monkeypatch):
        _prepare_field_for_image(field)
        first_sector = field.imaging_sectors[0]
        second_sector = deepcopy(first_sector)
        second_sector.name = "sector_2"
        second_sector.diagnostics = []
        field.imaging_sectors = [first_sector, second_sector]
        expected_outputs = {
            "sector_I_images": [
                ["sector_1-MFS-I-image-pb.fits", "sector_1-MFS-I-image.fits"],
                ["sector_2-MFS-I-image-pb.fits", "sector_2-MFS-I-image.fits"],
            ],
            "sector_extra_images": [
                [
                    "sector_1-MFS-I-residual.fits",
                    "sector_1-MFS-I-model-pb.fits",
                    "sector_1-MFS-I-dirty.fits",
                ],
                [
                    "sector_2-MFS-I-residual.fits",
                    "sector_2-MFS-I-model-pb.fits",
                    "sector_2-MFS-I-dirty.fits",
                ],
            ],
            "filtered_skymodel_true_sky": ["sector_1.true_sky.txt", "sector_2.true_sky.txt"],
            "filtered_skymodel_apparent_sky": [
                "sector_1.apparent_sky.txt",
                "sector_2.apparent_sky.txt",
            ],
            "pybdsf_catalog": ["sector_1.source_catalog.fits", "sector_2.source_catalog.fits"],
            "sector_diagnostics": ["sector_1_diagnostics.json", "sector_2_diagnostics.json"],
            "source_filtering_mask": ["sector_1_mask.fits", "sector_2_mask.fits"],
        }
        _mock_image_flow(monkeypatch, expected_outputs)
        image = _initialize_operation(Image(field=field, index=1), do_predict=False)

        success, outputs = image.execute_workflow()
        assert success
        Path(outputs["sector_diagnostics"][0]["path"]).write_text(json.dumps({"sector": 1}))
        Path(outputs["sector_diagnostics"][1]["path"]).write_text(json.dumps({"sector": 2}))
        image.outputs = outputs

        image.finalize()

        assert first_sector.diagnostics == [{"sector": 1, "cycle_number": 1}]
        assert second_sector.diagnostics == [{"sector": 2, "cycle_number": 1}]

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
        field.use_mpi = use_mpi
        field.parset["imaging_specific"]["shared_facet_rw"] = shared_facet_rw
        _prepare_field_for_image(field, h5parm_filename=h5parm_file)
        image = _initialize_operation(
            Image(field, index=1),
            do_predict=False,
            use_facets=use_facets,
        )
        assert image.input_parms["shared_facet_rw"] is (shared_facet_rw and use_facets)

    @pytest.mark.parametrize("solution_attr", ["di_h5parm_filename", "fulljones_h5parm_filename"])
    def test_set_parset_parameters_disables_facets_without_dd_scalar_h5parm(
        self, field, h5parm_file, solution_attr
    ):
        _prepare_field_for_image(field, h5parm_filename=h5parm_file)
        setattr(field, solution_attr, str(h5parm_file))
        if solution_attr == "di_h5parm_filename":
            field.h5parm_filename = str(h5parm_file)
        else:
            field.h5parm_filename = None

        image = Image(field, index=1)
        image.set_parset_parameters()

        assert image.use_facets is False

    def test_set_parset_parameters_disables_facets_for_previous_cycle_dd_h5parm(self, field):
        _prepare_field_for_image(field)
        field.h5parm_filename = "/work/solutions/calibrate_2/field-solutions.h5"
        field.dd_h5parm_filename = "/work/solutions/calibrate_2/field-solutions.h5"
        field.dd_h5parm_cycle_number = 2

        image = Image(field, index=3)
        image.set_parset_parameters()

        assert image.use_facets is False

    def test_save_model_image(self, field):
        # This is the required setup to configure an Image operation
        # avoiding any other setting will make it throw an expeception
        # refactoring of the fild and image classes seems advisable here
        field.parset["imaging_specific"]["save_filtered_model_image"] = True
        _prepare_field_for_image(field)
        image = _initialize_operation(
            Image(field, index=1),
            do_predict=False,
            apply_none=True,
            use_facets=False,
        )

        assert image.input_parms["save_filtered_model_image"]

    def test_diagnostic_skymodels(self, field):
        """Test to check that the paths to the comparison sky models are set"""
        field.photometry_skymodel = "path/to/photometry_skymodel.txt"
        field.astrometry_skymodel = "path/to/astrometry_skymodel.txt"
        _prepare_field_for_image(field)
        image = _initialize_operation(
            Image(field, index=1),
            do_predict=False,
            apply_none=True,
            use_facets=False,
        )

        assert image.input_parms["photometry_skymodel"]["path"] == "path/to/photometry_skymodel.txt"
        assert image.input_parms["astrometry_skymodel"]["path"] == "path/to/astrometry_skymodel.txt"

    @pytest.mark.parametrize("allow_internet_access", [True, False])
    def test_allow_internet_access(
        self, field, allow_internet_access, monkeypatch, expected_image_output
    ):
        """Test to check that the allow_internet_access flag is set"""
        field.parset["cluster_specific"]["allow_internet_access"] = allow_internet_access
        _prepare_field_for_image(field)
        image = _initialize_operation(
            Image(field, index=1),
            do_predict=False,
            apply_none=True,
            use_facets=False,
        )

        assert image.input_parms["allow_internet_access"] is allow_internet_access
        assert image.parset_parms["allow_internet_access"] is allow_internet_access
        assert image.allow_internet_access is allow_internet_access

        _mock_image_flow(monkeypatch, expected_image_output)
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

    @pytest.mark.parametrize(
        "calibration_strategy, apply_none, apply_normalizations, apply_amplitudes, expected_steps",
        [
            (
                {"dd": ["fast_phase", "medium_phase"], "di": ["full_jones"]},
                True,
                True,
                False,
                None,
            ),
            (
                {"dd": ["fast_phase", "medium_phase"], "di": ["full_jones"]},
                False,
                False,
                False,
                "[fastphase,mediumphase,fulljones]",
            ),
            (
                {"dd": ["fast_phase", "medium_phase"]},
                False,
                True,
                False,
                "[fastphase,mediumphase,normalization]",
            ),
            (
                {"dd": ["slow_gains"]},
                False,
                True,
                True,
                "[slowgain,normalization]",
            ),
            (
                {"dd": ["fast_phase", "medium_phase", "slow_gains"], "di": ["full_jones"]},
                False,
                True,
                True,
                "[fastphase,mediumphase,slowgain,fulljones,normalization]",
            ),
            (
                {},
                False,
                True,
                False,
                "[normalization]",
            ),
            (
                {},
                False,
                False,
                False,
                None,
            ),
            (
                {"dd": ["fast_phase", "medium_phase"], "di": []},
                False,
                True,
                False,
                "[fastphase,mediumphase,normalization]",
            ),
            (
                {"dd": ["medium_phase", "fast_phase"]},
                False,
                True,
                False,
                "[mediumphase,fastphase,normalization]",
            ),
            (
                {"dd": ["slow_gains", "fast_phase"], "di": ["full_jones"]},
                False,
                True,
                True,
                "[slowgain,fastphase,fulljones,normalization]",
            ),
            (
                {"di": ["full_jones"], "dd": ["fast_phase", "slow_gains"]},
                False,
                True,
                True,
                "[fulljones,fastphase,slowgain,normalization]",
            ),
            # Cases with apply_normalizations=False: normalization step should be absent
            (
                {"dd": ["fast_phase", "medium_phase"]},
                False,
                False,
                False,
                "[fastphase,mediumphase]",
            ),
            (
                {"dd": ["slow_gains"]},
                False,
                False,
                True,
                "[slowgain]",
            ),
            (
                {"dd": ["fast_phase", "medium_phase", "slow_gains"], "di": ["full_jones"]},
                False,
                False,
                True,
                "[fastphase,mediumphase,slowgain,fulljones]",
            ),
            (
                {"di": ["full_jones"], "dd": ["fast_phase", "slow_gains"]},
                False,
                False,
                True,
                "[fulljones,fastphase,slowgain]",
            ),
            # DI-only cases (no DD solutions)
            (
                {"di": ["full_jones"]},
                False,
                True,
                False,
                "[fulljones,normalization]",
            ),
            (
                {"di": ["full_jones"]},
                False,
                False,
                False,
                "[fulljones]",
            ),
            # DI slow_gains cases
            (
                {"di": ["slow_gains"]},
                False,
                True,
                True,
                "[slowgain,normalization]",
            ),
            (
                {"di": ["slow_gains"]},
                False,
                False,
                True,
                "[slowgain]",
            ),
            (
                {"di": ["fast_phase", "medium_phase", "slow_gains"]},
                False,
                False,
                True,
                "[fastphase]",
            ),
            (
                {"di": ["slow_gains", "full_jones"]},
                False,
                True,
                True,
                "[slowgain,fulljones,normalization]",
            ),
            (
                {"di": ["slow_gains", "full_jones"]},
                False,
                False,
                True,
                "[slowgain,fulljones]",
            ),
            # Edge case: slow_gains in strategy but no amplitude solutions present
            # (e.g. the slow-gain solve failed). slowgain must be omitted to prevent
            # DP3 from crashing when amplitude000 is absent from the H5parm.
            (
                {"dd": ["fast_phase", "medium_phase", "slow_gains"]},
                False,
                False,
                False,
                "[fastphase,mediumphase]",
            ),
        ],
    )
    def test_build_applycal_steps(
        self,
        field,
        h5parm_file,
        calibration_strategy,
        apply_none,
        apply_normalizations,
        apply_amplitudes,
        expected_steps,
    ):
        """Test that _build_applycal_steps returns the correct DP3 step string."""
        _prepare_field_for_image(field, h5parm_filename=h5parm_file)
        image = Image(field=field, index=1)
        image.set_parset_parameters()
        image.use_facets = False
        image.apply_none = apply_none
        image.apply_normalizations = apply_normalizations
        field.calibration_strategy = calibration_strategy

        image.apply_amplitudes = apply_amplitudes

        # Create a temporary fake normalize/fulljones h5parm so FileRecord can resolve it,
        # but only when the respective calibration was actually performed.
        if apply_normalizations:
            field.normalize_h5parm = str(h5parm_file)
        if "full_jones" in calibration_strategy.get("di", []):
            field.fulljones_h5parm_filename = str(h5parm_file)

        steps, _, _ = image._build_applycal_steps()
        assert steps == expected_steps

    def test_build_applycal_steps_keeps_mediumphase_for_imaging_prepare_data(
        self, field, h5parm_file
    ):
        """Medium phase is an imaging applycal step, not a DD calibration pre-apply step."""
        _prepare_field_for_image(field, h5parm_filename=h5parm_file)
        field.dd_h5parm_filename = str(h5parm_file)
        field.calibration_strategy = {"dd": ["fast_phase", "medium_phase"]}

        image = Image(field=field, index=1)
        image.use_facets = False
        image.set_parset_parameters()

        steps, _, _ = image._build_applycal_steps()

        assert steps == "[fastphase,mediumphase]"

    def test_build_applycal_steps_prefers_dd_scalar_h5parm_when_mixed(
        self, field, h5parm_file, tmp_path
    ):
        """DD scalar products are selected over DI scalar products when both exist."""
        di_h5parm = tmp_path / "di-solutions.h5"
        di_h5parm.touch()
        _prepare_field_for_image(field, h5parm_filename=h5parm_file)
        field.dd_h5parm_filename = str(h5parm_file)
        field.di_h5parm_filename = str(di_h5parm)
        field.calibration_strategy = {
            "di": ["fast_phase"],
            "dd": ["fast_phase", "slow_gains"],
        }

        image = Image(field=field, index=1)
        image.use_facets = False
        image.set_parset_parameters()
        image.apply_amplitudes = True

        steps, _, _ = image._build_applycal_steps()

        assert steps == "[fastphase,slowgain]"
        assert image._selected_applycal_h5parm == str(h5parm_file)

    def test_build_image_applycal_steps_prefers_dd_scalar_plan(self):
        steps, selected_h5parm = build_image_applycal_steps(
            {"di": ["fast_phase"], "dd": ["fast_phase", "slow_gains"]},
            dd_h5parm="dd-solutions.h5",
            di_h5parm="di-solutions.h5",
            has_fulljones_h5parm=False,
            use_facets=False,
            apply_amplitudes=True,
            apply_normalizations=False,
            apply_none=False,
        )

        assert steps == ["fastphase", "slowgain"]
        assert selected_h5parm == "dd-solutions.h5"

    @pytest.mark.parametrize(
        "preapply, average, bda_timebase, regular, screens, expected_steps",
        [
            (False, True, 10.0, True, False, ["applybeam", "shift", "avg", "bdaavg"]),
            (True, True, 10.0, True, True, ["applybeam", "shift", "applycal", "avg"]),
            (True, False, 10.0, False, False, ["applybeam", "shift", "applycal"]),
        ],
    )
    def test_build_image_prepare_data_steps(
        self, preapply, average, bda_timebase, regular, screens, expected_steps
    ):
        assert (
            build_image_prepare_data_steps(
                preapply_solutions=preapply,
                average_visibilities=average,
                image_bda_timebase=bda_timebase,
                all_channels_regular=regular,
                apply_screens=screens,
            )
            == expected_steps
        )

    @pytest.mark.parametrize(
        "image_pol, combine_method, niters, disable_clean, expected",
        [
            ("I", "link", [100, 200], False, (False, False, [100, 200])),
            ("IQUV", "link", [100, 200], False, ("I", False, [100, 200])),
            (["I", "Q"], "join", [100, 200], True, (False, True, [0, 0])),
        ],
    )
    def test_build_image_wsclean_control_inputs(
        self, image_pol, combine_method, niters, disable_clean, expected
    ):
        assert (
            build_image_wsclean_control_inputs(
                image_pol,
                combine_method,
                niters,
                disable_clean=disable_clean,
            )
            == expected
        )

    @pytest.mark.parametrize(
        "image_pol, apply_amplitudes, apply_diagonal, expected",
        [
            (
                "I",
                False,
                False,
                {
                    "soltabs": "phase000",
                    "diagonal_visibilities": False,
                    "scalar_visibilities": True,
                },
            ),
            (
                "I",
                True,
                True,
                {
                    "soltabs": "amplitude000,phase000",
                    "diagonal_visibilities": True,
                    "scalar_visibilities": False,
                },
            ),
            (
                "I",
                True,
                False,
                {
                    "soltabs": "amplitude000,phase000",
                    "diagonal_visibilities": False,
                    "scalar_visibilities": True,
                },
            ),
            (
                "IQUV",
                True,
                True,
                {
                    "soltabs": "amplitude000,phase000",
                    "diagonal_visibilities": False,
                    "scalar_visibilities": False,
                },
            ),
        ],
    )
    def test_build_image_facet_solution_controls(
        self, image_pol, apply_amplitudes, apply_diagonal, expected
    ):
        assert (
            build_image_facet_solution_controls(
                image_pol,
                apply_amplitudes=apply_amplitudes,
                apply_diagonal_solutions=apply_diagonal,
            )
            == expected
        )

    def test_build_applycal_steps_uses_dd_h5parm_for_facets_without_preapply(
        self, field, h5parm_file
    ):
        """Facet imaging receives the DD h5parm but does not preapply DD solves."""
        _prepare_field_for_image(field, h5parm_filename=h5parm_file)
        field.dd_h5parm_filename = str(h5parm_file)
        field.calibration_strategy = {"dd": ["fast_phase", "medium_phase", "slow_gains"]}

        image = Image(field=field, index=1)
        image.set_parset_parameters()
        image.use_facets = True
        image.apply_amplitudes = True
        image.apply_normalizations = False

        steps, _, _ = image._build_applycal_steps()

        assert steps is None
        assert image._selected_applycal_h5parm == str(h5parm_file)

    def test_build_image_applycal_steps_selects_facets_h5parm_without_preapply(self):
        steps, selected_h5parm = build_image_applycal_steps(
            {"dd": ["fast_phase", "medium_phase", "slow_gains"]},
            dd_h5parm="dd-solutions.h5",
            di_h5parm=None,
            has_fulljones_h5parm=False,
            use_facets=True,
            apply_amplitudes=True,
            apply_normalizations=False,
            apply_none=False,
        )

        assert steps == []
        assert selected_h5parm == "dd-solutions.h5"

    def test_build_applycal_steps_ignores_previous_cycle_dd_h5parm(self, field):
        _prepare_field_for_image(field)
        field.h5parm_filename = "/work/solutions/calibrate_2/field-solutions.h5"
        field.dd_h5parm_filename = "/work/solutions/calibrate_2/field-solutions.h5"
        field.dd_h5parm_cycle_number = 2
        field.calibration_strategy = {"dd": ["fast_phase", "medium_phase", "slow_gains"]}

        image = Image(field=field, index=3)
        image.use_facets = False
        image.set_parset_parameters()
        image.apply_amplitudes = True
        image.apply_normalizations = False

        steps, _, _ = image._build_applycal_steps()

        assert steps is None
        assert image._selected_applycal_h5parm is None

    def test_set_input_parameters_dd_slow_only_facets_get_h5parm(self, field, h5parm_file):
        """Single DD slow-gain phase solves still provide an h5parm to WSClean facets."""
        _prepare_field_for_image(field, h5parm_filename=h5parm_file)
        field.dd_h5parm_filename = str(h5parm_file)
        field.calibration_strategy = {"dd": ["slow_gains"]}
        field.apply_amplitudes = False

        image = Image(field=field, index=1)
        image.use_facets = True
        image.apply_normalizations = False
        image.set_parset_parameters()
        image.set_input_parameters()

        assert image.input_parms["prepare_data_applycal_steps"] is None
        assert image.input_parms["h5parm"]["path"] == str(h5parm_file)

    def test_image_operation_sets_mask_file(
        self, field, h5parm_file, monkeypatch, expected_image_output
    ):
        """
        Test that running an image operation with a mocked image flow
        sets sector.I_mask_file to the correct value when use_clean_mask
        is set to True in the parset.
        """
        field.parset["imaging_specific"]["use_clean_mask"] = True
        _prepare_field_for_image(field, h5parm_filename=h5parm_file)
        _mock_image_flow(monkeypatch, expected_image_output)
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
        The first image operation mocks the image flow, and the second one checks
        that previous_mask_filename is set correctly in input_parms.
        """
        field.parset["imaging_specific"]["use_clean_mask"] = use_clean_mask
        _prepare_field_for_image(field, h5parm_filename=h5parm_file)
        _mock_image_flow(monkeypatch, expected_image_output)

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
        # The first mask should be set to the File record from the first operation's mask
        if use_clean_mask:
            assert previous_masks[0] is not None
        else:
            assert previous_masks[0] is None


class TestImageInitial:
    def test_set_parset_parameters(self, image_initial):
        assert image_initial.parset_parms["apply_screens"] is False
        assert image_initial.parset_parms["use_facets"] is False
        assert image_initial.parset_parms["save_source_list"] is True
        assert image_initial.parset_parms["preapply_dde_solutions"] is False
        assert image_initial.parset_parms["make_image_cube"] is False
        assert image_initial.parset_parms["compress_images"] is (
            image_initial.field.compress_selfcal_images
        )

    def test_set_input_parameters(self, image_initial):
        assert image_initial.input_parms["image_name"] == [
            image_initial.field.full_field_sector.name
        ]
        assert image_initial.input_parms["pol"] == "I"
        assert image_initial.input_parms["save_source_list"] is True
        assert image_initial.input_parms["skip_final_iteration"] is True
        assert image_initial.do_predict is False
        assert image_initial.do_multiscale_clean is True
        assert image_initial.apply_amplitudes is False
        assert image_initial.apply_fulljones is False
        assert image_initial.apply_normalizations is False

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
        field.normalization_skymodels = [
            "path/to/normalization_skymodel_1.txt",
            "path/to/normalization_skymodel_2.txt",
        ]
        field.normalization_reference_frequencies = [142000000.0, 142001000.0]
        _prepare_field_for_normalize_image(field)
        image_norm = _initialize_operation(ImageNormalize(field, index=1))
        assert (
            image_norm.input_parms["normalization_skymodels"][0]["path"]
            == "path/to/normalization_skymodel_1.txt"
        )
        assert (
            image_norm.input_parms["normalization_skymodels"][1]["path"]
            == "path/to/normalization_skymodel_2.txt"
        )
        assert image_norm.input_parms["normalization_reference_frequencies"] == [
            142000000.0,
            142001000.0,
        ]

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

        _mock_image_flow(monkeypatch, expected_image_output)
        image_norm.run()
        assert image_norm.is_done()


def test_report_sector_diagnostics_returns_best_non_nvss_flux_ratio(mocker):
    diagnostics = {
        "theoretical_rms": 1.0e-5,
        "min_rms_flat_noise": 2.0e-5,
        "median_rms_flat_noise": 3.0e-5,
        "dynamic_range_global_flat_noise": 10.0,
        "min_rms_true_sky": 2.5e-5,
        "median_rms_true_sky": 3.5e-5,
        "dynamic_range_global_true_sky": 12.0,
        "nsources": 10,
        "freq": 150.0e6,
        "beam_fwhm": [0.001, 0.002, 45.0],
        "unflagged_data_fraction": 0.9,
        "meanClippedRatio_TGSS": 1.2,
        "stdClippedRatio_TGSS": 0.2,
        "meanClippedRatio_LOTSS": 0.95,
        "stdClippedRatio_LOTSS": 0.05,
        "meanClippedRatio_NVSS": 0.8,
        "stdClippedRatio_NVSS": 0.01,
        "meanClippedRAOffsetDeg": 1.0 / 3600.0,
        "stdClippedRAOffsetDeg": 0.1 / 3600.0,
        "meanClippedDecOffsetDeg": -2.0 / 3600.0,
        "stdClippedDecOffsetDeg": 0.2 / 3600.0,
    }
    log = mocker.Mock()

    ratio, std = report_sector_diagnostics("sector_1", diagnostics, log)

    assert ratio == 0.95
    assert std == 0.1
    log.warning.assert_not_called()

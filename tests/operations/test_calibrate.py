"""
Test cases for the `rapthor.operations.calibrate` module.
"""

from pathlib import Path

import pytest

from rapthor.operations.calibrate import CalibrateDD, CalibrateDI


@pytest.fixture
def calibrate_field(operation_parset, mocker, single_source_sky_model):
def calibrate_field(operation_parset, mocker, single_source_sky_model):
    """Create a mock field object for testing a Calibrate operation."""

    class Field:
        def __init__(self, parset):
            self.parset = parset
            self.calibration_diagnostics = []
            self.observations = []

            # Needed for arithmetic in set_input_parameters
            self.ntimechunks = 2
            self.calibration_skymodel_file = str(single_source_sky_model["path"])
            self.ra = 0.0
            self.dec = 0.0
            self.fast_smoothnessconstraint = 1.0
            self.medium_smoothnessconstraint = 1.0
            self.slow_smoothnessconstraint = 1.0
            self.data_colname = "DATA"

            # Callables that need to be mocked
            self.scan_h5parms = mocker.MagicMock()
            self.set_obs_parameters = mocker.MagicMock()
            self.get_obs_parameters = mocker.MagicMock(return_value=[1])
            self.get_source_distances = mocker.MagicMock(return_value=(["src"], [0.1]))

            # Remaining attributes accessed in set_input_parameters
            self.fast_smoothnessrefdistance = 0.0
            self.medium_smoothnessrefdistance = 0.0
            self.max_normalization_delta = 0.3
            self.scale_normalization_delta = True
            self.llssolver = "qr"
            self.maxiter = 50
            self.propagatesolutions = True
            self.solveralgorithm = "directionsolve"
            self.onebeamperpatch = False
            self.stepsize = 0.2
            self.stepsigma = 0.0
            self.tolerance = 1e-4
            self.solve_min_uv_lambda = 0.0
            self.parallelbaselines = False
            self.sagecalpredict = False
            self.solverlbfgs_dof = 200.0
            self.solverlbfgs_iter = 4
            self.solverlbfgs_minibatches = 1
            self.fast_datause = "full"
            self.medium_datause = "full"
            self.slow_datause = "full"
            self.correct_smearing_in_calibration = False
            self.sector_bounds_deg = "[0,0,1,1]"
            self.sector_bounds_mid_deg = "[0.5,0.5]"
            self.calibrator_patch_names = []
            self.calibrator_fluxes = []
            self.antenna = "HBA"
            self.stations = []

            # Feature flags (defaults off; individual tests override as needed)
            self.apply_diagonal_solutions = False
            self.use_image_based_predict = False
            self.do_slowgain_solve = False
            self.apply_normalizations = False
            self.generate_screens = False
            self.normalize_h5parm = None
            self.calibrate_bda_timebase = 0
            self.calibrate_bda_frequencybase = 0
            self.fast_phases_h5parm_filename = None
            self.medium1_phases_h5parm_filename = None
            self.medium2_phases_h5parm_filename = None
            self.slow_gains_h5parm_filename = None

    return Field(operation_parset)


def check_makedirs(mock_makedirs, *expected_paths):
    """Helper function to check that makedirs was called with the expected paths."""
    for path in expected_paths:
        mock_makedirs.assert_any_call(str(path), exist_ok=True)
    assert mock_makedirs.call_count == len(expected_paths)


def finalize_prepare_plots(pipelines_path, plots_path):
    """Helper function to prepare the plot files for the finalize() tests."""
    # Create dummy plot files. finalize() should copy them to the plots directory.
    pipelines_path.mkdir(parents=True)
    (pipelines_path / "plot1.png").touch()
    (pipelines_path / "plot2.png").touch()

    # Simulate one existing plot in the plots directory. finalize() should remove it.
    plots_path.mkdir(parents=True)
    (plots_path / "plot2.png").touch()


def parse_dp3(dp3_string):
    """
    Helper function to convert DP3 string like:
        "[solve1,solve2]"
    into:
        ["solve1", "solve2"]
    """
    return [x.strip() for x in dp3_string.strip("[]").split(",") if x.strip()]


class TestCalibrateDD:
    def test_set_parset_parameters(self):
        # calibrate_dd.set_parset_parameters()
        pass

    BDA_CASES = [
        # (bda_time, bda_freq, slowgain, expected_dp3)
        (0, 0, False, ["solve1", "solve2"]),
        (1, 1, False, ["avg", "solve1", "solve2", "null"]),
        (1, 1, True, ["avg", "solve1", "solve2", "solve3", "solve4", "null"]),
    ]

    def test_set_input_parameters_dd_all_disabled(self, calibrate_field, mocker):
        """
        Everything disabled,  minimal dp3, no optional features.
        """

        # calibrate_field.apply_diagonal_solutions = False
        # calibrate_field.use_image_based_predict = False
        # calibrate_field.do_slowgain_solve = False
        # calibrate_field.apply_normalizations = False

        # calibrate_field.calibrate_bda_timebase = 0
        # calibrate_field.calibrate_bda_frequencybase = 0

        # calibrate_field.fast_phases_h5parm_filename = None
        # calibrate_field.medium1_phases_h5parm_filename = None
        # calibrate_field.medium2_phases_h5parm_filename = None
        # calibrate_field.slow_gains_h5parm_filename = None

        mocker.patch("os.path.exists", return_value=False)
        calibrate_dd = CalibrateDD(field=calibrate_field, index=1)

        calibrate_dd.set_input_parameters()
        dp3 = parse_dp3(calibrate_dd.input_parms["dp3_steps"])
        params = calibrate_dd.input_parms

        assert dp3 == ["solve1", "solve2"]

        assert params["solution_combine_mode"] == "p1p2a2_scalar"
        assert params["normalize_h5parm"] is None
        assert params["ddecal_applycal_steps"] is None

        assert params["fast_initialsolutions_h5parm"] is None
        assert params["slow_initialsolutions_h5parm"] is None

    @pytest.mark.parametrize(
        "bda_time, bda_freq, slowgain, expected_dp3",
        BDA_CASES,
    )
    def test_set_input_parameters_dd_bda_cases(
        self, calibrate_field, mocker, bda_time, bda_freq, slowgain, expected_dp3
    ):
        """
        Test the effect of BDA and slowgain settings on the dp3 steps.
        """
        calibrate_field.calibrate_bda_timebase = bda_time
        calibrate_field.calibrate_bda_frequencybase = bda_freq
        calibrate_field.do_slowgain_solve = slowgain
        #calibrate_field.use_image_based_predict = False

        # set os.path.exists to always return False to test the default dp3 steps without loading from existing h5parms
        mocker.patch("os.path.exists", return_value=False)
        calibrate_dd = CalibrateDD(field=calibrate_field, index=1)
        calibrate_dd.set_input_parameters()
        dp3 = parse_dp3(calibrate_dd.input_parms["dp3_steps"])

        assert dp3 == expected_dp3

    IMAGE_BASED_PREDICT_CASES = [
        # (normalize, expected_prefix, expect_applycal)
        (False, ["predict", "applybeam"], False),
        (True, ["predict", "applybeam", "applycal"], True),
    ]

    @pytest.mark.parametrize(
        "normalize, expected_prefix, expect_applycal",
        IMAGE_BASED_PREDICT_CASES,
    )
    def test_set_input_parameters_dd_ibp_cases(
        self, calibrate_field, mocker, tmp_path, normalize, expected_prefix, expect_applycal
    ):
        """
        Test the effect of image-based predict and normalization settings on the dp3 steps and applycal steps.
        """
        calibrate_field.use_image_based_predict = True
        calibrate_field.apply_normalizations = normalize
        if normalize:
            calibrate_field.normalize_h5parm = str(tmp_path / "normalize.h5parm")

        # set os.path.exists to always return False to test the default dp3 steps without loading from existing h5parms
        mocker.patch("os.path.exists", return_value=False)
        calibrate_dd = CalibrateDD(field=calibrate_field, index=1)
        calibrate_dd.set_input_parameters()
        dp3 = parse_dp3(calibrate_dd.input_parms["dp3_steps"])
        params = calibrate_dd.input_parms

        assert dp3[: len(expected_prefix)] == expected_prefix

        if expect_applycal:
            assert params["ddecal_applycal_steps"] == "[normalization]"
            assert params["applycal_steps"] == "[normalization]"
        else:
            assert params["ddecal_applycal_steps"] is None

    COMBINE_MODE_CASES = [
        # (diagonal_flag, expected_mode)
        (True, "p1p2a2_diagonal"),
        (False, "p1p2a2_scalar"),
    ]

    @pytest.mark.parametrize(
        "diagonal_flag, expected_mode",
        COMBINE_MODE_CASES,
    )
    def test_set_input_parameters_dd_solution_combine_mode_switch(
        self, calibrate_field, mocker, diagonal_flag, expected_mode
    ):
        """
        Test the effect of diagonal solutions on the solution_combine_mode.
        """
        calibrate_field.apply_diagonal_solutions = diagonal_flag

        mocker.patch("os.path.exists", return_value=False)
        calibrate_dd = CalibrateDD(field=calibrate_field, index=1)
        calibrate_dd.set_input_parameters()

        assert calibrate_dd.input_parms["solution_combine_mode"] == expected_mode

    BASELINES_CORE_CASES = [
        (
            "LBA",
            ["CS001LBA", "CS002LBA", "RS106LBA", "DE601LBA", "UK608LBA"],
            "[CR]*&&;!DE601LBA;!UK608LBA",
        ),
        (
            "HBA",
            ["CS003HBA0", "RS106HBA0", "DE601HBA", "UK902HBA"],
            "[CR]*&&;!DE601HBA;!UK902HBA",
        ),
    ]

    @pytest.mark.parametrize("antenna, stations, expected", BASELINES_CORE_CASES)
    def test_get_baselines_core(self, calibrate_field, antenna, stations, expected):
        calibrate_field.antenna = antenna
        calibrate_field.stations = stations
        calibrate_dd = CalibrateDD(field=calibrate_field, index=1)

        baselines = calibrate_dd.get_baselines_core()
        assert baselines == expected

    SUPERTERP_STATION_CASES = [
        (
            "HBA",
            ["RS106HBA0", "DE601HBA"],
            [],
        ),
        (
            "HBA",
            ["CS003HBA0", "RS106HBA0", "CS007HBA1", "DE601HBA"],
            ["CS003HBA0", "CS007HBA1"],
        ),
        (
            "LBA",
            ["RS205LBA", "CS004LBA", "CS007LBA", "DE601LBA"],
            ["CS004LBA", "CS007LBA"],
        ),
    ]

    @pytest.mark.parametrize("antenna,stations,expected", SUPERTERP_STATION_CASES)
    def test_get_superterp_stations(self, calibrate_field, antenna, stations, expected):
        calibrate_field.antenna = antenna
        calibrate_field.stations = stations
        calibrate_dd = CalibrateDD(field=calibrate_field, index=1)
        assert calibrate_dd.get_superterp_stations() == expected

    CORE_STATION_CASES = [
        (
            "HBA",
            True,
            ["RS106HBA0", "CS002HBA0", "DE601HBA"],
            ["CS002HBA0", "RS106HBA0"],
        ),
        (
            "HBA",
            False,
            ["RS106HBA0", "CS002HBA0", "DE601HBA"],
            ["CS002HBA0"],
        ),
        (
            "LBA",
            True,
            ["RS205LBA", "CS003LBA", "CS999LBA"],
            ["CS003LBA", "RS205LBA"],
        ),
        (
            "LBA",
            False,
            ["RS205LBA", "CS003LBA", "CS999LBA"],
            ["CS003LBA"],
        ),
        (
            "HBA",
            True,
            ["DE601HBA", "DE602HBA"],
            [],
        ),
    ]

    @pytest.mark.parametrize("antenna,include_remote,stations,expected", CORE_STATION_CASES)
    def test_get_core_stations(self, calibrate_field, antenna, include_remote, stations, expected):
        calibrate_field.antenna = antenna
        calibrate_field.stations = stations
        calibrate_dd = CalibrateDD(field=calibrate_field, index=1)
        result = calibrate_dd.get_core_stations(include_nearest_remote=include_remote)
        assert result == expected

    @pytest.mark.parametrize("cycle,have_full_field_sector", [(1, False), (1, True), (2, False)])
    def test_get_model_image_parameters(
        self, single_source_sky_model, calibrate_field, mocker, cycle, have_full_field_sector
    ):
        sky_model = single_source_sky_model
        field = calibrate_field

        bandwidth = 1e6  # hardcoded value in Rapthor.
        cellsize_arcsec = 2
        cellsize_degrees = cellsize_arcsec / 3600.0
        cellsize_pixels = 1800
        width_ra_pixels = field.sector_bounds_width_ra * cellsize_pixels
        width_dec_pixels = field.sector_bounds_width_dec * cellsize_pixels
        source_distance_ra = 6
        source_distance_dec = 7
        expected_source_distance_size = [25200, 25200]  # 2 * 7 * cellsize_pixels

        # Setup the field for the test.
        field.parset["imaging_specific"] = {"cellsize_arcsec": cellsize_arcsec}
        field.get_source_distances.return_value = ("foo", [source_distance_ra, source_distance_dec])
        if have_full_field_sector:
            field.full_field_sector = mocker.NonCallableMagicMock()
            field.full_field_sector.cellsize_deg = cellsize_degrees
            field.full_field_sector.imsize = [width_ra_pixels, width_dec_pixels]

        # Act
        calibrate_dd = CalibrateDD(field, index=cycle)
        frequency_bandwidth, center_coords, size, cellsize = (
            calibrate_dd.get_model_image_parameters()
        )

        # Assert. In this test, all scenarios yield equal values, except for the size.
        assert frequency_bandwidth == [sky_model["reference_frequency"], bandwidth]
        assert center_coords == ("2:48:00.000000", "-42.00.00.000000")
        if cycle == 1 and not have_full_field_sector:
            field.get_source_distances.assert_called_once_with(
                {sky_model["name"]: [sky_model["ra"], sky_model["dec"]]}
            )
            assert size == expected_source_distance_size
        else:
            field.get_source_distances.assert_not_called()
            assert size == [width_ra_pixels, width_dec_pixels]
        assert cellsize == cellsize_degrees


class TestCalibrateDI:
    def test_set_parset_parameters(self):
        # calibrate_di.set_parset_parameters()
        pass

    def test_set_input_parameters_di(self, calibrate_field, mocker):
        # Set up the field attributes required by CalibrateDI.set_input_parameters()
        calibrate_field.parset["cluster_specific"]["max_threads"] = 4

        calibrate_field.set_obs_parameters = mocker.MagicMock()
        calibrate_field.get_obs_parameters = mocker.MagicMock(return_value=[])
        calibrate_field.ntimechunks = 2

        calibrate_field.smoothnessconstraint_fulljones = 1.0
        calibrate_field.max_normalization_delta = 0.3
        calibrate_field.llssolver = "qr"
        calibrate_field.maxiter = 50
        calibrate_field.propagatesolutions = True
        calibrate_field.solveralgorithm = "directionsolve"
        calibrate_field.stepsize = 0.2
        calibrate_field.stepsigma = 0.0
        calibrate_field.tolerance = 1e-4
        calibrate_field.solve_min_uv_lambda = 0.0
        calibrate_field.solverlbfgs_dof = 200.0
        calibrate_field.solverlbfgs_iter = 4
        calibrate_field.solverlbfgs_minibatches = 1
        calibrate_field.correct_smearing_in_calibration = False

        calibrate_di = CalibrateDI(field=calibrate_field, index=1)
        calibrate_di.set_input_parameters()

        assert calibrate_di.collected_h5parm_fulljones == "fulljones_gains.h5"
        assert calibrate_di.input_parms["data_colname"] == "DATA"
        assert calibrate_di.input_parms["collected_h5parm_fulljones"] == "fulljones_gains.h5"
        assert calibrate_di.input_parms["output_h5parm_fulljones"] == [
            "fulljones_gain_0.h5parm",
            "fulljones_gain_1.h5parm",
        ]
        assert calibrate_di.input_parms["smoothnessconstraint_fulljones"] == 1.0
        assert calibrate_di.input_parms["max_normalization_delta"] == 0.3
        assert calibrate_di.input_parms["max_threads"] == 4


class TestCalibrate:
    @pytest.mark.parametrize(
        "scenario, batch_system, generate_screens, use_image_based_predict",
        [
            ("dd_fast_only", "slurm", False, False),
            ("dd_fast_only", "slurm", True, False),
            ("dd_with_slowgain", "some_other_batch_system", False, True),
            ("dd_with_slowgain", "some_other_batch_system", True, True),
            ("di_fulljones", "slurm", "don't", "care"),
            ("di_fulljones", "some_other_batch_system", "don't", "care"),
        ],
    )
    def test_set_parset_parameters(
        self, calibrate_field, scenario, batch_system, generate_screens, use_image_based_predict
    ):
        is_dd = scenario.startswith("dd")
        with_slow = scenario == "dd_with_slowgain"
        max_cores = 42

        # Setup field object
        calibrate_field.generate_screens = generate_screens
        calibrate_field.use_image_based_predict = use_image_based_predict
        calibrate_field.do_slowgain_solve = with_slow
        calibrate_field.parset["cluster_specific"]["batch_system"] = batch_system
        calibrate_field.parset["cluster_specific"]["max_cores"] = max_cores

        # Setup calibrate object
        calibrate = (
            CalibrateDD(calibrate_field, index=1)
            if is_dd
            else CalibrateDI(calibrate_field, index=2)
        )

        # Act
        calibrate.set_parset_parameters()

        # Assert
        rapthor_pipeline_path = Path(rapthor.__file__).parent / "pipeline"
        assert calibrate.parset_parms["rapthor_pipeline_dir"] == str(rapthor_pipeline_path)

        expected_max_cores = None if batch_system == "slurm" else max_cores
        assert calibrate.parset_parms["max_cores"] == expected_max_cores

        if is_dd:  # CalibrateDD sets some extra parameters.
            expected_use_image_based_predict = generate_screens or use_image_based_predict
            assert calibrate.use_image_based_predict is expected_use_image_based_predict
            assert (
                calibrate.parset_parms["use_image_based_predict"]
                is expected_use_image_based_predict
            )
            assert calibrate.parset_parms["generate_screens"] is generate_screens
            assert calibrate.parset_parms["do_slowgain_solve"] is with_slow

    @pytest.mark.parametrize("scenario", ["dd_fast_only", "dd_with_slowgain", "di_fulljones"])
    def test_finalize(self, mocker, calibrate_field, tmp_path, scenario):
        field = calibrate_field
        is_dd = scenario.startswith("dd")
        with_slow = scenario == "dd_with_slowgain"

        # Setup mocks
        flagged_fraction = 0.042
        mocker.patch(
            "rapthor.lib.miscellaneous.get_flagged_solution_fraction", return_value=flagged_fraction
        )
        mock_makedirs = mocker.patch("os.makedirs")
        mock_remove = mocker.patch("os.remove")
        mock_copy = mocker.patch("shutil.copy")

        # Setup working directory
        workdir_path = tmp_path / "working"
        name = "calibrate_2" if is_dd else "calibrate_di_4"
        solutions_path = workdir_path / "solutions" / name
        solutions_path.mkdir(parents=True)

        # Create an existing solutions file. finalize() should remove it.
        h5parm_filename = "field-solutions.h5" if is_dd else "fulljones-solutions.h5"
        h5parm_path = solutions_path / h5parm_filename
        h5parm_path.touch()

        pipelines_path = workdir_path / "pipelines" / name
        plots_path = workdir_path / "plots" / name
        finalize_prepare_plots(pipelines_path, plots_path)

        # Setup the object itself
        # field.generate_screens = False
        field.do_slowgain_solve = scenario == "dd_with_slowgain"

        calibrate = CalibrateDD(field, index=2) if is_dd else CalibrateDI(field, index=4)

        if is_dd:
            calibrate.combined_h5parms = "combined.test.h5"
            calibrate.fast_h5parm = "fast.test.h5"
            if with_slow:
                calibrate.slow_h5parm = "slow.test.h5"
                calibrate.medium1_h5parm = "medium1.test.h5"
                calibrate.medium2_h5parm = "medium2.test.h5"
        else:
            calibrate.collected_h5parm_fulljones = "collected_fulljones.h5"

        # Ignore os.makedirs calls from the base Operation class constructor.
        mock_makedirs.reset_mock()

        # Act
        calibrate.finalize()

        # Assert
        if is_dd:
            assert field.h5parm_filename == str(h5parm_path)
            assert field.fast_phases_h5parm_filename == str(
                solutions_path / "field-solutions-fast-phase.h5"
            )
            if with_slow:
                assert field.medium1_phases_h5parm_filename == str(
                    solutions_path / "field-solutions-medium1-phase.h5"
                )
                assert field.medium2_phases_h5parm_filename == str(
                    solutions_path / "field-solutions-medium2-phase.h5"
                )
                assert field.slow_gains_h5parm_filename == str(
                    solutions_path / "field-solutions-slow-gain.h5"
                )
        else:
            assert field.fulljones_h5parm_filename == str(h5parm_path)

        check_makedirs(mock_makedirs, solutions_path, plots_path)

        # Check removing and copying solutions.
        mock_remove.assert_any_call(str(h5parm_path))

        if with_slow:
            solution_src_dst_list = [
                ("combined.test.h5", h5parm_filename),
                ("slow.test.h5", "field-solutions-slow-gain.h5"),
                ("medium1.test.h5", "field-solutions-medium1-phase.h5"),
                ("medium2.test.h5", "field-solutions-medium2-phase.h5"),
                ("fast.test.h5", "field-solutions-fast-phase.h5"),
            ]
        elif scenario == "dd_fast_only":
            solution_src_dst_list = [
                ("fast.test.h5", h5parm_filename),
                ("fast.test.h5", "field-solutions-fast-phase.h5"),
            ]
        else:  # di_fulljones scenario
            solution_src_dst_list = [
                ("collected_fulljones.h5", h5parm_filename),
            ]

        for src, dst in solution_src_dst_list:
            mock_copy.assert_any_call(str(pipelines_path / src), str(solutions_path / dst))

        field.scan_h5parms.assert_called_once()

        if is_dd:
            assert field.calibration_diagnostics == [
                {
                    "cycle_number": 2,
                    "solution_flagged_fraction": flagged_fraction,
                }
            ]

        # Check that the correct plot files were removed and copied.
        mock_remove.assert_any_call(str(plots_path / "plot2.png"))
        mock_copy.assert_any_call(str(pipelines_path / "plot1.png"), str(plots_path / "plot1.png"))
        mock_copy.assert_any_call(str(pipelines_path / "plot2.png"), str(plots_path / "plot2.png"))

        assert mock_remove.call_count == 2  # h5parm_path and plot2.png
        assert mock_copy.call_count == len(solution_src_dst_list) + 2

        # finalize() should create a .done file (via the base Operation class).
        assert (pipelines_path / ".done").exists()

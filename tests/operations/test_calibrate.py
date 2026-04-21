"""
Test cases for the `rapthor.operations.calibrate` module.
"""

from pathlib import Path

import numpy as np
import pytest

import rapthor
from rapthor.lib.cwl import CWLDir, CWLFile
from rapthor.lib.operation import DIR as OPERATION_DIR
from rapthor.operations.calibrate import CalibrateDD, CalibrateDI


@pytest.fixture
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
            self.ra = 42.0
            self.dec = -42.0
            self.sector_bounds_mid_ra = self.ra
            self.sector_bounds_mid_dec = self.dec
            self.sector_bounds_width_ra = 4
            self.sector_bounds_width_dec = 5
            self.smoothnessconstraint_fulljones = 1.0
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

    @pytest.mark.parametrize(
        "antenna, stations, expected",
        [
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
        ],
    )
    def test_get_baselines_core(self, calibrate_field, antenna, stations, expected):
        calibrate_field.antenna = antenna
        calibrate_field.stations = stations
        calibrate_dd = CalibrateDD(field=calibrate_field, index=1)

        baselines = calibrate_dd.get_baselines_core()
        assert baselines == expected

    @pytest.mark.parametrize(
        "antenna,stations,expected",
        [
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
        ],
    )
    def test_get_superterp_stations(self, calibrate_field, antenna, stations, expected):
        calibrate_field.antenna = antenna
        calibrate_field.stations = stations
        calibrate_dd = CalibrateDD(field=calibrate_field, index=1)
        assert calibrate_dd.get_superterp_stations() == expected

    @pytest.mark.parametrize(
        "antenna,include_remote,stations,expected",
        [
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
        ],
    )
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

    # basic case for both di and dd
    @pytest.mark.parametrize("scenario", ["dd", "di"])
    def test_set_input_parameters(self, calibrate_field, scenario):
        """
        Test that set_input_parameters() populates input_parms with the correct
        values for all keys, for both CalibrateDD and CalibrateDI.
        """
        is_dd = scenario == "dd"
        calibrate = (
            CalibrateDD(field=calibrate_field, index=1)
            if is_dd
            else CalibrateDI(field=calibrate_field, index=1)
        )

        calibrate.set_input_parameters()
        params = calibrate.input_parms
        f = calibrate_field
        obs_param = f.get_obs_parameters.return_value  # [1]

        # Common to both DD and DI
        assert params["max_normalization_delta"] == f.max_normalization_delta
        assert params["llssolver"] == f.llssolver
        assert params["maxiter"] == f.maxiter
        assert params["propagatesolutions"] == f.propagatesolutions
        assert params["solveralgorithm"] == f.solveralgorithm
        assert params["stepsize"] == f.stepsize
        assert params["stepsigma"] == f.stepsigma
        assert params["tolerance"] == f.tolerance
        assert params["uvlambdamin"] == f.solve_min_uv_lambda
        assert params["solverlbfgs_dof"] == f.solverlbfgs_dof
        assert params["solverlbfgs_iter"] == f.solverlbfgs_iter
        assert params["solverlbfgs_minibatches"] == f.solverlbfgs_minibatches
        assert params["correctfreqsmearing"] == f.correct_smearing_in_calibration
        assert params["correcttimesmearing"] == f.correct_smearing_in_calibration
        assert params["max_threads"] == f.parset["cluster_specific"]["max_threads"]

        if is_dd:  # speific to CalibrateDD
            # CWL-wrapped inputs
            assert params["timechunk_filename"] == CWLDir(obs_param).to_json()
            assert (
                params["calibration_skymodel_file"]
                == CWLFile(f.calibration_skymodel_file).to_json()
            )

            # get_obs_parameters pass-throughs
            assert params["starttime"] == obs_param
            assert params["ntimes"] == obs_param
            assert params["solint_fast_timestep"] == obs_param
            assert params["solint_medium_timestep"] == obs_param
            assert params["solint_slow_timestep"] == obs_param
            assert params["solint_fast_freqstep"] == obs_param
            assert params["solint_medium_freqstep"] == obs_param
            assert params["solint_slow_freqstep"] == obs_param
            assert params["fast_solutions_per_direction"] == obs_param
            assert params["medium_solutions_per_direction"] == obs_param
            assert params["slow_solutions_per_direction"] == obs_param
            assert params["bda_maxinterval"] == obs_param
            assert params["bda_minchannels"] == obs_param
            assert params["fast_smoothness_dd_factors"] == obs_param
            assert params["medium_smoothness_dd_factors"] == obs_param
            assert params["slow_smoothness_dd_factors"] == obs_param
            assert params["fast_smoothnessreffrequency"] == obs_param
            assert params["medium_smoothnessreffrequency"] == obs_param

            # Field pass-throughs
            assert params["data_colname"] == f.data_colname
            assert params["ra_mid"] == f.ra
            assert params["dec_mid"] == f.dec
            assert params["phase_center_ra"] == f.ra
            assert params["phase_center_dec"] == f.dec
            assert params["calibrator_patch_names"] == f.calibrator_patch_names
            assert params["calibrator_fluxes"] == f.calibrator_fluxes
            assert params["fast_smoothnessrefdistance"] == f.fast_smoothnessrefdistance
            assert params["medium_smoothnessrefdistance"] == f.medium_smoothnessrefdistance
            assert params["onebeamperpatch"] == f.onebeamperpatch
            assert params["parallelbaselines"] == f.parallelbaselines
            assert params["sagecalpredict"] == f.sagecalpredict
            assert params["fast_datause"] == f.fast_datause
            assert params["medium_datause"] == f.medium_datause
            assert params["slow_datause"] == f.slow_datause
            assert params["bda_timebase"] == f.calibrate_bda_timebase
            assert params["bda_frequencybase"] == f.calibrate_bda_frequencybase

            # String-converted values
            assert params["sector_bounds_deg"] == str(f.sector_bounds_deg)
            assert params["sector_bounds_mid_deg"] == str(f.sector_bounds_mid_deg)
            assert params["scale_normalization_delta"] == str(f.scale_normalization_delta)

            # Computed smoothness constraints: field.X / min(obs_params)
            assert params["fast_smoothnessconstraint"] == f.fast_smoothnessconstraint / np.min(
                obs_param
            )
            assert params["medium_smoothnessconstraint"] == f.medium_smoothnessconstraint / np.min(
                obs_param
            )
            assert params["slow_smoothnessconstraint"] == f.slow_smoothnessconstraint / np.min(
                obs_param
            )

            # Antenna constraints
            assert params["fast_antennaconstraint"] == "[]"
            assert params["medium_antennaconstraint"] == "[]"
            assert params["slow_antennaconstraint"] == "[]"
            assert params["idgcal_antennaconstraint"] == "[]"

            # Output h5parm filenames (derived from ntimechunks)
            assert params["output_fast_h5parm"] == [
                f"fast_phase_{i}.h5parm" for i in range(f.ntimechunks)
            ]
            assert params["collected_fast_h5parm"] == "fast_phases.h5parm"
            assert params["output_medium1_h5parm"] == [
                f"medium1_phase_{i}.h5parm" for i in range(f.ntimechunks)
            ]
            assert params["output_medium2_h5parm"] == [
                f"medium2_phase_{i}.h5parm" for i in range(f.ntimechunks)
            ]
            assert params["collected_medium1_h5parm"] == "medium1_phases.h5parm"
            assert params["collected_medium2_h5parm"] == "medium2_phases.h5parm"
            assert params["combined_fast_medium1_h5parm"] == "combined_fast_medium1_phases.h5parm"
            assert (
                params["combined_fast_medium1_medium2_h5parm"]
                == "combined_fast_medium1_medium2_phases.h5parm"
            )
            assert params["output_slow_h5parm"] == [
                f"slow_gain_{i}.h5parm" for i in range(f.ntimechunks)
            ]
            assert params["collected_slow_h5parm"] == "slow_gains.h5parm"
            assert params["output_idgcal_h5parm"] == [
                f"idgcal_{i}.h5parm" for i in range(f.ntimechunks)
            ]
            assert params["combined_h5parms"] == "combined_solutions.h5"

            # Model image parameters
            assert params["model_image_root"] == "calibration_model"
            assert params["facet_region_file"] == "field_facets_ds9.reg"
            # Single source sky model has no SpectralIndex column, so num_spectral_terms == 1
            assert params["num_spectral_terms"] == 1
            # facet_region_width is derived from model_image_imsize and model_image_cellsize
            assert params["facet_region_width_ra"] == params["facet_region_width_dec"]
            assert params["facet_region_width_ra"] == pytest.approx(
                max(params["model_image_imsize"]) * params["model_image_cellsize"] * 1.2
            )

            # Conditional defaults
            assert params["solution_combine_mode"] == "p1p2a2_scalar"
            assert params["normalize_h5parm"] is None
            assert params["ddecal_applycal_steps"] is None
            assert params["applycal_steps"] is None
            assert params["fast_initialsolutions_h5parm"] is None
            assert params["medium1_initialsolutions_h5parm"] is None
            assert params["medium2_initialsolutions_h5parm"] is None
            assert params["slow_initialsolutions_h5parm"] is None

        else:  # Specific to DI
            # CWL-wrapped inputs
            assert params["timechunk_filename_fulljones"] == CWLDir(obs_param).to_json()

            # get_obs_parameters pass-throughs
            assert params["data_colname"] == "DATA"
            assert params["starttime_fulljones"] == obs_param
            assert params["ntimes_fulljones"] == obs_param
            assert params["solint_fulljones_timestep"] == obs_param
            assert params["solint_fulljones_freqstep"] == obs_param

            # Output h5parm filenames
            assert calibrate.collected_h5parm_fulljones == "fulljones_gains.h5"
            assert params["collected_h5parm_fulljones"] == "fulljones_gains.h5"
            assert params["output_h5parm_fulljones"] == [
                f"fulljones_gain_{i}.h5parm" for i in range(f.ntimechunks)
            ]

            # Field pass-throughs
            assert params["smoothnessconstraint_fulljones"] == f.smoothnessconstraint_fulljones

    # special cases for dd
    @pytest.mark.parametrize(
        "bda_time, bda_freq, slowgain, expected_dp3_steps",
        [
            (0, 0, False, ["solve1", "solve2"]),
            (1, 1, False, ["avg", "solve1", "solve2", "null"]),
            (1, 1, True, ["avg", "solve1", "solve2", "solve3", "solve4", "null"]),
        ],
    )
    def test_set_input_parameters_dd_bda_cases(
        self, calibrate_field, bda_time, bda_freq, slowgain, expected_dp3_steps
    ):
        """
        Test the effect of BDA and slowgain settings on the dp3 steps.
        """
        calibrate_field.calibrate_bda_timebase = bda_time
        calibrate_field.calibrate_bda_frequencybase = bda_freq
        calibrate_field.do_slowgain_solve = slowgain

        calibrate_dd = CalibrateDD(field=calibrate_field, index=1)
        calibrate_dd.set_input_parameters()
        dp3_steps = parse_dp3(calibrate_dd.input_parms["dp3_steps"])

        assert dp3_steps == expected_dp3_steps

    @pytest.mark.parametrize(
        "normalize, expected_prefix, expect_applycal",
        [
            (False, ["predict", "applybeam"], False),
            (True, ["predict", "applybeam", "applycal"], True),
        ],
    )
    def test_set_input_parameters_dd_ibp_cases(
        self, calibrate_field, tmp_path, normalize, expected_prefix, expect_applycal
    ):
        """
        DD: Test the effect of image-based predict and normalization settings on the dp3 steps and applycal steps.
        """
        calibrate_field.use_image_based_predict = True
        calibrate_field.apply_normalizations = normalize
        if normalize:
            calibrate_field.normalize_h5parm = str(tmp_path / "normalize.h5parm")

        calibrate_dd = CalibrateDD(field=calibrate_field, index=1)
        calibrate_dd.set_input_parameters()
        params = calibrate_dd.input_parms
        dp3 = parse_dp3(params["dp3_steps"])

        assert dp3[: len(expected_prefix)] == expected_prefix
        if expect_applycal:
            assert params["ddecal_applycal_steps"] == "[normalization]"
            assert params["applycal_steps"] == "[normalization]"
        else:
            assert params["ddecal_applycal_steps"] is None

    @pytest.mark.parametrize(
        "diagonal_flag, expected_mode",
        [
            (True, "p1p2a2_diagonal"),
            (False, "p1p2a2_scalar"),
        ],
    )
    def test_set_input_parameters_dd_solution_combine_mode(
        self, calibrate_field, diagonal_flag, expected_mode
    ):
        """
        DD: Test the effect of diagonal solutions on the solution_combine_mode.
        """
        calibrate_field.apply_diagonal_solutions = diagonal_flag

        calibrate_dd = CalibrateDD(field=calibrate_field, index=1)
        calibrate_dd.set_input_parameters()

        assert calibrate_dd.input_parms["solution_combine_mode"] == expected_mode

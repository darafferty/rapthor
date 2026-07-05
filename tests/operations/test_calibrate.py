"""
Test cases for the calibration operation modules.
"""

from pathlib import Path

import pytest

import rapthor
from rapthor.execution.calibrate.builders import calibrate_payload_from_inputs
from rapthor.execution.calibrate.solves import build_calibrate_chunk_command
from rapthor.lib.field import Field as RapthorField
from rapthor.lib.operation import DIR as OPERATION_DIR
from rapthor.operations.calibrate.base import Calibrate
from rapthor.operations.calibrate.plan import (
    build_calibration_core_baseline_selection,
    build_calibration_core_stations,
    build_calibration_dp3_steps,
    build_calibration_preapply_steps,
    build_calibration_solve_plan,
    build_calibration_solve_slot_inputs,
    build_calibration_superterp_stations,
    requested_calibration_solves,
)

CALIBRATE_COMMON_INPUT_KEYS = {
    "timechunk_filename",
    "data_colname",
    "modeldatacolumn",
    "calibration_skymodel_file",
    "starttime",
    "ntimes",
    "has_slow_gain_solve",
    "bda_maxinterval",
    "bda_minchannels",
    "bda_timebase",
    "bda_frequencybase",
    "onebeamperpatch",
    "parallelbaselines",
    "sagecalpredict",
    "normalize_h5parm",
    "calibration_applycal_steps",
    "applycal_steps",
    "applycal_h5parm",
    "fulljones_h5parm",
    "solve1_initialsolutions_h5parm",
    "solve2_initialsolutions_h5parm",
    "solve3_initialsolutions_h5parm",
    "solve4_initialsolutions_h5parm",
    "combined_phase_1_2_h5parm",
    "combined_phase_1_2_3_h5parm",
    "combined_h5parms",
    "sector_bounds_deg",
    "sector_bounds_mid_deg",
    "calibrator_patch_names",
    "solve_directions",
    "calibrator_fluxes",
    "dp3_steps",
    "max_normalization_delta",
    "scale_normalization_delta",
    "phase_center_ra",
    "phase_center_dec",
    "llssolver",
    "maxiter",
    "propagatesolutions",
    "solveralgorithm",
    "stepsize",
    "stepsigma",
    "tolerance",
    "uvlambdamin",
    "solverlbfgs_dof",
    "solverlbfgs_iter",
    "solverlbfgs_minibatches",
    "solve1_antennaconstraint",
    "solve2_antennaconstraint",
    "solve3_antennaconstraint",
    "solve4_antennaconstraint",
    "solve1_smoothnessreffrequency",
    "solve2_smoothnessreffrequency",
    "solve4_smoothnessreffrequency",
    "solution_combine_mode",
    "correctfreqsmearing",
    "correcttimesmearing",
    "max_threads",
}

CALIBRATE_DD_INPUT_KEYS = {
    "generate_screens",
    "use_wsclean_predict",
    "solint_fast_timestep",
    "solint_medium_timestep",
    "solint_slow_timestep",
    "solint_fast_freqstep",
    "solint_medium_freqstep",
    "solint_slow_freqstep",
    "model_image_root",
    "model_image_ra_dec",
    "model_image_imsize",
    "model_image_cellsize",
    "model_image_frequency_bandwidth",
    "num_spectral_terms",
    "ra_mid",
    "dec_mid",
    "facet_region_width_ra",
    "facet_region_width_dec",
    "facet_region_file",
    "predict_facet_region_file",
    "solve1_smoothnessrefdistance",
    "solve2_smoothnessrefdistance",
    "solve4_smoothnessrefdistance",
    "bda_maxinterval",
    "bda_minchannels",
    "bda_timebase",
    "bda_frequencybase",
    "output_idgcal_h5parm",
    "idgcal_antennaconstraint",
}

CALIBRATE_DI_INPUT_KEYS = {
    "solint_fast_timestep",
    "solint_fast_freqstep",
    "solint_slow_timestep",
    "solint_slow_freqstep",
    "smoothnessconstraint_fulljones",
    "solve3_smoothnessreffrequency",
    "solve1_smoothnessrefdistance",
    "solve2_smoothnessrefdistance",
    "solve4_smoothnessrefdistance",
}


def _calibrate_solve_input_keys():
    keys = set()
    for slot in range(1, 5):
        keys.update(
            {
                f"output_solve{slot}_h5parm",
                f"collected_solve{slot}_h5parm",
                f"solve{slot}_mode",
                f"solint_solve{slot}_timestep",
                f"solint_solve{slot}_freqstep",
                f"solve{slot}_datause",
                f"solve{slot}_solutions_per_direction",
                f"solve{slot}_smoothness_dd_factors",
                f"solve{slot}_smoothnessconstraint",
            }
        )
    return keys


@pytest.fixture
def calibrate_field(operation_parset, mocker, single_source_sky_model):
    """Create a mock field object for testing a Calibrate operation."""

    class Field:
        solution_cycle_number = RapthorField.solution_cycle_number

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
            self.apply_normalizations = False
            self.generate_screens = False
            self.calibration_strategy = {"dd": ["fast_phase", "medium_phase"], "di": ["full_jones"]}
            self._calibration_strategy_defaulted = False
            self.use_wsclean_predict = False
            self.normalize_h5parm = None
            self.calibrate_bda_timebase = 0
            self.calibrate_bda_frequencybase = 0
            self.h5parm_filename = None
            self.fulljones_h5parm_filename = None
            self.fast_phases_h5parm_filename = None
            self.medium1_phases_h5parm_filename = None
            self.medium2_phases_h5parm_filename = None
            self.slow_gains_h5parm_filename = None
            self.dd_h5parm_filename = None
            self.di_h5parm_filename = None
            self.di_fast_phases_h5parm_filename = None
            self.di_medium1_phases_h5parm_filename = None
            self.di_medium2_phases_h5parm_filename = None
            self.di_slow_gains_h5parm_filename = None

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
        "mode, expected_name, index",
        [
            ("dd", "calibrate", 1),
            ("di", "calibrate_di", 2),
        ],
    )
    def test_init_sets_name_and_mode(self, calibrate_field, mode, expected_name, index):
        calibrate = Calibrate(mode=mode, field=calibrate_field, index=index)
        assert calibrate.mode == mode
        assert calibrate.name == f"{expected_name}_{index}"

    def test_init_raises_on_invalid_mode(self, calibrate_field):
        with pytest.raises(ValueError, match="Only di and dd mode are supported"):
            Calibrate(mode="invalid", field=calibrate_field, index=1)

    @pytest.mark.parametrize(
        "mode, solve, batch_system, generate_screens, use_image_based_predict",
        [
            ("dd", "fast_only", "slurm", False, False),
            ("dd", "fast_only", "slurm", True, False),
            ("dd", "with_slowgain", "some_other_batch_system", False, True),
            ("dd", "with_slowgain", "some_other_batch_system", True, True),
            ("di", "fulljones", "slurm", "don't", "care"),
            ("di", "fulljones", "some_other_batch_system", "don't", "care"),
        ],
    )
    def test_set_parset_parameters(
        self, calibrate_field, mode, solve, batch_system, generate_screens, use_image_based_predict
    ):
        max_cores = 42

        # Setup field object
        calibrate_field.generate_screens = generate_screens
        calibrate_field.use_image_based_predict = use_image_based_predict
        calibrate_field.parset["cluster_specific"]["batch_system"] = batch_system
        calibrate_field.parset["cluster_specific"]["max_cores"] = max_cores

        calibrate = Calibrate(mode=mode, field=calibrate_field, index=1 if mode == "dd" else 2)

        # Act
        calibrate.set_parset_parameters()

        # Assert
        rapthor_pipeline_path = Path(rapthor.__file__).parent / "pipeline"
        assert calibrate.parset_parms["rapthor_pipeline_dir"] == str(rapthor_pipeline_path)

        expected_max_cores = None if batch_system == "slurm" else max_cores
        assert calibrate.parset_parms["max_cores"] == expected_max_cores

        if mode == "dd":  # CalibrateDD sets some extra parameters.
            expected_use_image_based_predict = generate_screens or use_image_based_predict
            assert calibrate.use_image_based_predict is expected_use_image_based_predict
            assert (
                calibrate.parset_parms["use_image_based_predict"]
                is expected_use_image_based_predict
            )
            assert calibrate.parset_parms["generate_screens"] is generate_screens
            assert calibrate.parset_parms["use_wsclean_predict"] is False

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
    def test_build_calibration_core_baseline_selection(self, antenna, stations, expected):
        assert build_calibration_core_baseline_selection(antenna, stations) == expected

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
    def test_build_calibration_superterp_stations(self, antenna, stations, expected):
        assert build_calibration_superterp_stations(antenna, stations) == expected

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
        calibrate_dd = Calibrate("dd", field=calibrate_field, index=1)
        result = calibrate_dd._get_core_stations(include_nearest_remote=include_remote)
        assert result == expected
        assert (
            build_calibration_core_stations(
                antenna,
                stations,
                include_nearest_remote=include_remote,
            )
            == expected
        )

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
        calibrate_dd = Calibrate("dd", field, index=cycle)
        frequency_bandwidth, center_coords, size, cellsize = (
            calibrate_dd._get_model_image_parameters()
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

    @pytest.mark.parametrize(
        "mode, solve", [("dd", "fast_only"), ("dd", "with_slowgain"), ("di", "fulljones")]
    )
    def test_finalize(self, mocker, calibrate_field, tmp_path, mode, solve):
        field = calibrate_field
        index = 2 if mode == "dd" else 4
        with_slow = solve == "with_slowgain"

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
        name = "calibrate_2" if mode == "dd" else "calibrate_di_4"
        solutions_path = workdir_path / "solutions" / name
        solutions_path.mkdir(parents=True)

        # Create an existing solutions file. finalize() should remove it.
        h5parm_filename = "field-solutions.h5" if mode == "dd" else "fulljones-solutions.h5"
        h5parm_path = solutions_path / h5parm_filename
        h5parm_path.touch()

        pipelines_path = workdir_path / "pipelines" / name
        plots_path = workdir_path / "plots" / name
        finalize_prepare_plots(pipelines_path, plots_path)

        # Setup the object itself
        if mode == "dd":
            field.calibration_strategy = (
                {"dd": ["fast_phase", "medium_phase", "slow_gains", "medium_phase"], "di": []}
                if with_slow
                else {"dd": ["fast_phase"], "di": []}
            )
        else:
            field.calibration_strategy = {"di": ["full_jones"]}
        field._calibration_strategy_defaulted = False

        calibrate = Calibrate(mode=mode, field=field, index=index)

        if mode == "dd":
            calibrate.combined_h5parms = "combined.test.h5"
            calibrate.fast_h5parm = "fast.test.h5"
            if with_slow:
                calibrate.slow_h5parm = "slow.test.h5"
                calibrate.medium1_h5parm = "medium1.test.h5"
                calibrate.medium2_h5parm = "medium2.test.h5"
        else:
            calibrate.fulljones_solutions_h5parm = "collected_fulljones.h5"

        # Ignore os.makedirs calls from the base Operation class constructor.
        mock_makedirs.reset_mock()

        # Act
        calibrate.finalize()

        # Assert
        if mode == "dd":
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
        elif solve == "fast_only":
            solution_src_dst_list = [
                ("fast.test.h5", h5parm_filename),
                ("fast.test.h5", "field-solutions-fast-phase.h5"),
            ]
        else:  # di fulljones scenario
            solution_src_dst_list = [
                ("collected_fulljones.h5", h5parm_filename),
            ]

        for src, dst in solution_src_dst_list:
            mock_copy.assert_any_call(str(pipelines_path / src), str(solutions_path / dst))

        field.scan_h5parms.assert_called_once()

        if mode == "dd":
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

    def test_finalize_di_scalar_solutions(self, mocker, calibrate_field, tmp_path):
        field = calibrate_field
        field.calibration_strategy = {"di": ["fast_phase", "medium_phase"]}
        field._calibration_strategy_defaulted = False

        flagged_fraction = 0.042
        mock_flagged_fraction = mocker.patch(
            "rapthor.lib.miscellaneous.get_flagged_solution_fraction",
            return_value=flagged_fraction,
        )
        mock_makedirs = mocker.patch("os.makedirs")
        mock_remove = mocker.patch("os.remove")
        mock_copy = mocker.patch("shutil.copy")

        workdir_path = tmp_path / "working"
        solutions_path = workdir_path / "solutions" / "calibrate_di_1"
        solutions_path.mkdir(parents=True)
        (solutions_path / "di-solutions.h5").touch()

        pipelines_path = workdir_path / "pipelines" / "calibrate_di_1"
        plots_path = workdir_path / "plots" / "calibrate_di_1"
        finalize_prepare_plots(pipelines_path, plots_path)

        calibrate = Calibrate(mode="di", field=field, index=1)
        mock_makedirs.reset_mock()

        calibrate.finalize()

        assert field.h5parm_filename == str(solutions_path / "di-solutions.h5")
        assert field.di_h5parm_filename == str(solutions_path / "di-solutions.h5")
        assert field.di_fast_phases_h5parm_filename == str(
            solutions_path / "di-solutions-fast-phase.h5"
        )
        assert field.di_medium1_phases_h5parm_filename == str(
            solutions_path / "di-solutions-medium1-phase.h5"
        )
        assert field.fulljones_h5parm_filename is None

        check_makedirs(mock_makedirs, solutions_path, plots_path)
        mock_remove.assert_any_call(str(solutions_path / "di-solutions.h5"))

        solution_src_dst_list = [
            ("combined_solve1_solve2_di.h5parm", "di-solutions.h5"),
            ("fast_phases_di.h5parm", "di-solutions-fast-phase.h5"),
            ("medium1_phases_di.h5parm", "di-solutions-medium1-phase.h5"),
        ]
        for src, dst in solution_src_dst_list:
            mock_copy.assert_any_call(str(pipelines_path / src), str(solutions_path / dst))

        field.scan_h5parms.assert_called_once()
        mock_flagged_fraction.assert_called_once_with(str(solutions_path / "di-solutions.h5"))

        mock_remove.assert_any_call(str(plots_path / "plot2.png"))
        mock_copy.assert_any_call(str(pipelines_path / "plot1.png"), str(plots_path / "plot1.png"))
        mock_copy.assert_any_call(str(pipelines_path / "plot2.png"), str(plots_path / "plot2.png"))

        assert mock_remove.call_count == 2
        assert mock_copy.call_count == len(solution_src_dst_list) + 2

        # finalize() should create a .done file (via the base Operation class).
        assert (pipelines_path / ".done").exists()

    def test_finalize_di_scalar_and_fulljones_solutions(self, mocker, calibrate_field, tmp_path):
        field = calibrate_field
        field.calibration_strategy = {
            "di": ["fast_phase", "medium_phase", "slow_gains", "full_jones"]
        }
        field._calibration_strategy_defaulted = False

        flagged_fraction = 0.042
        mock_flagged_fraction = mocker.patch(
            "rapthor.lib.miscellaneous.get_flagged_solution_fraction",
            return_value=flagged_fraction,
        )
        mock_makedirs = mocker.patch("os.makedirs")
        mock_remove = mocker.patch("os.remove")
        mock_copy = mocker.patch("shutil.copy")

        workdir_path = tmp_path / "working"
        solutions_path = workdir_path / "solutions" / "calibrate_di_1"
        solutions_path.mkdir(parents=True)
        (solutions_path / "di-solutions.h5").touch()
        (solutions_path / "fulljones-solutions.h5").touch()

        pipelines_path = workdir_path / "pipelines" / "calibrate_di_1"
        plots_path = workdir_path / "plots" / "calibrate_di_1"
        finalize_prepare_plots(pipelines_path, plots_path)

        calibrate = Calibrate(mode="di", field=field, index=1)
        mock_makedirs.reset_mock()

        calibrate.finalize()

        assert field.h5parm_filename == str(solutions_path / "di-solutions.h5")
        assert field.di_h5parm_filename == str(solutions_path / "di-solutions.h5")
        assert field.di_fast_phases_h5parm_filename == str(
            solutions_path / "di-solutions-fast-phase.h5"
        )
        assert field.di_medium1_phases_h5parm_filename == str(
            solutions_path / "di-solutions-medium1-phase.h5"
        )
        assert field.di_slow_gains_h5parm_filename == str(
            solutions_path / "di-solutions-slow-gain.h5"
        )
        assert field.fulljones_h5parm_filename == str(solutions_path / "fulljones-solutions.h5")

        check_makedirs(mock_makedirs, solutions_path, plots_path)
        mock_remove.assert_any_call(str(solutions_path / "di-solutions.h5"))
        mock_remove.assert_any_call(str(solutions_path / "fulljones-solutions.h5"))

        solution_src_dst_list = [
            ("fulljones_solutions.h5", "fulljones-solutions.h5"),
            ("combined_di_solutions.h5parm", "di-solutions.h5"),
            ("fast_phases_di.h5parm", "di-solutions-fast-phase.h5"),
            ("medium1_phases_di.h5parm", "di-solutions-medium1-phase.h5"),
            ("slow_gains_di.h5parm", "di-solutions-slow-gain.h5"),
        ]
        for src, dst in solution_src_dst_list:
            mock_copy.assert_any_call(str(pipelines_path / src), str(solutions_path / dst))

        field.scan_h5parms.assert_called_once()
        mock_flagged_fraction.assert_called_once_with(
            str(solutions_path / "fulljones-solutions.h5")
        )

        mock_remove.assert_any_call(str(plots_path / "plot2.png"))
        mock_copy.assert_any_call(str(pipelines_path / "plot1.png"), str(plots_path / "plot1.png"))
        mock_copy.assert_any_call(str(pipelines_path / "plot2.png"), str(plots_path / "plot2.png"))

        assert mock_remove.call_count == 3
        assert mock_copy.call_count == len(solution_src_dst_list) + 2

        # finalize() should create a .done file (via the base Operation class).
        assert (pipelines_path / ".done").exists()

    def test_finalize_dd_explicit_single_slow_gains_solution(
        self, mocker, calibrate_field, tmp_path
    ):
        field = calibrate_field
        field.calibration_strategy = {"dd": ["slow_gains"]}
        field._calibration_strategy_defaulted = False

        flagged_fraction = 0.042
        mocker.patch(
            "rapthor.lib.miscellaneous.get_flagged_solution_fraction",
            return_value=flagged_fraction,
        )
        mock_makedirs = mocker.patch("os.makedirs")
        mock_remove = mocker.patch("os.remove")
        mock_copy = mocker.patch("shutil.copy")

        workdir_path = tmp_path / "working"
        solutions_path = workdir_path / "solutions" / "calibrate_1"
        solutions_path.mkdir(parents=True)
        (solutions_path / "field-solutions.h5").touch()

        pipelines_path = workdir_path / "pipelines" / "calibrate_1"
        plots_path = workdir_path / "plots" / "calibrate_1"
        finalize_prepare_plots(pipelines_path, plots_path)

        calibrate = Calibrate(mode="dd", field=field, index=1)
        mock_makedirs.reset_mock()

        calibrate.finalize()

        assert field.h5parm_filename == str(solutions_path / "field-solutions.h5")
        assert field.dd_h5parm_filename == str(solutions_path / "field-solutions.h5")
        assert field.slow_gains_h5parm_filename == str(
            solutions_path / "field-solutions-slow-gain.h5"
        )

        check_makedirs(mock_makedirs, solutions_path, plots_path)
        mock_remove.assert_any_call(str(solutions_path / "field-solutions.h5"))
        mock_copy.assert_any_call(
            str(pipelines_path / "slow_gains.h5parm"),
            str(solutions_path / "field-solutions.h5"),
        )
        mock_copy.assert_any_call(
            str(pipelines_path / "slow_gains.h5parm"),
            str(solutions_path / "field-solutions-slow-gain.h5"),
        )

        field.scan_h5parms.assert_called_once()
        assert field.calibration_diagnostics == [
            {
                "cycle_number": 1,
                "solution_flagged_fraction": flagged_fraction,
            }
        ]

        assert mock_remove.call_count == 2
        assert mock_copy.call_count == 4

        # finalize() should create a .done file (via the base Operation class).
        assert (pipelines_path / ".done").exists()

    @pytest.mark.parametrize(
        "mode, generate_screens, use_image_based_predict, calibration_strategy, has_slow_gain_solve",
        [
            ("dd", False, False, {"dd": ["fast_phase", "medium_phase"], "di": []}, False),
            ("dd", True, False, {"dd": ["fast_phase", "medium_phase"], "di": []}, False),
            ("dd", False, True, {"dd": ["fast_phase", "medium_phase"], "di": []}, False),
            (
                "dd",
                False,
                False,
                {"dd": ["fast_phase", "medium_phase", "slow_gains", "medium_phase"], "di": []},
                True,
            ),
            (
                "dd",
                True,
                False,
                {"dd": ["fast_phase", "medium_phase", "slow_gains", "medium_phase"], "di": []},
                True,
            ),
            ("di", False, False, {"di": ["full_jones"]}, False),
        ],
    )
    def test_set_input_parameters(
        self,
        mode,
        calibrate_field,
        generate_screens,
        use_image_based_predict,
        calibration_strategy,
        has_slow_gain_solve,
    ):
        """
        Test that set_input_parameters() provides the inputs needed by the
        calibration flow for representative DD flag combinations and DI mode.
        """
        field = calibrate_field
        field.generate_screens = generate_screens
        field.use_image_based_predict = use_image_based_predict
        field.calibration_strategy = calibration_strategy
        field._calibration_strategy_defaulted = False

        calibrate = Calibrate(mode=mode, field=calibrate_field, index=1 if mode == "dd" else 2)
        calibrate.set_input_parameters()

        expected_input_keys = CALIBRATE_COMMON_INPUT_KEYS | _calibrate_solve_input_keys()
        if mode == "dd":
            expected_input_keys |= CALIBRATE_DD_INPUT_KEYS
        else:
            expected_input_keys |= CALIBRATE_DI_INPUT_KEYS

        input_parms_keys = set(calibrate.input_parms.keys())
        assert expected_input_keys.issubset(input_parms_keys), (
            f"input_parms is missing flow inputs: {expected_input_keys - input_parms_keys}"
        )
        assert calibrate.input_parms["has_slow_gain_solve"] is has_slow_gain_solve
        if mode == "dd":
            assert calibrate.input_parms["generate_screens"] is generate_screens
        else:
            assert "generate_screens" not in input_parms_keys

    # special cases for dd
    @pytest.mark.parametrize(
        "bda_time, bda_freq, calibration_strategy, expected_dp3_steps",
        [
            (0, 0, {"dd": ["fast_phase", "medium_phase"], "di": []}, ["solve1", "solve2"]),
            (
                1,
                1,
                {"dd": ["fast_phase", "medium_phase"], "di": []},
                ["avg", "solve1", "solve2", "null"],
            ),
            (
                1,
                1,
                {"dd": ["fast_phase", "medium_phase", "slow_gains", "medium_phase"], "di": []},
                ["avg", "solve1", "solve2", "solve3", "solve4", "null"],
            ),
        ],
    )
    def test_set_input_parameters_dd_bda_cases(
        self, calibrate_field, bda_time, bda_freq, calibration_strategy, expected_dp3_steps
    ):
        """
        Test the effect of BDA and solve strategy on the DP3 steps.
        """
        calibrate_field.calibrate_bda_timebase = bda_time
        calibrate_field.calibrate_bda_frequencybase = bda_freq
        calibrate_field.calibration_strategy = calibration_strategy
        calibrate_field._calibration_strategy_defaulted = False

        calibrate_dd = Calibrate("dd", field=calibrate_field, index=1)
        calibrate_dd.set_input_parameters()
        dp3_steps = parse_dp3(calibrate_dd.input_parms["dp3_steps"])

        assert dp3_steps == expected_dp3_steps
        assert (
            build_calibration_dp3_steps(
                bda_time,
                bda_freq,
                all_channels_regular=True,
                use_image_based_predict=False,
                has_slow_gain_solve="slow_gains" in calibration_strategy["dd"],
            )
            == expected_dp3_steps
        )

    @pytest.mark.parametrize(
        "preapply_solutions, expected_steps",
        [
            (False, ["predict", "applybeam", "solve1", "solve2"]),
            (True, ["predict", "applybeam", "applycal", "solve1", "solve2"]),
        ],
    )
    def test_build_calibration_dp3_steps_image_based_predict(
        self, preapply_solutions, expected_steps
    ):
        assert (
            build_calibration_dp3_steps(
                0,
                0,
                all_channels_regular=True,
                use_image_based_predict=True,
                has_slow_gain_solve=False,
                solve_steps=["solve1", "solve2"],
                preapply_solutions=preapply_solutions,
            )
            == expected_steps
        )

    def test_build_calibration_dp3_steps_wsclean_predict_skips_preprocessing(self):
        assert build_calibration_dp3_steps(
            10,
            10,
            all_channels_regular=True,
            use_image_based_predict=False,
            use_wsclean_predict=True,
            has_slow_gain_solve=False,
            solve_steps=["solve1", "solve2"],
            preapply_solutions=True,
        ) == ["solve1", "solve2"]

    @pytest.mark.parametrize(
        "strategy, apply_amplitudes, expected_steps",
        [
            (None, True, ["fastphase", "slowgain", "fulljones", "normalization"]),
            ({"di": ["fast_phase"]}, True, ["fastphase", "fulljones", "normalization"]),
            (None, False, ["fastphase", "fulljones", "normalization"]),
        ],
    )
    def test_build_calibration_preapply_steps(self, strategy, apply_amplitudes, expected_steps):
        assert (
            build_calibration_preapply_steps(
                "dd",
                has_di_h5parm=True,
                has_fulljones_h5parm=True,
                apply_amplitudes=apply_amplitudes,
                apply_normalizations=True,
                calibration_strategy=strategy,
            )
            == expected_steps
        )

    def test_build_calibration_solve_slot_inputs_for_scalar_phase_slot(self):
        assert build_calibration_solve_slot_inputs(
            2,
            "medium_phase",
            ntimechunks=2,
            datause="full",
            solutions_per_direction=[[1, 2], [3, 4]],
            smoothness_dd_factors=[[2.0, 4.0], [3.0, 5.0]],
            smoothnessconstraint=12.0,
            antenna_constraint="[[CS001HBA0,CS002HBA0]]",
            include_smoothnessreffrequency=True,
            smoothnessreffrequency=[150.0, 151.0],
            include_smoothnessrefdistance=True,
            smoothnessrefdistance=2500.0,
        ) == {
            "solve2_datause": "full",
            "solve2_solutions_per_direction": [[1, 2], [3, 4]],
            "solve2_smoothness_dd_factors": [[2.0, 4.0], [3.0, 5.0]],
            "solve2_smoothnessconstraint": 6.0,
            "solve2_antennaconstraint": "[[CS001HBA0,CS002HBA0]]",
            "solve2_smoothnessreffrequency": [150.0, 151.0],
            "solve2_smoothnessrefdistance": 2500.0,
        }

    def test_build_calibration_solve_slot_inputs_for_slow_gain_slot(self):
        assert build_calibration_solve_slot_inputs(
            1,
            "slow_gains",
            ntimechunks=2,
            datause="dual",
            solutions_per_direction=[[1], [1]],
            smoothness_dd_factors=[[3.0], [4.0]],
            smoothnessconstraint=12.0,
            antenna_constraint="[[CS001HBA0]]",
            include_smoothnessreffrequency=True,
            include_smoothnessrefdistance=True,
        ) == {
            "solve1_datause": "dual",
            "solve1_solutions_per_direction": [[1], [1]],
            "solve1_smoothness_dd_factors": [[3.0], [4.0]],
            "solve1_smoothnessconstraint": 4.0,
            "solve1_antennaconstraint": "[]",
            "solve1_smoothnessreffrequency": [0, 0],
            "solve1_smoothnessrefdistance": None,
        }

    @pytest.mark.parametrize(
        "mode, strategy, defaulted, expected_plan",
        [
            (
                "dd",
                None,
                False,
                [
                    (
                        "fast_phase",
                        "fast",
                        "solve1",
                        "scalarphase",
                        "fast_phase",
                        "fast_phases.h5parm",
                    ),
                    (
                        "medium_phase",
                        "medium1",
                        "solve2",
                        "scalarphase",
                        "medium1_phase",
                        "medium1_phases.h5parm",
                    ),
                    (
                        "slow_gains",
                        "slow",
                        "solve3",
                        "diagonal",
                        "slow_gain",
                        "slow_gains.h5parm",
                    ),
                    (
                        "medium_phase",
                        "medium2",
                        "solve4",
                        "scalarphase",
                        "medium2_phase",
                        "medium2_phases.h5parm",
                    ),
                ],
            ),
            (
                "dd",
                {"dd": ["slow_gains"]},
                False,
                [
                    (
                        "slow_gains",
                        "slow",
                        "solve1",
                        "diagonal",
                        "slow_gain",
                        "slow_gains.h5parm",
                    )
                ],
            ),
            (
                "dd",
                {"dd": ["medium_phase", "fast_phase"]},
                False,
                [
                    (
                        "medium_phase",
                        "medium1",
                        "solve1",
                        "scalarphase",
                        "medium1_phase",
                        "medium1_phases.h5parm",
                    ),
                    (
                        "fast_phase",
                        "fast",
                        "solve2",
                        "scalarphase",
                        "fast_phase",
                        "fast_phases.h5parm",
                    ),
                ],
            ),
            (
                "di",
                {"di": ["fast_phase", "medium_phase", "slow_gains", "full_jones"]},
                False,
                [
                    (
                        "fast_phase",
                        "fast",
                        "solve1",
                        "scalarphase",
                        "fast_phase_di",
                        "fast_phases_di.h5parm",
                    ),
                    (
                        "medium_phase",
                        "medium1",
                        "solve2",
                        "scalarphase",
                        "medium1_phase_di",
                        "medium1_phases_di.h5parm",
                    ),
                    (
                        "slow_gains",
                        "slow",
                        "solve3",
                        "diagonal",
                        "slow_gains_di",
                        "slow_gains_di.h5parm",
                    ),
                    (
                        "full_jones",
                        "fulljones",
                        "solve4",
                        "fulljones",
                        "fulljones_gain",
                        "fulljones_solutions.h5",
                    ),
                ],
            ),
            (
                "di",
                {"di": ["slow_gains"]},
                False,
                [
                    (
                        "slow_gains",
                        "slow",
                        "solve1",
                        "diagonal",
                        "slow_gains_di",
                        "slow_gains_di.h5parm",
                    )
                ],
            ),
            (
                "di",
                None,
                False,
                [],
            ),
        ],
    )
    def test_build_solve_plan(self, calibrate_field, mode, strategy, defaulted, expected_plan):
        calibrate_field.calibration_strategy = strategy
        calibrate_field._calibration_strategy_defaulted = defaulted

        calibrate = Calibrate(mode, field=calibrate_field, index=1)
        plan = calibrate._build_solve_plan()

        assert [
            (
                solve.solve_type,
                solve.solution_label,
                solve.step,
                solve.mode,
                solve.output_prefix,
                solve.collected_h5parm,
            )
            for solve in plan
        ] == expected_plan

        requested_solves, helper_defaulted = requested_calibration_solves(
            mode,
            strategy,
            strategy_defaulted=defaulted,
        )
        helper_plan = build_calibration_solve_plan(
            mode,
            requested_solves,
            defaulted_strategy=helper_defaulted,
        )
        assert [
            (
                solve.solve_type,
                solve.solution_label,
                solve.step,
                solve.mode,
                solve.output_prefix,
                solve.collected_h5parm,
            )
            for solve in helper_plan
        ] == expected_plan

    def test_set_input_parameters_dd_uses_explicit_solve_strategy(self, calibrate_field):
        calibrate_field.calibration_strategy = {"dd": ["slow_gains"]}
        calibrate_field._calibration_strategy_defaulted = False

        calibrate = Calibrate("dd", field=calibrate_field, index=1)
        calibrate.set_input_parameters()

        assert parse_dp3(calibrate.input_parms["dp3_steps"]) == ["solve1"]
        assert calibrate.input_parms["solve1_mode"] == "diagonal"
        assert calibrate.input_parms["output_solve1_h5parm"] == [
            "slow_gain_0.h5parm",
            "slow_gain_1.h5parm",
        ]
        assert "solve1_smoothnessreffrequency" not in calibrate.input_parms
        assert "solve1_smoothnessrefdistance" not in calibrate.input_parms
        assert calibrate.input_parms["has_slow_gain_solve"] is True

    def test_set_input_parameters_dd_applies_antenna_constraints_to_fast_and_medium2(
        self, calibrate_field
    ):
        calibrate_field.stations = ["CS001HBA0", "CS002HBA0", "RS106HBA0"]
        calibrate_field.calibration_strategy = {
            "dd": ["fast_phase", "medium_phase", "slow_gains", "medium_phase"]
        }
        calibrate_field._calibration_strategy_defaulted = False

        calibrate = Calibrate("dd", field=calibrate_field, index=1)
        calibrate.set_input_parameters()

        antenna_constraint = "[[CS001HBA0,CS002HBA0,RS106HBA0]]"
        assert calibrate.input_parms["solve1_antennaconstraint"] == antenna_constraint
        assert calibrate.input_parms["solve2_antennaconstraint"] == "[]"
        assert calibrate.input_parms["solve3_antennaconstraint"] == "[]"
        assert calibrate.input_parms["solve4_antennaconstraint"] == antenna_constraint
        assert calibrate.input_parms["solve1_smoothnessreffrequency"] == [1]
        assert calibrate.input_parms["solve2_smoothnessreffrequency"] == [1]
        assert "solve3_smoothnessreffrequency" not in calibrate.input_parms
        assert calibrate.input_parms["solve4_smoothnessreffrequency"] == [1]

    def test_set_input_parameters_dd_constrains_medium_phase_after_slow_gains(
        self, calibrate_field
    ):
        calibrate_field.stations = ["CS001HBA0", "CS002HBA0", "RS106HBA0"]
        calibrate_field.calibration_strategy = {"dd": ["slow_gains", "medium_phase"]}
        calibrate_field._calibration_strategy_defaulted = False

        calibrate = Calibrate("dd", field=calibrate_field, index=1)
        calibrate.set_input_parameters()

        antenna_constraint = "[[CS001HBA0,CS002HBA0,RS106HBA0]]"
        assert parse_dp3(calibrate.input_parms["dp3_steps"]) == ["solve1", "solve2"]
        assert calibrate.input_parms["solve1_mode"] == "diagonal"
        assert calibrate.input_parms["solve1_antennaconstraint"] == "[]"
        assert calibrate.input_parms["output_solve2_h5parm"] == [
            "medium1_phase_0.h5parm",
            "medium1_phase_1.h5parm",
        ]
        assert calibrate.input_parms["solve2_mode"] == "scalarphase"
        assert calibrate.input_parms["solve2_antennaconstraint"] == antenna_constraint
        assert "solve1_smoothnessreffrequency" not in calibrate.input_parms
        assert "solve1_smoothnessrefdistance" not in calibrate.input_parms
        assert calibrate.input_parms["solve2_smoothnessreffrequency"] == [1]
        assert calibrate.input_parms["solve2_smoothnessrefdistance"] == 0.0

    def test_set_input_parameters_di_uses_explicit_solve_strategy(self, calibrate_field):
        calibrate_field.calibration_strategy = {"di": ["fast_phase", "medium_phase"]}
        calibrate_field._calibration_strategy_defaulted = False

        calibrate = Calibrate("di", field=calibrate_field, index=1)
        calibrate.set_input_parameters()

        assert parse_dp3(calibrate.input_parms["dp3_steps"]) == ["solve1", "solve2"]
        assert calibrate.input_parms["solve1_mode"] == "scalarphase"
        assert calibrate.input_parms["solve2_mode"] == "scalarphase"
        assert calibrate.input_parms["output_solve1_h5parm"] == [
            "fast_phase_di_0.h5parm",
            "fast_phase_di_1.h5parm",
        ]
        assert calibrate.input_parms["output_solve2_h5parm"] == [
            "medium1_phase_di_0.h5parm",
            "medium1_phase_di_1.h5parm",
        ]

    def test_set_input_parameters_dd_allows_current_cycle_solve_initial_solutions(
        self, calibrate_field, tmp_path
    ):
        calibrate_field.calibration_strategy = {"dd": ["fast_phase", "medium_phase", "slow_gains"]}
        calibrate_field._calibration_strategy_defaulted = False

        current_solution_dir = tmp_path / "solutions" / "calibrate_1"
        future_solution_dir = tmp_path / "solutions" / "calibrate_2"
        current_solution_dir.mkdir(parents=True)
        future_solution_dir.mkdir(parents=True)

        current_fast = current_solution_dir / "field-solutions-fast-phase.h5"
        future_medium = future_solution_dir / "field-solutions-medium1-phase.h5"
        future_slow = future_solution_dir / "field-solutions-slow-gain.h5"
        for path in (current_fast, future_medium, future_slow):
            path.write_text("h5parm")

        calibrate_field.fast_phases_h5parm_filename = str(current_fast)
        calibrate_field.medium1_phases_h5parm_filename = str(future_medium)
        calibrate_field.slow_gains_h5parm_filename = str(future_slow)

        calibrate = Calibrate("dd", field=calibrate_field, index=1)
        calibrate.set_input_parameters()

        assert calibrate.input_parms["solve1_initialsolutions_h5parm"]["path"] == str(current_fast)
        assert calibrate.input_parms["solve2_initialsolutions_h5parm"] is None
        assert calibrate.input_parms["solve3_initialsolutions_h5parm"] is None

    def test_set_input_parameters_dd_allows_previous_cycle_solve_initial_solutions(
        self, calibrate_field, tmp_path
    ):
        """Previous-cycle same-solve products seed DP3 but are not pre-applied."""
        calibrate_field.calibration_strategy = {"dd": ["fast_phase", "medium_phase", "slow_gains"]}
        calibrate_field._calibration_strategy_defaulted = False

        previous_solution_dir = tmp_path / "solutions" / "calibrate_1"
        previous_solution_dir.mkdir(parents=True)
        previous_fast = previous_solution_dir / "field-solutions-fast-phase.h5"
        previous_medium = previous_solution_dir / "field-solutions-medium1-phase.h5"
        previous_slow = previous_solution_dir / "field-solutions-slow-gain.h5"
        for path in (previous_fast, previous_medium, previous_slow):
            path.write_text("h5parm")

        calibrate_field.fast_phases_h5parm_filename = str(previous_fast)
        calibrate_field.medium1_phases_h5parm_filename = str(previous_medium)
        calibrate_field.slow_gains_h5parm_filename = str(previous_slow)

        calibrate = Calibrate("dd", field=calibrate_field, index=2)
        calibrate.set_input_parameters()

        assert calibrate.input_parms["solve1_initialsolutions_h5parm"]["path"] == str(previous_fast)
        assert calibrate.input_parms["solve2_initialsolutions_h5parm"]["path"] == str(
            previous_medium
        )
        assert calibrate.input_parms["solve3_initialsolutions_h5parm"]["path"] == str(previous_slow)
        assert calibrate.input_parms["applycal_steps"] is None
        assert calibrate.input_parms["applycal_h5parm"] is None

    def test_set_input_parameters_dd_phase_only_reuses_only_previous_fast_initial_solution(
        self, calibrate_field, tmp_path
    ):
        """Match master: phase-only DD cycles carry forward only the fast-phase seed."""
        calibrate_field.calibration_strategy = {"dd": ["fast_phase", "medium_phase"]}
        calibrate_field._calibration_strategy_defaulted = False

        previous_solution_dir = tmp_path / "solutions" / "calibrate_1"
        previous_solution_dir.mkdir(parents=True)
        previous_fast = previous_solution_dir / "field-solutions-fast-phase.h5"
        previous_medium = previous_solution_dir / "field-solutions-medium1-phase.h5"
        for path in (previous_fast, previous_medium):
            path.write_text("h5parm")

        calibrate_field.fast_phases_h5parm_filename = str(previous_fast)
        calibrate_field.medium1_phases_h5parm_filename = str(previous_medium)

        calibrate = Calibrate("dd", field=calibrate_field, index=2)
        calibrate.set_input_parameters()

        assert calibrate.input_parms["solve1_initialsolutions_h5parm"]["path"] == str(previous_fast)
        assert calibrate.input_parms["solve2_initialsolutions_h5parm"] is None
        assert calibrate.input_parms["applycal_steps"] is None
        assert calibrate.input_parms["applycal_h5parm"] is None

    def test_set_input_parameters_dd_rejects_future_cycle_solve_initial_solutions(
        self, calibrate_field, tmp_path, caplog
    ):
        calibrate_field.calibration_strategy = {"dd": ["fast_phase", "medium_phase", "slow_gains"]}
        calibrate_field._calibration_strategy_defaulted = False

        future_solution_dir = tmp_path / "solutions" / "calibrate_3"
        future_solution_dir.mkdir(parents=True)
        future_fast = future_solution_dir / "field-solutions-fast-phase.h5"
        future_medium = future_solution_dir / "field-solutions-medium1-phase.h5"
        future_slow = future_solution_dir / "field-solutions-slow-gain.h5"
        for path in (future_fast, future_medium, future_slow):
            path.write_text("h5parm")

        calibrate_field.fast_phases_h5parm_filename = str(future_fast)
        calibrate_field.medium1_phases_h5parm_filename = str(future_medium)
        calibrate_field.slow_gains_h5parm_filename = str(future_slow)

        calibrate = Calibrate("dd", field=calibrate_field, index=2)
        calibrate.set_input_parameters()

        assert calibrate.input_parms["solve1_initialsolutions_h5parm"] is None
        assert calibrate.input_parms["solve2_initialsolutions_h5parm"] is None
        assert calibrate.input_parms["solve3_initialsolutions_h5parm"] is None
        assert "future cycle 3" in caplog.text

    def test_set_input_parameters_dd_does_not_use_di_solutions_as_dd_initial_solutions(
        self, calibrate_field, tmp_path
    ):
        calibrate_field.calibration_strategy = {"dd": ["fast_phase", "medium_phase"]}
        calibrate_field._calibration_strategy_defaulted = False

        di_solution_dir = tmp_path / "solutions" / "calibrate_di_1"
        di_solution_dir.mkdir(parents=True)
        di_fast = di_solution_dir / "di-solutions-fast-phase.h5"
        di_medium = di_solution_dir / "di-solutions-medium1-phase.h5"
        for path in (di_fast, di_medium):
            path.write_text("h5parm")

        calibrate_field.di_fast_phases_h5parm_filename = str(di_fast)
        calibrate_field.di_medium1_phases_h5parm_filename = str(di_medium)

        calibrate = Calibrate("dd", field=calibrate_field, index=2)
        calibrate.set_input_parameters()

        assert calibrate.input_parms["solve1_initialsolutions_h5parm"] is None
        assert calibrate.input_parms["solve2_initialsolutions_h5parm"] is None

    def test_set_input_parameters_di_allows_current_cycle_solve_initial_solutions(
        self, calibrate_field, tmp_path
    ):
        calibrate_field.calibration_strategy = {
            "di": ["fast_phase", "medium_phase", "slow_gains", "full_jones"]
        }
        calibrate_field._calibration_strategy_defaulted = False

        current_solution_dir = tmp_path / "solutions" / "calibrate_di_1"
        future_solution_dir = tmp_path / "solutions" / "calibrate_di_2"
        current_solution_dir.mkdir(parents=True)
        future_solution_dir.mkdir(parents=True)

        current_fast = current_solution_dir / "di-solutions-fast-phase.h5"
        future_medium = future_solution_dir / "di-solutions-medium1-phase.h5"
        future_slow = future_solution_dir / "di-solutions-slow-gain.h5"
        current_fulljones = current_solution_dir / "fulljones-solutions.h5"
        for path in (current_fast, future_medium, future_slow, current_fulljones):
            path.write_text("h5parm")

        calibrate_field.di_fast_phases_h5parm_filename = str(current_fast)
        calibrate_field.di_medium1_phases_h5parm_filename = str(future_medium)
        calibrate_field.di_slow_gains_h5parm_filename = str(future_slow)
        calibrate_field.fulljones_h5parm_filename = str(current_fulljones)

        calibrate = Calibrate("di", field=calibrate_field, index=1)
        calibrate.set_input_parameters()

        assert calibrate.input_parms["solve1_initialsolutions_h5parm"]["path"] == str(current_fast)
        assert calibrate.input_parms["solve2_initialsolutions_h5parm"] is None
        assert calibrate.input_parms["solve3_initialsolutions_h5parm"] is None
        assert calibrate.input_parms["solve4_initialsolutions_h5parm"]["path"] == str(
            current_fulljones
        )

    def test_set_input_parameters_di_allows_previous_cycle_solve_initial_solutions(
        self, calibrate_field, tmp_path
    ):
        calibrate_field.calibration_strategy = {
            "di": ["fast_phase", "medium_phase", "slow_gains", "full_jones"]
        }
        calibrate_field._calibration_strategy_defaulted = False

        previous_solution_dir = tmp_path / "solutions" / "calibrate_di_1"
        previous_solution_dir.mkdir(parents=True)
        previous_fast = previous_solution_dir / "di-solutions-fast-phase.h5"
        previous_medium = previous_solution_dir / "di-solutions-medium1-phase.h5"
        previous_slow = previous_solution_dir / "di-solutions-slow-gain.h5"
        previous_fulljones = previous_solution_dir / "fulljones-solutions.h5"
        for path in (previous_fast, previous_medium, previous_slow, previous_fulljones):
            path.write_text("h5parm")

        calibrate_field.di_fast_phases_h5parm_filename = str(previous_fast)
        calibrate_field.di_medium1_phases_h5parm_filename = str(previous_medium)
        calibrate_field.di_slow_gains_h5parm_filename = str(previous_slow)
        calibrate_field.fulljones_h5parm_filename = str(previous_fulljones)

        calibrate = Calibrate("di", field=calibrate_field, index=2)
        calibrate.set_input_parameters()

        assert calibrate.input_parms["solve1_initialsolutions_h5parm"]["path"] == str(previous_fast)
        assert calibrate.input_parms["solve2_initialsolutions_h5parm"]["path"] == str(
            previous_medium
        )
        assert calibrate.input_parms["solve3_initialsolutions_h5parm"]["path"] == str(previous_slow)
        assert calibrate.input_parms["solve4_initialsolutions_h5parm"]["path"] == str(
            previous_fulljones
        )

    def test_set_input_parameters_di_does_not_use_dd_solutions_as_di_initial_solutions(
        self, calibrate_field, tmp_path
    ):
        calibrate_field.calibration_strategy = {"di": ["fast_phase", "medium_phase", "slow_gains"]}
        calibrate_field._calibration_strategy_defaulted = False

        dd_solution_dir = tmp_path / "solutions" / "calibrate_1"
        dd_solution_dir.mkdir(parents=True)
        dd_fast = dd_solution_dir / "field-solutions-fast-phase.h5"
        dd_medium = dd_solution_dir / "field-solutions-medium1-phase.h5"
        dd_slow = dd_solution_dir / "field-solutions-slow-gain.h5"
        for path in (dd_fast, dd_medium, dd_slow):
            path.write_text("h5parm")

        calibrate_field.fast_phases_h5parm_filename = str(dd_fast)
        calibrate_field.medium1_phases_h5parm_filename = str(dd_medium)
        calibrate_field.slow_gains_h5parm_filename = str(dd_slow)

        calibrate = Calibrate("di", field=calibrate_field, index=2)
        calibrate.set_input_parameters()

        assert calibrate.input_parms["solve1_initialsolutions_h5parm"] is None
        assert calibrate.input_parms["solve2_initialsolutions_h5parm"] is None
        assert calibrate.input_parms["solve3_initialsolutions_h5parm"] is None

    def test_set_input_parameters_dd_preapplies_di_solutions(self, calibrate_field, tmp_path):
        di_h5parm = tmp_path / "di-solutions.h5"
        fulljones_h5parm = tmp_path / "fulljones-solutions.h5"
        di_h5parm.touch()
        fulljones_h5parm.touch()
        calibrate_field.di_h5parm_filename = str(di_h5parm)
        calibrate_field.fulljones_h5parm_filename = str(fulljones_h5parm)
        calibrate_field.apply_amplitudes = True

        calibrate = Calibrate("dd", field=calibrate_field, index=1)
        calibrate.set_input_parameters()

        assert parse_dp3(calibrate.input_parms["dp3_steps"])[0] == "applycal"
        assert calibrate.input_parms["applycal_steps"] == "[fastphase,slowgain,fulljones]"
        assert (
            calibrate.input_parms["calibration_applycal_steps"] == "[fastphase,slowgain,fulljones]"
        )
        assert calibrate.input_parms["applycal_h5parm"]["path"] == str(di_h5parm)
        assert calibrate.input_parms["fulljones_h5parm"]["path"] == str(fulljones_h5parm)

    def test_set_input_parameters_dd_skips_previous_cycle_preapply_solutions(
        self, calibrate_field, tmp_path
    ):
        di_h5parm = tmp_path / "di-solutions.h5"
        fulljones_h5parm = tmp_path / "fulljones-solutions.h5"
        di_h5parm.touch()
        fulljones_h5parm.touch()
        calibrate_field.di_h5parm_filename = str(di_h5parm)
        calibrate_field.di_h5parm_cycle_number = 1
        calibrate_field.fulljones_h5parm_filename = str(fulljones_h5parm)
        calibrate_field.fulljones_h5parm_cycle_number = 1
        calibrate_field.apply_amplitudes = True

        calibrate = Calibrate("dd", field=calibrate_field, index=2)
        calibrate.set_input_parameters()

        assert parse_dp3(calibrate.input_parms["dp3_steps"]) == ["solve1", "solve2"]
        assert calibrate.input_parms["applycal_steps"] is None
        assert calibrate.input_parms["calibration_applycal_steps"] is None
        assert calibrate.input_parms["applycal_h5parm"] is None
        assert calibrate.input_parms["fulljones_h5parm"] is None

    def test_dd_commands_do_not_preapply_previous_cycle_di_solutions(
        self, calibrate_field, tmp_path
    ):
        di_h5parm = tmp_path / "solutions" / "calibrate_di_1" / "di-solutions.h5"
        fulljones_h5parm = tmp_path / "solutions" / "calibrate_di_1" / "fulljones-solutions.h5"
        di_h5parm.parent.mkdir(parents=True)
        di_h5parm.touch()
        fulljones_h5parm.touch()
        calibrate_field.di_h5parm_filename = str(di_h5parm)
        calibrate_field.di_h5parm_cycle_number = 1
        calibrate_field.fulljones_h5parm_filename = str(fulljones_h5parm)
        calibrate_field.fulljones_h5parm_cycle_number = 1
        calibrate_field.apply_amplitudes = True

        def get_obs_parameters(name):
            values = {
                "timechunk_filename": ["chunk_0.ms", "chunk_1.ms"],
                "starttime": [0, 100],
                "ntimes": [10, 10],
                "solint_fast_timestep": [3, 4],
                "solint_medium_timestep": [9, 10],
                "solint_slow_timestep": [11, 12],
                "solint_fast_freqstep": [1, 2],
                "solint_medium_freqstep": [5, 6],
                "solint_slow_freqstep": [7, 8],
                "fast_solutions_per_direction": [[1], [1]],
                "medium_solutions_per_direction": [[1], [1]],
                "slow_solutions_per_direction": [None, None],
                "fast_smoothness_dd_factors": [[1.0], [1.5]],
                "medium_smoothness_dd_factors": [[2.0], [2.5]],
                "slow_smoothness_dd_factors": [[1.0], [1.0]],
                "fast_smoothnessreffrequency": [150000000.0, 151000000.0],
                "medium_smoothnessreffrequency": [152000000.0, 153000000.0],
                "bda_maxinterval": [8.0, 9.0],
                "bda_minchannels": [1, 1],
            }
            return values[name]

        calibrate_field.get_obs_parameters.side_effect = get_obs_parameters

        calibrate = Calibrate("dd", field=calibrate_field, index=2)
        calibrate.set_input_parameters()
        payload = calibrate_payload_from_inputs(
            "dd",
            calibrate.input_parms,
            tmp_path / "pipeline" / calibrate.name,
        )

        commands = [build_calibrate_chunk_command(payload, chunk) for chunk in payload["chunks"]]

        for command in commands:
            command_string = " ".join(command)
            assert "steps=[solve1,solve2]" in command
            assert not any(token.startswith("applycal.steps=") for token in command)
            assert not any(token.startswith("applycal.parmdb=") for token in command)
            assert not any(token.startswith("applycal.fulljones.parmdb=") for token in command)
            assert str(di_h5parm) not in command_string
            assert str(fulljones_h5parm) not in command_string

    def test_set_input_parameters_dd_preapply_skips_di_slowgain_with_phase_solves(
        self, calibrate_field, tmp_path
    ):
        di_h5parm = tmp_path / "di-solutions.h5"
        di_h5parm.touch()
        calibrate_field.di_h5parm_filename = str(di_h5parm)
        calibrate_field.apply_amplitudes = True
        calibrate_field.calibration_strategy = {
            "di": ["fast_phase", "medium_phase", "slow_gains"],
            "dd": ["fast_phase"],
        }

        calibrate = Calibrate("dd", field=calibrate_field, index=1)
        calibrate.set_input_parameters()

        assert calibrate.input_parms["applycal_steps"] == "[fastphase]"
        assert calibrate.input_parms["calibration_applycal_steps"] == "[fastphase]"
        assert calibrate.input_parms["applycal_h5parm"]["path"] == str(di_h5parm)
        assert "mediumphase" not in parse_dp3(calibrate.input_parms["applycal_steps"])

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

        calibrate_dd = Calibrate("dd", field=calibrate_field, index=1)
        calibrate_dd.set_input_parameters()
        params = calibrate_dd.input_parms
        dp3 = parse_dp3(params["dp3_steps"])

        assert dp3[: len(expected_prefix)] == expected_prefix
        if expect_applycal:
            assert params["calibration_applycal_steps"] == "[normalization]"
            assert params["applycal_steps"] == "[normalization]"
        else:
            assert params["calibration_applycal_steps"] is None

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

        calibrate_dd = Calibrate("dd", field=calibrate_field, index=1)
        calibrate_dd.set_input_parameters()

        assert calibrate_dd.input_parms["solution_combine_mode"] == expected_mode

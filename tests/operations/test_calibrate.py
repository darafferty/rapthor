"""
Test cases for the `rapthor.operations.calibrate` module.
"""

from pathlib import Path

import pytest

import rapthor
from rapthor.lib.operation import DIR as OPERATION_DIR
from rapthor.operations.calibrate import CalibrateDD, CalibrateDI

@pytest.fixture
def calibrate_field(operation_parset, mocker):
    """Create a mock field object for testing a Calibrate operation."""

    class Field:
        def __init__(self, parset):
            self.parset = parset
            self.scan_h5parms = mocker.MagicMock()
            self.calibration_diagnostics = []

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


class TestCalibrateDD:
    def test_set_input_parameters(self):
        # calibrate_dd.set_input_parameters()
        pass

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

    def test_get_model_image_parameters(self, tmp_path, calibrate_field):
        ref_frequency = 142000000.0
        bandwidth = 1e6 # hardcoded value in Rapthor.
        center_ra = 42.0
        center_dec = -42.0
        cellsize_arcsec = 1.8
        cellsize_degrees = cellsize_arcsec / 3600.0
        width_ra_pixels = 3600
        width_dec_pixels = 4800
        width_ra_degrees = cellsize_degrees * width_ra_pixels
        width_dec_degrees = cellsize_degrees * width_dec_pixels

        # Create a dummy skymodel. Only the reference frequency is relevant for this test.
        skymodel_path = tmp_path / "test_skymodel.txt"
        skymodel_path.write_text(
            "FORMAT = Name, Type, Ra, Dec, I, ReferenceFrequency\n"
            f"src, POINT, 13:42:42, -24.24.42.42, 0.042, {ref_frequency}\n"
        )
        field = calibrate_field
        field.calibration_skymodel_file = str(skymodel_path)
        field.parset["imaging_specific"] = { "cellsize_arcsec": cellsize_arcsec }
        field.sector_bounds_mid_ra = center_ra
        field.sector_bounds_mid_dec = center_dec
        field.sector_bounds_width_ra = width_ra_degrees
        field.sector_bounds_width_dec = width_dec_degrees

        calibrate_dd = CalibrateDD(field, index=2)
        frequency_bandwidth, center_coords, size, cellsize = calibrate_dd.get_model_image_parameters()
        assert frequency_bandwidth == [ ref_frequency, bandwidth ]
        assert center_coords == ("2:48:00.000000", "-42.00.00.000000")
        assert size == [width_ra_pixels, width_dec_pixels]
        assert cellsize == cellsize_degrees


class TestCalibrateDI:
    def test_set_input_parameters(self):
        # calibrate_di.set_input_parameters()
        pass


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
        field.generate_screens = False
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

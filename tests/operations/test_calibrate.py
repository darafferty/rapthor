"""
Test cases for the `rapthor.operations.calibrate` module.
"""

import pytest

from rapthor.operations.calibrate import CalibrateDD, CalibrateDI


@pytest.fixture
def parset(tmp_path):
    """Create a mock parset, which only has keys for calibration."""
    return {
        "dir_working": str(tmp_path / "working"),
        # Cluster-specific parameters are needed for the base Operation class.
        "cluster_specific": {
            "cwl_runner": "mock_cwl_runner",
            "debug_workflow": False,
            "keep_temporary_files": False,
            "max_nodes": 1,
            "batch_system": "mock_batch_system",
            "cpus_per_task": 1,
            "mem_per_node_gb": 1,
            "dir_local": str(tmp_path / "scratch"),
            "local_scratch_dir": str(tmp_path / "local_scratch"),
            "global_scratch_dir": str(tmp_path / "global_scratch"),
            "use_container": False,
        },
    }


@pytest.fixture
def field(parset, mocker):
    """Create a mock field object for testing."""

    class Field:
        def __init__(self, parset):
            self.parset = parset
            self.scan_h5parms = mocker.MagicMock()
            self.calibration_diagnostics = []

    return Field(parset)


@pytest.fixture
def finalize_mocks(mocker):
    """Setup mocks for the finalize() method tests."""
    mocker.patch("rapthor.lib.miscellaneous.get_flagged_solution_fraction", return_value=0.042)
    return {
        "makedirs": mocker.patch("os.makedirs"),
        "remove": mocker.patch("os.remove"),
        "copy": mocker.patch("shutil.copy"),
    }


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


def finalize_check_plots(pipelines_path, plots_path, mocks):
    """Helper function to check that the correct plot files were removed and copied."""
    mocks["remove"].assert_any_call(str(plots_path / "plot2.png"))
    mocks["copy"].assert_any_call(str(pipelines_path / "plot1.png"), str(plots_path / "plot1.png"))
    mocks["copy"].assert_any_call(str(pipelines_path / "plot2.png"), str(plots_path / "plot2.png"))


class TestCalibrateDD:
    def test_set_parset_parameters(self, calibrate_dd):
        # calibrate_dd.set_parset_parameters()
        pass

    def test_set_input_parameters(self, calibrate_dd):
        # calibrate_dd.set_input_parameters()
        pass

    def test_get_baselines_core(self, calibrate_dd):
        # calibrate_dd.get_baselines_core()
        pass

    def test_get_superterp_stations(self, calibrate_dd):
        # calibrate_dd.get_superterp_stations()
        pass

    def test_get_core_stations(self, calibrate_dd):
        # calibrate_dd.get_core_stations(include_nearest_remote=True)
        pass

    def test_get_model_image_parameters(self, calibrate_dd):
        # calibrate_dd.get_model_image_parameters()
        pass

    @pytest.mark.parametrize("do_slowgain", [True, False])
    def test_finalize(self, field, tmp_path, finalize_mocks, do_slowgain):
        # Setup working directory
        workdir_path = tmp_path / "working"
        solutions_path = workdir_path / "solutions" / "calibrate_2"
        solutions_path.mkdir(parents=True)

        # Create an existing solutions file. finalize() should remove it.
        h5parm_path = solutions_path / "field-solutions.h5"
        h5parm_path.touch()

        pipelines_path = workdir_path / "pipelines" / "calibrate_2"
        plots_path = workdir_path / "plots" / "calibrate_2"
        finalize_prepare_plots(pipelines_path, plots_path)

        # Setup the object itself
        field.generate_screens = False
        field.do_slowgain_solve = do_slowgain
        calibrate_dd = CalibrateDD(field, index=2)
        calibrate_dd.combined_h5parms = "combined.test.h5"
        if do_slowgain:
            calibrate_dd.slow_h5parm = "slow.test.h5"
            calibrate_dd.medium1_h5parm = "medium1.test.h5"
            calibrate_dd.medium2_h5parm = "medium2.test.h5"
        calibrate_dd.fast_h5parm = "fast.test.h5"

        # Ignore os.makedirs calls from the base Operation class constructor.
        finalize_mocks["makedirs"].reset_mock()

        calibrate_dd.finalize()

        assert field.h5parm_filename == str(h5parm_path)
        assert field.fast_phases_h5parm_filename == str(
            solutions_path / "field-solutions-fast-phase.h5"
        )
        assert field.medium1_phases_h5parm_filename == str(
            solutions_path / "field-solutions-medium1-phase.h5"
        )
        assert field.medium2_phases_h5parm_filename == str(
            solutions_path / "field-solutions-medium2-phase.h5"
        )
        assert field.slow_gains_h5parm_filename == str(
            solutions_path / "field-solutions-slow-gain.h5"
        )

        check_makedirs(finalize_mocks["makedirs"], solutions_path, plots_path)

        # Check removing and copying solutions.
        finalize_mocks["remove"].assert_any_call(str(h5parm_path))
        if do_slowgain:
            solution_src_dst_list = [
                ("combined.test.h5", str(h5parm_path)),
                ("slow.test.h5", "field-solutions-slow-gain.h5"),
                ("medium1.test.h5", "field-solutions-medium1-phase.h5"),
                ("medium2.test.h5", "field-solutions-medium2-phase.h5"),
                ("fast.test.h5", "field-solutions-fast-phase.h5"),
            ]
        else:
            solution_src_dst_list = [
                ("fast.test.h5", str(h5parm_path)),
                ("fast.test.h5", "field-solutions-fast-phase.h5"),
            ]

        for src, dst in solution_src_dst_list:
            finalize_mocks["copy"].assert_any_call(
                str(pipelines_path / src), str(solutions_path / dst)
            )

        field.scan_h5parms.assert_called_once()
        assert field.calibration_diagnostics == [
            {
                "cycle_number": 2,
                "solution_flagged_fraction": 0.042,  # See finalize_mocks fixture.
            }
        ]

        finalize_check_plots(pipelines_path, plots_path, finalize_mocks)

        # For the plots, there is 1 remove and 2 copies. See finalize_check_plots.
        assert finalize_mocks["remove"].call_count == 1 + 1
        assert finalize_mocks["copy"].call_count == len(solution_src_dst_list) + 2

        # finalize() should create a .done file (via the base Operation class).
        assert (pipelines_path / ".done").exists()


class TestCalibrateDI:
    def test_set_parset_parameters(self, calibrate_di):
        # calibrate_di.set_parset_parameters()
        pass

    def test_set_input_parameters(self, calibrate_di):
        # calibrate_di.set_input_parameters()
        pass

    def test_finalize(self, field, tmp_path, finalize_mocks):
        # Setup working directory
        workdir_path = tmp_path / "working"
        solutions_path = workdir_path / "solutions" / "calibrate_di_4"
        solutions_path.mkdir(parents=True)

        # Create an existing fulljones solutions file. finalize() should remove it.
        fulljones_h5parm_path = solutions_path / "fulljones-solutions.h5"
        fulljones_h5parm_path.touch()

        pipelines_path = workdir_path / "pipelines" / "calibrate_di_4"
        plots_path = workdir_path / "plots" / "calibrate_di_4"
        finalize_prepare_plots(pipelines_path, plots_path)

        # Setup the object itself
        calibrate_di = CalibrateDI(field, index=4)
        collected_fulljones_filename = "collected_fulljones.h5"
        calibrate_di.collected_h5parm_fulljones = collected_fulljones_filename

        # Ignore os.makedirs calls from the base Operation class constructor.
        finalize_mocks["makedirs"].reset_mock()

        # Act
        calibrate_di.finalize()

        # Assert
        assert field.fulljones_h5parm_filename == str(fulljones_h5parm_path)
        field.scan_h5parms.assert_called_once()

        check_makedirs(finalize_mocks["makedirs"], solutions_path, plots_path)

        # Check removing and copying solutions.
        finalize_mocks["remove"].assert_any_call(str(fulljones_h5parm_path))
        finalize_mocks["copy"].assert_any_call(
            str(pipelines_path / collected_fulljones_filename), str(fulljones_h5parm_path)
        )

        finalize_check_plots(pipelines_path, plots_path, finalize_mocks)

        assert finalize_mocks["remove"].call_count == 2
        assert finalize_mocks["copy"].call_count == 3

        # finalize() should create a .done file (via the base Operation class).
        assert (pipelines_path / ".done").exists()


class TestCalibrate:
    @pytest.mark.parametrize("scenario", ["dd_fast_only", "dd_with_slowgain", "di_fulljones"])
    def test_finalize(self, mocker, field, tmp_path, scenario):
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
        name = "calibrate_2" if scenario.startswith("dd") else "calibrate_di_4"
        solutions_path = workdir_path / "solutions" / name
        solutions_path.mkdir(parents=True)

        # Create an existing solutions file. finalize() should remove it.
        if scenario.startswith("dd"):
            h5parm_path = solutions_path / "field-solutions.h5"
        else:
            h5parm_path = solutions_path / "fulljones-solutions.h5"
        h5parm_path.touch()

        pipelines_path = workdir_path / "pipelines" / name
        plots_path = workdir_path / "plots" / name
        finalize_prepare_plots(pipelines_path, plots_path)

        # Setup the object itself
        field.generate_screens = False
        field.do_slowgain_solve = scenario == "dd_with_slowgain"

        if scenario.startswith("dd"):
            calibrate = CalibrateDD(field, index=2)
        else:
            calibrate = CalibrateDI(field, index=4)

        if scenario.startswith("dd"):
            calibrate.combined_h5parms = "combined.test.h5"
            calibrate.fast_h5parm = "fast.test.h5"
        if scenario == "dd_with_slowgain":
            calibrate.slow_h5parm = "slow.test.h5"
            calibrate.medium1_h5parm = "medium1.test.h5"
            calibrate.medium2_h5parm = "medium2.test.h5"
        elif scenario == "di_fulljones":
            calibrate.collected_h5parm_fulljones = "collected_fulljones.h5"

        # Ignore os.makedirs calls from the base Operation class constructor.
        mock_makedirs.reset_mock()

        # Act
        calibrate.finalize()

        # Assert
        if scenario.startswith("dd"):
            assert field.h5parm_filename == str(h5parm_path)
            assert field.fast_phases_h5parm_filename == str(
                solutions_path / "field-solutions-fast-phase.h5"
            )
        if scenario == "dd_with_slowgain":
            assert field.medium1_phases_h5parm_filename == str(
                solutions_path / "field-solutions-medium1-phase.h5"
            )
            assert field.medium2_phases_h5parm_filename == str(
                solutions_path / "field-solutions-medium2-phase.h5"
            )
            assert field.slow_gains_h5parm_filename == str(
                solutions_path / "field-solutions-slow-gain.h5"
            )
        elif scenario == "di_fulljones":
            assert field.fulljones_h5parm_filename == str(h5parm_path)

        check_makedirs(mock_makedirs, solutions_path, plots_path)

        # Check removing and copying solutions.
        mock_remove.assert_any_call(str(h5parm_path))

        if scenario == "dd_with_slowgain":
            solution_src_dst_list = [
                ("combined.test.h5", h5parm_path.name),
                ("slow.test.h5", "field-solutions-slow-gain.h5"),
                ("medium1.test.h5", "field-solutions-medium1-phase.h5"),
                ("medium2.test.h5", "field-solutions-medium2-phase.h5"),
                ("fast.test.h5", "field-solutions-fast-phase.h5"),
            ]
        elif scenario == "dd_fast_only":
            solution_src_dst_list = [
                ("fast.test.h5", h5parm_path.name),
                ("fast.test.h5", "field-solutions-fast-phase.h5"),
            ]
        elif scenario == "di_fulljones":
            solution_src_dst_list = [
                ("collected_fulljones.h5", h5parm_path.name),
            ]

        for src, dst in solution_src_dst_list:
            mock_copy.assert_any_call(str(pipelines_path / src), str(solutions_path / dst))

        field.scan_h5parms.assert_called_once()

        if scenario.startswith("dd"):
            assert field.calibration_diagnostics == [
                {
                    "cycle_number": 2,
                    "solution_flagged_fraction": 0.042,  # See finalize_mocks fixture.
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

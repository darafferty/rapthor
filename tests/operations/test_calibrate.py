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

    return Field(parset)


@pytest.fixture
def calibrate_dd(field, index=1):
    """
    Create an instance of the CalibrateDD operation.
    """
    # return CalibrateDD(field, index=index)
    return "mock_calibrate_dd"


@pytest.fixture
def calibrate_di(field, tmp_path):
    """Create an instance of the CalibrateDI operation."""
    return CalibrateDI(field, index=0)


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


def finalize_prepare_plots(pipeline_path, plots_path):
    """Helper function to prepare the plot files for the finalize() tests."""
    # Create dummy plot files. finalize() should copy them to the plots directory.
    (pipeline_path / "plot1.png").touch()
    (pipeline_path / "plot2.png").touch()

    # Simulate one existing plot in the plots directory. finalize() should remove it.
    plots_path.mkdir(parents=True)
    (plots_path / "plot2.png").touch()


def finalize_check_plots(pipeline_path, plots_path, mock_remove, mock_copy):
    """Helper function to check that the correct plot files were removed and copied."""
    mock_remove.assert_any_call(str(plots_path / "plot2.png"))
    mock_copy.assert_any_call(str(pipeline_path / "plot1.png"), str(plots_path / "plot1.png"))
    mock_copy.assert_any_call(str(pipeline_path / "plot2.png"), str(plots_path / "plot2.png"))


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

    def test_finalize(self, calibrate_dd):
        # calibrate_dd.finalize()
        pass


class TestCalibrateDI:
    def test_set_parset_parameters(self, calibrate_di):
        # calibrate_di.set_parset_parameters()
        pass

    def test_set_input_parameters(self, calibrate_di):
        # calibrate_di.set_input_parameters()
        pass

    def test_finalize(self, calibrate_di, field, mocker, tmp_path, finalize_mocks):
        # Setup working directory
        workdir_path = tmp_path / "working"
        solutions_path = workdir_path / "solutions" / "calibrate_di_0"
        solutions_path.mkdir(parents=True)

        # Create an existing fulljones solutions file. finalize() should remove it.
        fulljones_h5parm_path = solutions_path / "fulljones-solutions.h5"
        fulljones_h5parm_path.touch()

        pipeline_path = workdir_path / "pipelines" / "calibrate_di_0"
        plots_path = workdir_path / "plots" / "calibrate_di_0"
        finalize_prepare_plots(pipeline_path, plots_path)

        # Setup the object itself
        collected_fulljones_filename = "collected_fulljones.h5"
        calibrate_di.collected_h5parm_fulljones = collected_fulljones_filename

        # Act
        calibrate_di.finalize()

        # Assert
        assert field.fulljones_h5parm_filename == str(fulljones_h5parm_path)
        field.scan_h5parms.assert_called_once()

        check_makedirs(finalize_mocks["makedirs"], solutions_path, plots_path)

        # Check removing and copying solutions.
        finalize_mocks["remove"].assert_any_call(str(fulljones_h5parm_path))
        finalize_mocks["copy"].assert_any_call(
            str(pipeline_path / collected_fulljones_filename), str(fulljones_h5parm_path)
        )

        finalize_check_plots(pipeline_path, plots_path, finalize_mocks["remove"], finalize_mocks["copy"])

        assert finalize_mocks["remove"].call_count == 2
        assert finalize_mocks["copy"].call_count == 3

        # finalize() should create a .done file (via the base Operation class).
        assert (pipeline_path / ".done").exists()

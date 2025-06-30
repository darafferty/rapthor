"""
Test cases for the `rapthor.operations.calibrate` module.
"""

import pytest
from rapthor.operations.calibrate import CalibrateDD, CalibrateDI


@pytest.fixture
def field():
    # Mock or create a field object as needed for testing
    return "mock_field"


@pytest.fixture
def calibrate_dd(field, index=1):
    """
    Create an instance of the CalibrateDD operation.
    """
    # return CalibrateDD(field, index=index)
    return "mock_calibrate_dd"


@pytest.fixture
def calibrate_di(field, index=1):
    """
    Create an instance of the CalibrateDI operation.
    """
    # return CalibrateDI(field, index=index)
    return "mock_calibrate_di"


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

    def test_finalize(self, calibrate_di):
        # calibrate_di.finalize()
        pass

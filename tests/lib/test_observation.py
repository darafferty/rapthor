"""
Test cases for the `rapthor.lib.observation` module.
"""

import pytest
from logging import Logger
import numpy as np

class TestObservation:
    """
    Test cases for the Observation class.
    """
    def test_constructor(self, observation, test_ms):
        assert observation.ms_filename == test_ms
        assert observation.ms_predict_di_filename is None
        assert observation.ms_predict_nc_filename is None
        assert observation.name == "test.ms"
        assert isinstance(observation.log, Logger)
        assert np.isclose(observation.starttime, 4871282392.906812, rtol=1e-9)
        assert np.isclose(observation.endtime, 4871282443.176593, rtol=1e-9)
        assert observation.data_fraction == 1.0
        assert observation.parameters == {}

    def test_copy(self):
        pass

    def test_scan_ms(self):
        pass

    def test_set_calibration_parameters(
        self,
        parset=None,
        ndir=None,
        nobs=None,
        calibrator_fluxes=None,
        target_fast_timestep=None,
        target_slow_timestep_joint=None,
        target_slow_timestep_separate=None,
        target_fulljones_timestep=None,
        target_flux=None,
    ):
        pass

    def test_set_prediction_parameters(self, sector_name=None, patch_names=None):
        pass

    def test_set_imaging_parameters(
        self,
        sector_name=None,
        cellsize_arcsec=None,
        max_peak_smearing=None,
        width_ra=None,
        width_dec=None,
        solve_fast_timestep=None,
        solve_slow_timestep=None,
        solve_slow_freqstep=None,
        preapply_dde_solutions=None,
    ):
        pass

    def test_get_nearest_freqstep(self, freqstep=None):
        pass

    def test_get_target_timewidth(
        self, delta_theta=None, resolution=None, reduction_factor=None
    ):
        pass

    def test_get_bandwidth_smearing_factor(
        self, freq=None, delta_freq=None, delta_theta=None, resolution=None
    ):
        pass

    def test_get_target_bandwidth(
        self, freq=None, delta_theta=None, resolution=None, reduction_factor=None
    ):
        pass

    @pytest.mark.parametrize(
        "solints_seconds, solve_max_factor, expected_max",
        [
            # timepersample in test MS is 10.0139008 seconds
            ([20, 120, 600, 600], 1, 60),  # Default case
            ([20, 120, 600, 600], 2, 120),  # Increased solve_max_factor
            ([10, 10, 10, 10], 1, 1),  # All solints the same
            ([0.5, 0.5, 0.5, 0.5], 1, 1),  # All solints < 1 should return 1
            ([10.0139008, 10.0139008, 10.0139008, 10.0139008], 1, 1), # Exact match to timepersample
            ([10, 10.014, 10, 10], 1, 2), # Max solint above timepersample
        ],
    )
    def test_get_max_solint_timesteps(
        self, observation, solints_seconds, solve_max_factor, expected_max
    ):
        """
        Test the get_max_solint_timesteps method of the Observation class.
        """
        max_solint = observation.get_max_solint_timesteps(
            solints_seconds, solve_max_factor
        )
        assert max_solint == expected_max
        assert isinstance(max_solint, int)

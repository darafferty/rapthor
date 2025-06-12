"""
Test cases for the `rapthor.lib.observation` module.
"""

from rapthor.lib.observation import Observation


class TestObservation:
    """
    Test cases for the Observation class.
    """

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

"""
Test cases for the `rapthor.lib.observation` module.
"""

import pytest
from logging import Logger
import numpy as np
from unittest import mock

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
        assert observation.antenna == 'HBA'
        assert np.isclose(observation.channelwidth, 24414.0625)
        assert observation.numchannels == 8

    def test_copy(self):
        pass

    def test_scan_ms(self):
        pass

    def test_set_calibration_parameters_basic(self, observation, test_ms):
        """
        Basic set_calibration_parameters() test.
        All solution intervals and all DD factors are 1.
        """
        parset = {
            "calibration_specific": {
                "fast_freqstep_hz": observation.channelwidth,
                "medium_freqstep_hz": observation.channelwidth,
                "slow_freqstep_hz": observation.channelwidth,
                "fulljones_freqstep_hz": observation.channelwidth,
                "dd_interval_factor": 1,
                "dd_smoothness_factor": 1,
                "fast_smoothnessreffrequency": None,
                "medium_smoothnessreffrequency": None,
            }
        }

        n_observations = -1 # Not used in this test, since chunk_by_time is False.
        calibrator_fluxes = [1.0]
        observation.set_calibration_parameters(
            parset,
            n_observations,
            calibrator_fluxes,
            observation.timepersample,
            observation.timepersample,
            observation.timepersample,
            observation.timepersample,
        )

        assert observation.ntimechunks == 1

        params = observation.parameters
        assert params['timechunk_filename'] == [test_ms]
        assert params['predict_di_output_filename'] == [None]
        assert params['starttime'] == ['29Mar2013/13:59:52.907']
        assert params['ntimes'] == [0]

        for solve_type in ['fast', 'medium', 'slow', 'fulljones']:
            assert params[f'solint_{solve_type}_timestep'] == [1]
            assert params[f'solint_{solve_type}_freqstep'] == [1]

        assert params['bda_maxinterval'] == [observation.timepersample]
        assert params['bda_minchannels'] == [observation.numchannels]

        for solve_type in ['fast', 'medium', 'slow']:
            assert params[f'{solve_type}_solutions_per_direction'] == [[1]]
            assert params[f'{solve_type}_smoothness_dd_factors'] == [[1]]

        hba_reference_frequency = 144e6 # Hardcoded value for HBA antennas.
        assert params['fast_smoothnessreffrequency'] == [hba_reference_frequency]
        assert params['medium_smoothnessreffrequency'] == [hba_reference_frequency]



    @pytest.mark.parametrize("generate_screens", [True, False])
    @pytest.mark.parametrize("chunk_by_time", [True, False])
    def test_set_calibration_parameters(self, observation, test_ms, generate_screens, chunk_by_time):
        solint_fast_timestep = 20
        solint_medium_timestep = 120
        solint_slow_timestep = 300
        solint_fulljones_timestep = 600

        dd_interval_factor = 2
        dd_smoothness_factor = 10

        parset = {
            "calibration_specific": {
                "fast_freqstep_hz": 1e5,
                "medium_freqstep_hz": 1.5e5,
                "slow_freqstep_hz": 2e6,
                "fulljones_freqstep_hz": 1e6,
                "dd_interval_factor": dd_interval_factor,
                "dd_smoothness_factor": dd_smoothness_factor,
                "fast_smoothnessreffrequency": None,
                "medium_smoothnessreffrequency": None,
            }
        }

        if generate_screens:
            dd_interval_factor = 1
            dd_smoothness_factor = 1
        else:
            # When generate_screens is True, set_calibration_parameters should
            # not read these parset keys.
            parset["calibration_specific"]["dd_interval_factor"] = dd_interval_factor
            parset["calibration_specific"]["dd_smoothness_factor"] = dd_smoothness_factor

        if chunk_by_time:
            cluster_specific_parameters = "mock cluster specific parameters"
            parset["cluster_specific"] = cluster_specific_parameters

        with mock.patch("rapthor.lib.observation.get_chunk_size") as mock_get_chunk_size:
            mock_get_chunk_size.return_value = 10

            n_observations = 4
            calibrator_fluxes = [1.0, 0.5, 0.25, 0.125]
            observation.set_calibration_parameters(
                parset,
                n_observations,
                calibrator_fluxes,
                solint_fast_timestep * observation.timepersample,
                solint_medium_timestep * observation.timepersample,
                solint_slow_timestep * observation.timepersample,
                solint_fulljones_timestep * observation.timepersample,
                generate_screens=generate_screens,
                chunk_by_time=chunk_by_time,
            )

            if chunk_by_time:
                mock_get_chunk_size.assert_called_once_with(
                    cluster_specific_parameters, observation.numsamples, n_observations,
                    solint_fulljones_timestep * dd_interval_factor)
            else:
                mock_get_chunk_size.assert_not_called()

        assert observation.ntimechunks == 1

        params = observation.parameters
        assert params['timechunk_filename'] == [test_ms]
        assert params['predict_di_output_filename'] == [None]
        assert params['starttime'] == ['29Mar2013/13:59:52.907']
        assert params['ntimes'] == [0]

        assert params['solint_fast_timestep'] == [solint_fast_timestep * dd_interval_factor]
        assert params['solint_fast_freqstep'] == [4]
        assert params['solint_medium_timestep'] == [solint_medium_timestep * dd_interval_factor]
        assert params['solint_medium_freqstep'] == [8]
        assert params['solint_slow_timestep'] == [solint_slow_timestep * dd_interval_factor]
        assert params['solint_slow_freqstep'] == [8]
        assert params['solint_fulljones_timestep'] == [(solint_fulljones_timestep + 1) * dd_interval_factor]
        assert params['solint_fulljones_freqstep'] == [8]

        assert params['bda_maxinterval'] == [200.278016]
        assert params['bda_minchannels'] == [2]

        solutions_per_direction = [dd_interval_factor, dd_interval_factor, dd_interval_factor, 1]
        assert params['fast_solutions_per_direction'] == [solutions_per_direction]
        assert params['medium_solutions_per_direction'] == [solutions_per_direction]
        assert params['slow_solutions_per_direction'] == [solutions_per_direction]

        if generate_screens:
            smoothness_dd_factors = np.array([[1, 1, 1, 1]])
        else:
            smoothness_dd_factors = np.array([[1/3, 1/3, 1/2, 1]])
        assert (params['fast_smoothness_dd_factors'] == smoothness_dd_factors).all()
        assert (params['medium_smoothness_dd_factors'] == smoothness_dd_factors).all()
        assert (params['slow_smoothness_dd_factors'] == smoothness_dd_factors).all()

        assert params['fast_smoothnessreffrequency'] == [144e6]
        assert params['medium_smoothnessreffrequency'] == [144e6]


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

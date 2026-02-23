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
        assert observation.numsamples == 6
        assert observation.data_fraction == 1.0
        assert observation.parameters == {}
        assert observation.antenna == 'HBA'
        assert np.isclose(observation.channelwidth, 24414.0625)
        assert observation.numchannels == 8

    def test_copy(self):
        pass

    def test_scan_ms(self):
        pass

    def check_single_timechunk(self, observation, test_ms):
        """Check if the observation has a single time chunk."""
        assert observation.ntimechunks == 1
        params = observation.parameters
        assert params['timechunk_filename'] == [test_ms]
        assert params['predict_di_output_filename'] == [None]
        assert params['starttime'] == ['29Mar2013/13:59:52.907']
        assert params['ntimes'] == [0]

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

        self.check_single_timechunk(observation, test_ms)

        params = observation.parameters

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


    def test_set_calibration_parameters_solint(self, observation, test_ms):
        """
        Test set_calibration_parameters() with custom solutionn interval settings.
        All solution intervals and all DD factors are larger than 1.
        """

        time_factor = {
            'fast': 3,
            'medium': 5,
            'slow': 7,
            'fulljones': 9,
        }

        freq_factor = {
            'fast': 2,
            'medium': 4,
            'slow': 6,
            'fulljones': 8,
        }
        expected_freq_factor = {
            'fast': 2,
            'medium': 4,
            'slow': 8, # 6 does not divide 'numchannels', so should be rounded up to 8.
            'fulljones': 8,
        }


        dd_interval_factor = 10
        dd_smoothness_factor = 11

        parset = {
            "calibration_specific": {
                "fast_freqstep_hz": freq_factor['fast'] * observation.channelwidth,
                "medium_freqstep_hz": freq_factor['medium'] * observation.channelwidth,
                "slow_freqstep_hz": freq_factor['slow'] * observation.channelwidth,
                "fulljones_freqstep_hz": freq_factor['fulljones'] * observation.channelwidth,
                "dd_interval_factor": dd_interval_factor,
                "dd_smoothness_factor": dd_smoothness_factor,
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
            time_factor['fast'] * observation.timepersample,
            time_factor['medium'] * observation.timepersample,
            time_factor['slow'] * observation.timepersample,
            time_factor['fulljones'] * observation.timepersample,
        )

        self.check_single_timechunk(observation, test_ms)

        params = observation.parameters

        for solve_type in ['fast', 'medium', 'slow', 'fulljones']:
            expected_timestep = time_factor[solve_type]
            if solve_type != 'fulljones':
                expected_timestep *= dd_interval_factor
            expected_freqstep = expected_freq_factor[solve_type]
            assert params[f'solint_{solve_type}_timestep'] == [expected_timestep]
            assert params[f'solint_{solve_type}_freqstep'] == [expected_freqstep]

        # The fast solution interval is the smallest, so bda_maxinterval and
        # bda_minchannels should be based on that.
        expected_bda_maxinterval = time_factor['fast'] * observation.timepersample
        expected_bda_minchannels = observation.numchannels / expected_freq_factor['fast']
        assert params['bda_maxinterval'] == [expected_bda_maxinterval]
        assert params['bda_minchannels'] == [expected_bda_minchannels]

    @pytest.mark.parametrize("generate_screens", [True, False])
    def test_set_calibration_parameters_multiple_fluxes(self, observation, test_ms, generate_screens):
        """Test set_calibration_parameters() with multiple calibrator fluxes."""
        parset = {
            "calibration_specific": {
                "fast_freqstep_hz": observation.channelwidth,
                "medium_freqstep_hz": observation.channelwidth,
                "slow_freqstep_hz": observation.channelwidth,
                "fulljones_freqstep_hz": observation.channelwidth,
                "dd_interval_factor": 4,
                "dd_smoothness_factor": 2,
                "fast_smoothnessreffrequency": None,
                "medium_smoothnessreffrequency": None,
            }
        }

        n_observations = -1 # Not used in this test, since chunk_by_time is False.
        calibrator_fluxes = [1.0, 1.5, 2.5, 5.0]
        observation.set_calibration_parameters(
            parset,
            n_observations,
            calibrator_fluxes,
            observation.timepersample,
            observation.timepersample,
            observation.timepersample,
            observation.timepersample,
            generate_screens=generate_screens,
        )

        self.check_single_timechunk(observation, test_ms)

        params = observation.parameters

        for solve_type in ['fast', 'medium', 'slow']:
            solution_per_direction = params[f'{solve_type}_solutions_per_direction']
            smoothness_dd_factors = params[f'{solve_type}_smoothness_dd_factors']

            # There is one list for each time chunk.
            assert len(solution_per_direction) == 1
            assert len(smoothness_dd_factors) == 1

            if generate_screens:
                # When generate_screens is True, the dd factors are always 1.
                # The solutions per direction and smoothness factors are then also all 1.
                assert solution_per_direction[0] == [1, 1, 1, 1]
                assert smoothness_dd_factors[0] == [1, 1, 1, 1]
            else:
                # The dd_interval_factor limits the solutions per direction to 4.
                assert params[f'{solve_type}_solutions_per_direction'] == [[1, 2, 2, 4]]
                # The dd_smoothness factor limits the minimum value to 1.0/2.0.
                # The inner list is an np.array instead of a plain list now.
                assert (smoothness_dd_factors[0] == [1.0, 1.0/1.5, 1.0/2.0, 1.0/2.0]).all()


    @pytest.mark.parametrize("chunk_size, expected_n_chunks", [(1, 6), (2, 3), (4, 2), (42, 1)])
    def test_set_calibration_parameters_time_chunking(self, observation, test_ms, chunk_size, expected_n_chunks):
        """Test set_calibration_parameters() with time chunking enabled."""

        # Set expected values for the get_chunk_size() call.
        dd_interval_factor = 5
        cluster_specific_parameters = "mock cluster specific parameters"
        n_observations = 42

        parset = {
            "calibration_specific": {
                "fast_freqstep_hz": observation.channelwidth,
                "medium_freqstep_hz": observation.channelwidth,
                "slow_freqstep_hz": observation.channelwidth,
                "fulljones_freqstep_hz": observation.channelwidth,
                "dd_interval_factor": dd_interval_factor,
                "dd_smoothness_factor": 1,
                "fast_smoothnessreffrequency": None,
                "medium_smoothnessreffrequency": None,
            },
            "cluster_specific": cluster_specific_parameters,
        }
        calibrator_fluxes = [1.0]

        with mock.patch("rapthor.lib.observation.get_chunk_size") as mock_get_chunk_size:
            mock_get_chunk_size.return_value = chunk_size
            observation.set_calibration_parameters(
                parset,
                n_observations,
                calibrator_fluxes,
                observation.timepersample,
                observation.timepersample,
                observation.timepersample,
                observation.timepersample,
                chunk_by_time=True,
            )
            mock_get_chunk_size.assert_called_once_with(
                cluster_specific_parameters, observation.numsamples, n_observations,
                dd_interval_factor)

        assert observation.ntimechunks == expected_n_chunks

        params = observation.parameters

        assert len(params['timechunk_filename']) == expected_n_chunks
        assert len(params['predict_di_output_filename']) == expected_n_chunks
        assert len(params['starttime']) == expected_n_chunks
        assert len(params['ntimes']) == expected_n_chunks

        for i in range(expected_n_chunks):
            assert params['timechunk_filename'][i] == test_ms
            assert params['predict_di_output_filename'][i] is None

        start_times = [
            '29Mar2013/13:59:52.907',
            '29Mar2013/14:00:02.921',
            '29Mar2013/14:00:12.935',
            '29Mar2013/14:00:22.949',
            '29Mar2013/14:00:32.962',
            '29Mar2013/14:00:42.976',
        ]

        if chunk_size == 1: # Expecting 6 chunks of 1 time sample each.
            assert params['starttime'] == start_times
            assert params['ntimes'] == [1, 1, 1, 1, 1, 0]
        elif chunk_size == 2: # Expecting 3 chunks of 2 time samples each.
            assert params['starttime'] == [
                start_times[0],
                start_times[chunk_size],
                start_times[chunk_size*2],
            ]
            assert params['ntimes'] == [chunk_size, chunk_size, 0]
        elif chunk_size == 4: # Expecting 2 chunks, with 4 and 2 time samples.
            assert params['starttime'] == [start_times[0], start_times[chunk_size]]
            assert params['ntimes'] == [chunk_size, 0]
        elif chunk_size == 42: # Expecting 1 chunk with all time samples.
            assert params['starttime'] == [start_times[0]]
            assert params['ntimes'] == [0]
        else:
            assert False, f"Error in test: invalid chunk_size value: {chunk_size}"


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

"""
Test cases for the `rapthor.lib.observation` module.
"""

import math
from logging import Logger
from unittest import mock

import numpy as np
import pytest

from rapthor.execution.pipeline.lifecycle import chunk_observations


@pytest.fixture
def calibration_parset(observation):
    """Create a basic parset dictionary for testing calibration."""
    return {
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


class TestObservation:
    """
    Test cases for the Observation class.
    """

    hba_reference_frequency = 144e6  # Hardcoded value for HBA antennas.

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
        assert observation.antenna == "HBA"
        assert np.isclose(observation.channelwidth, 24414.0625)
        assert observation.numchannels == 8

    def test_copy_returns_independent_observation_with_logger(self, observation):
        copied = observation.copy()

        assert copied is not observation
        assert copied.ms_filename == observation.ms_filename
        assert copied.parameters == observation.parameters
        assert isinstance(copied.log, Logger)
        assert copied.log.name == observation.log.name

        copied.parameters["new_parameter"] = "copy only"

        assert "new_parameter" not in observation.parameters

    def test_scan_ms_populates_measurement_set_metadata(self, observation):
        """scan_ms records the MS time, frequency, pointing, and station metadata."""
        assert bool(observation.startsat_startofms) is True
        assert bool(observation.goesto_endofms) is True
        assert observation.infix == ""
        assert np.isclose(observation.timepersample, 10.0139008)
        assert np.isclose(observation.referencefreq, 134375000.0)
        assert np.isclose(observation.startfreq, 134288024.90234375)
        assert np.isclose(observation.endfreq, 134458923.33984375)
        assert observation.channels_are_regular is True
        assert np.isclose(observation.ra, 24.422081000000002)
        assert np.isclose(observation.dec, 33.15975900000001)
        assert list(observation.stations) == [
            "CS001HBA0",
            "CS002HBA0",
            "CS002HBA1",
            "CS004HBA1",
            "RS106HBA",
            "RS208HBA",
            "RS305HBA",
            "RS307HBA",
        ]
        assert np.isclose(observation.diam, 31.262533139844095)
        assert np.isclose(observation.mean_el_rad, 1.1529779067059374)
        assert np.isclose(observation.high_el_starttime, observation.starttime)
        assert np.isclose(observation.high_el_endtime, observation.endtime)

    def check_single_timechunk(self, observation, test_ms):
        """Check if the observation has a single time chunk."""
        assert observation.ntimechunks == 1
        params = observation.parameters
        assert params["timechunk_filename"] == [test_ms]
        assert params["predict_di_output_filename"] == [None]
        assert params["starttime"] == ["29Mar2013/13:59:52.907"]
        assert params["ntimes"] == [0]

    def test_set_calibration_parameters_basic(self, observation, test_ms, calibration_parset):
        """
        Basic set_calibration_parameters() test.
        All solution intervals and all DD factors are 1.
        """
        n_observations = -1  # Not used in this test, since chunk_by_time is False.
        calibrator_fluxes = [1.0]
        observation.set_calibration_parameters(
            calibration_parset,
            n_observations,
            calibrator_fluxes,
            observation.timepersample,
            observation.timepersample,
            observation.timepersample,
            observation.timepersample,
        )

        self.check_single_timechunk(observation, test_ms)

        params = observation.parameters

        for solve_type in ["fast", "medium", "slow", "fulljones"]:
            assert params[f"solint_{solve_type}_timestep"] == [1]
            assert params[f"solint_{solve_type}_freqstep"] == [1]

        assert params["bda_maxinterval"] == [observation.timepersample]
        assert params["bda_minchannels"] == [observation.numchannels]

        for solve_type in ["fast", "medium", "slow"]:
            assert params[f"{solve_type}_solutions_per_direction"] == [[1]]
            assert params[f"{solve_type}_smoothness_dd_factors"] == [[1]]

        assert params["fast_smoothnessreffrequency"] == [self.hba_reference_frequency]
        assert params["medium_smoothnessreffrequency"] == [self.hba_reference_frequency]

    def test_set_calibration_parameters_solint(self, observation, test_ms):
        """
        Test set_calibration_parameters() with custom solutionn interval settings.
        All solution intervals and all DD factors are larger than 1.
        """

        time_factor = {
            "fast": 3,
            "medium": 5,
            "slow": 7,
            "fulljones": 9,
        }

        freq_factor = {
            "fast": 2,
            "medium": 4,
            "slow": 6,
            "fulljones": 8,
        }
        expected_freq_factor = {
            "fast": 2,
            "medium": 4,
            "slow": 8,  # 6 does not divide 'numchannels', so should be rounded up to 8.
            "fulljones": 8,
        }

        dd_interval_factor = 10
        dd_smoothness_factor = 11

        parset = {
            "calibration_specific": {
                "fast_freqstep_hz": freq_factor["fast"] * observation.channelwidth,
                "medium_freqstep_hz": freq_factor["medium"] * observation.channelwidth,
                "slow_freqstep_hz": freq_factor["slow"] * observation.channelwidth,
                "fulljones_freqstep_hz": freq_factor["fulljones"] * observation.channelwidth,
                "dd_interval_factor": dd_interval_factor,
                "dd_smoothness_factor": dd_smoothness_factor,
                "fast_smoothnessreffrequency": None,
                "medium_smoothnessreffrequency": None,
            }
        }

        n_observations = -1  # Not used in this test, since chunk_by_time is False.
        calibrator_fluxes = [1.0]
        observation.set_calibration_parameters(
            parset,
            n_observations,
            calibrator_fluxes,
            time_factor["fast"] * observation.timepersample,
            time_factor["medium"] * observation.timepersample,
            time_factor["slow"] * observation.timepersample,
            time_factor["fulljones"] * observation.timepersample,
        )

        self.check_single_timechunk(observation, test_ms)

        params = observation.parameters

        for solve_type in ["fast", "medium", "slow", "fulljones"]:
            expected_timestep = time_factor[solve_type]
            if solve_type != "fulljones":
                expected_timestep *= dd_interval_factor
            expected_freqstep = expected_freq_factor[solve_type]
            assert params[f"solint_{solve_type}_timestep"] == [expected_timestep]
            assert params[f"solint_{solve_type}_freqstep"] == [expected_freqstep]

        # The fast solution interval is the smallest, so bda_maxinterval and
        # bda_minchannels should be based on that.
        expected_bda_maxinterval = time_factor["fast"] * observation.timepersample
        expected_bda_minchannels = observation.numchannels / expected_freq_factor["fast"]
        assert params["bda_maxinterval"] == [expected_bda_maxinterval]
        assert params["bda_minchannels"] == [expected_bda_minchannels]

    @pytest.mark.parametrize("generate_screens", [True, False])
    def test_set_calibration_parameters_multiple_fluxes(
        self, observation, test_ms, calibration_parset, generate_screens
    ):
        """Test set_calibration_parameters() with multiple calibrator fluxes."""
        parset = calibration_parset
        parset["calibration_specific"]["dd_interval_factor"] = 4
        parset["calibration_specific"]["dd_smoothness_factor"] = 2

        n_observations = -1  # Not used in this test, since chunk_by_time is False.
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

        for solve_type in ["fast", "medium", "slow"]:
            solutions_per_direction = params[f"{solve_type}_solutions_per_direction"]
            smoothness_dd_factors = params[f"{solve_type}_smoothness_dd_factors"]

            # There is one list for each time chunk.
            assert len(solutions_per_direction) == 1
            assert len(smoothness_dd_factors) == 1

            if generate_screens:
                # When generate_screens is True, the dd factors are always 1.
                # The solutions per direction and smoothness factors are then also all 1.
                assert solutions_per_direction[0] == [1, 1, 1, 1]
                assert smoothness_dd_factors[0] == [1, 1, 1, 1]
            else:
                # The dd_interval_factor limits the solutions per direction to 4.
                assert solutions_per_direction[0] == [1, 2, 2, 4]
                # The dd_smoothness factor limits the minimum value to 1.0/2.0.
                # The inner list is an np.array instead of a plain list now.
                assert (smoothness_dd_factors[0] == [1.0, 1.0 / 1.5, 1.0 / 2.0, 1.0 / 2.0]).all()

    @pytest.mark.parametrize("chunk_size, expected_n_chunks", [(1, 6), (2, 3), (4, 2), (42, 1)])
    def test_set_calibration_parameters_time_chunking(
        self, observation, test_ms, calibration_parset, chunk_size, expected_n_chunks
    ):
        """Test set_calibration_parameters() with time chunking enabled."""

        # Set expected values for the get_chunk_size() call.
        dd_interval_factor = 5
        cluster_specific_parameters = "mock cluster specific parameters"
        n_observations = 42

        parset = calibration_parset
        parset["calibration_specific"]["dd_interval_factor"] = dd_interval_factor
        parset["cluster_specific"] = cluster_specific_parameters

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
                cluster_specific_parameters,
                observation.numsamples,
                n_observations,
                dd_interval_factor,
            )

        assert observation.ntimechunks == expected_n_chunks

        params = observation.parameters

        assert len(params["timechunk_filename"]) == expected_n_chunks
        assert len(params["predict_di_output_filename"]) == expected_n_chunks
        assert len(params["starttime"]) == expected_n_chunks
        assert len(params["ntimes"]) == expected_n_chunks

        for i in range(expected_n_chunks):
            assert params["timechunk_filename"][i] == test_ms
            assert params["predict_di_output_filename"][i] is None

        start_times = [
            "29Mar2013/13:59:52.907",
            "29Mar2013/14:00:02.921",
            "29Mar2013/14:00:12.935",
            "29Mar2013/14:00:22.949",
            "29Mar2013/14:00:32.962",
            "29Mar2013/14:00:42.976",
        ]

        if chunk_size == 1:  # Expecting 6 chunks of 1 time sample each.
            assert params["starttime"] == start_times
            assert params["ntimes"] == [1, 1, 1, 1, 1, 0]
        elif chunk_size == 2:  # Expecting 3 chunks of 2 time samples each.
            assert params["starttime"] == [
                start_times[0],
                start_times[chunk_size],
                start_times[chunk_size * 2],
            ]
            assert params["ntimes"] == [chunk_size, chunk_size, 0]
        elif chunk_size == 4:  # Expecting 2 chunks, with 4 and 2 time samples.
            assert params["starttime"] == [start_times[0], start_times[chunk_size]]
            assert params["ntimes"] == [chunk_size, 0]
        elif chunk_size == 42:  # Expecting 1 chunk with all time samples.
            assert params["starttime"] == [start_times[0]]
            assert params["ntimes"] == [0]
        else:
            assert False, f"Error in test: invalid chunk_size value: {chunk_size}"

        for solve_type in ["fast", "medium", "slow"]:
            assert (
                params[f"solint_{solve_type}_timestep"] == [dd_interval_factor] * expected_n_chunks
            )
            assert params[f"solint_{solve_type}_freqstep"] == [1] * expected_n_chunks
            assert params[f"{solve_type}_smoothness_dd_factors"] == [[1]] * expected_n_chunks

        # fulljones time steps are not multiplied by dd_interval_factor
        assert params["solint_fulljones_timestep"] == [1] * expected_n_chunks
        assert params["solint_fulljones_freqstep"] == [1] * expected_n_chunks

        assert (
            params["fast_smoothnessreffrequency"]
            == [self.hba_reference_frequency] * expected_n_chunks
        )
        assert (
            params["medium_smoothnessreffrequency"]
            == [self.hba_reference_frequency] * expected_n_chunks
        )

    def test_set_calibration_parameters_smoothness_ref_frequency(
        self, observation, test_ms, calibration_parset
    ):
        """Test set_calibration_parameters() with custom smoothness reference frequencies."""
        fast_reference_frequency = 42e6
        medium_reference_frequency = 43e6

        parset = calibration_parset
        parset["calibration_specific"]["fast_smoothnessreffrequency"] = fast_reference_frequency
        parset["calibration_specific"]["medium_smoothnessreffrequency"] = medium_reference_frequency

        n_observations = -1  # Not used in this test, since chunk_by_time is False.
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
        assert params["fast_smoothnessreffrequency"] == [fast_reference_frequency]
        assert params["medium_smoothnessreffrequency"] == [medium_reference_frequency]

    def test_set_prediction_parameters(self, observation, test_ms):
        observation.set_prediction_parameters("sector_1", ["patch_a", "patch_b"])

        assert observation.parameters["ms_filename"] == test_ms
        assert observation.parameters["ms_model_filename"] == "test.ms.sector_1_modeldata"
        assert observation.parameters["ms_subtracted_filename"] == "test.ms.sector_1"
        assert observation.ms_subtracted_filename == "test.ms.sector_1"
        assert observation.ms_field == "test.ms_field"
        assert observation.ms_predict_di == "test.ms.sector_1_di.ms"
        assert observation.parameters["patch_names"] == ["patch_a", "patch_b"]
        assert observation.parameters["predict_starttime"] == "29Mar2013/13:59:52.907"
        assert observation.parameters["predict_ntimes"] == 0

    @pytest.mark.parametrize(
        "preapply_dd_solutions, expected_image_timestep",
        [(False, 2), (True, 3)],
    )
    def test_set_imaging_parameters(
        self, observation, test_ms, preapply_dd_solutions, expected_image_timestep
    ):
        """Preapplied DD solutions allow more time averaging during imaging."""
        observation.set_imaging_parameters(
            sector_name="sector_1",
            cellsize_arcsec=3.0,
            max_peak_smearing=0.1,
            width_ra=1.2,
            width_dec=0.8,
            # Without preapplied DD solutions, imaging must preserve the fast-solve
            # cadence so solutions can still be applied accurately. That caps the
            # image timestep at 20 s / 10.0139008 s ~= 2 slots. With preapplied DD
            # solutions, the smearing limit is the active constraint, giving 3 slots.
            solve_fast_timestep=20.0,
            solve_slow_timestep=60.0,
            solve_slow_freqstep=observation.channelwidth * 4,
            preapply_dd_solutions=preapply_dd_solutions,
        )

        assert observation.parameters["ms_filename"] == test_ms
        assert observation.parameters["ms_prep_filename"] == "test.sector_1_prep.ms"
        assert observation.parameters["image_freqstep"] == 4
        assert observation.parameters["image_timestep"] == expected_image_timestep
        assert observation.parameters["image_bda_maxinterval"] == 6

    @pytest.mark.parametrize(
        "freqstep, expected",
        [(1, 1), (2, 2), (3, 4), (5, 4), (6, 8), (7, 8), (8, 8), (9, 8)],
    )
    def test_get_nearest_freqstep(self, observation, freqstep, expected):
        assert observation.get_nearest_freqstep(freqstep) == expected

    def test_get_target_timewidth(self, observation):
        """The time smearing helper should match the analytic Rapthor formula."""
        delta_theta = 1.0
        resolution = 0.01
        reduction_factor = 0.9

        expected_delta_time = np.sqrt(
            (1.0 - reduction_factor) / (1.22e-9 * (delta_theta / resolution) ** 2.0)
        )

        assert np.isclose(
            observation.get_target_timewidth(delta_theta, resolution, reduction_factor),
            expected_delta_time,
        )

    def test_get_bandwidth_smearing_factor(self, observation):
        """The bandwidth smearing helper should match the analytic Rapthor formula."""
        freq = 150.0
        delta_freq = 1.0
        delta_theta = 1.0
        resolution = 0.01
        beta = (delta_freq / freq) * (delta_theta / resolution)
        gamma = 2 * (np.log(2) ** 0.5)
        expected_reduction_factor = ((np.pi**0.5) / (gamma * beta)) * (math.erf(beta * gamma / 2.0))

        assert np.isclose(
            observation.get_bandwidth_smearing_factor(freq, delta_freq, delta_theta, resolution),
            expected_reduction_factor,
        )

    def test_get_target_bandwidth(self, observation):
        """The target bandwidth is the first 10% step below the requested reduction."""
        freq = 150.0
        delta_theta = 1.0
        resolution = 0.01
        reduction_factor = 0.9

        target_bandwidth = observation.get_target_bandwidth(
            freq, delta_theta, resolution, reduction_factor
        )
        previous_step = target_bandwidth / 1.1

        assert (
            observation.get_bandwidth_smearing_factor(
                freq, target_bandwidth, delta_theta, resolution
            )
            <= reduction_factor
        )
        assert (
            observation.get_bandwidth_smearing_factor(freq, previous_step, delta_theta, resolution)
            > reduction_factor
        )

    @pytest.mark.parametrize(
        "solints_seconds, solve_max_factor, expected_max",
        [
            # timepersample in test MS is 10.0139008 seconds
            ([20, 120, 600, 600], 1, 60),  # Default case
            ([20, 120, 600, 600], 2, 120),  # Increased solve_max_factor
            ([10, 10, 10, 10], 1, 1),  # All solints the same
            ([0.5, 0.5, 0.5, 0.5], 1, 1),  # All solints < 1 should return 1
            (
                [10.0139008, 10.0139008, 10.0139008, 10.0139008],
                1,
                1,
            ),  # Exact match to timepersample
            ([10, 10.014, 10, 10], 1, 2),  # Max solint above timepersample
        ],
    )
    def test_get_max_solint_timesteps(
        self, observation, solints_seconds, solve_max_factor, expected_max
    ):
        """
        Test the get_max_solint_timesteps method of the Observation class.
        """
        max_solint = observation.get_max_solint_timesteps(solints_seconds, solve_max_factor)
        assert max_solint == expected_max
        assert isinstance(max_solint, int)


@pytest.mark.parametrize(
    "max_nodes, data_fraction, num_chunks",
    [(1, 1.0, 1), (1, 0.2, 1), (3, 1.0, 3), (3, 0.5, 3), (5, 1.0, 5)],
)
def test_chunking_by_time(observation, field, monkeypatch, max_nodes, data_fraction, num_chunks):
    if data_fraction < 1.0:
        pytest.xfail(
            "Number of chunks currently becomes larger than the number of nodes when data_fraction is less than 1.0."
        )

    observation.starttime = 4453731483.92
    observation.endtime = 4453738676.08
    # Note that high_el_starttime is much larger than starttime. When data_fraction < 1.0 in this
    # test, chunk_observations creates chunks between high_el_starttime and high_el_endtime.
    observation.high_el_starttime = 4453732884.0
    observation.high_el_endtime = 4453738676.0
    observation.numsamples = 900
    observation.timepersample = 8.0
    observation.data_fraction = data_fraction

    def scan_ms(self):
        self.startsat_startofms = True
        self.goesto_endofms = False
        return None

    monkeypatch.setattr("rapthor.lib.observation.Observation.scan_ms", scan_ms)

    steps = [
        {
            "do_calibrate": True,
            "fast_timestep_sec": 20,
            "medium_timestep_sec": 120,
            "slow_timestep_sec": 600,
            "fulljones_timestep_sec": 600,
        },
    ]
    field.parset["cluster_specific"]["max_nodes"] = max_nodes
    field.full_observations = [observation]

    chunk_observations(field, steps, data_fraction)

    assert len(field.observations) == num_chunks

    if data_fraction == 1.0:
        assert field.observations[0].starttime == observation.starttime
        assert field.observations[-1].endtime == observation.endtime
    else:
        assert field.observations[0].starttime == observation.high_el_starttime
        assert field.observations[-1].endtime == observation.high_el_endtime

    if data_fraction == 1.0:
        # With a data fraction of 1.0 there are no gaps between the chunks.
        gap_time = 0 * observation.timepersample
    elif data_fraction == 0.5:
        # chunk_observations only chunks the time between high_el_starttime and high_el_endtime,
        # which has 725 samples. The chunks consume 6*75=450 samples, which leaves 275 samples
        # for 5 gaps, thus 55 samples per gap.
        gap_time = 55 * observation.timepersample
    elif data_fraction == 0.2:
        # With a data fraction of 0.2, there are again 725 samples between high_el_starttime and
        # high_el_endtime. There are two chunks of 75 samples and a single gap of 575 samples.
        gap_time = 575 * observation.timepersample

    for i in range(len(field.observations) - 1):
        # Since the start time and end time are mid points, add one time sample.
        # The tolerance is 5 % of the sample time, since Rapthor may adjust the start and
        # end times for avoiding rounding errors.
        assert np.isclose(
            field.observations[i].endtime + observation.timepersample + gap_time,
            field.observations[i + 1].starttime,
            rtol=0,
            atol=observation.timepersample * 0.05,
        )

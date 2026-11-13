"""
Test cases for the `rapthor.lib.observation` module.
"""

import math
import warnings
from logging import Logger
from unittest import mock

import casacore.tables as pt
import numpy as np
import pytest

from rapthor.execution.pipeline.lifecycle import chunk_observations
from rapthor.lib.observation import Observation


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

    @staticmethod
    def set_averaging_test_parameters(
        observation,
        calibration_parset,
        *,
        timesteps,
        freqsteps,
        calibrator_fluxes=(1.0,),
        dd_interval_factor=1,
    ):
        """Set fast, medium and slow solution intervals in time slots and channels."""
        settings = calibration_parset["calibration_specific"]
        settings.update(
            {
                "average_visibilities": True,
                "max_peak_smearing": 0.15,
                "dd_interval_factor": dd_interval_factor,
            }
        )
        for solve_type in ["fast", "medium", "slow"]:
            settings[f"{solve_type}_freqstep_hz"] = freqsteps[solve_type] * observation.channelwidth
        calibration_parset["imaging_specific"] = {"cellsize_arcsec": 1.5}
        observation.set_calibration_parameters(
            calibration_parset,
            -1,  # not used, since chunk_by_time is False
            list(calibrator_fluxes),
            timesteps["fast"] * observation.timepersample,
            timesteps["medium"] * observation.timepersample,
            timesteps["slow"] * observation.timepersample,
            observation.timepersample,
            target_flux=min(calibrator_fluxes),
        )

    def test_set_calibration_averaging_parameters_disabled(self, observation, calibration_parset):
        """Without average_visibilities, the data are not averaged before calibration."""
        steps = {"fast": 6, "medium": 6, "slow": 6}
        freqsteps = {"fast": 8, "medium": 8, "slow": 8}
        self.set_averaging_test_parameters(
            observation, calibration_parset, timesteps=steps, freqsteps=freqsteps
        )
        calibration_parset["calibration_specific"]["average_visibilities"] = False

        observation.set_calibration_averaging_parameters(
            calibration_parset, 0.3, ["fast", "medium", "slow"]
        )

        assert observation.parameters["calibration_timestep"] == [1]
        assert observation.parameters["calibration_freqstep"] == [1]

    @pytest.mark.parametrize(
        "calibration_radius, expected_timestep, expected_freqstep",
        [
            # Smearing allows 3.3 time slots and 14 channels; the solution
            # intervals allow at most 8 channels.
            pytest.param(0.3, 3, 8, id="time-smearing-limited"),
            # Smearing allows 1.4 time slots and 5.9 channels; 5 channels would
            # not divide the 8 channels of the data and solution intervals.
            pytest.param(0.7, 1, 4, id="frequency-smearing-limited"),
            # A single calibrator at the phase center gives no smearing limit.
            pytest.param(0.0, 6, 8, id="calibrator-at-phase-center"),
        ],
    )
    def test_set_calibration_averaging_parameters_limits_smearing(
        self,
        observation,
        calibration_parset,
        calibration_radius,
        expected_timestep,
        expected_freqstep,
    ):
        """The averaging is as large as the smearing at the furthest calibrator allows."""
        steps = {"fast": 6, "medium": 6, "slow": 6}
        freqsteps = {"fast": 8, "medium": 8, "slow": 8}
        self.set_averaging_test_parameters(
            observation, calibration_parset, timesteps=steps, freqsteps=freqsteps
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error")  # e.g. a division by zero at radius 0
            observation.set_calibration_averaging_parameters(
                calibration_parset, calibration_radius, ["fast", "medium", "slow"]
            )

        assert observation.parameters["calibration_timestep"] == [expected_timestep]
        assert observation.parameters["calibration_freqstep"] == [expected_freqstep]

    @pytest.mark.parametrize(
        "solution_intervals, expected_timestep, expected_freqstep",
        [
            # Every averaged step must fit the 4 and 6 time slot and the 2 and 8
            # channel solution intervals exactly.
            pytest.param(["fast", "slow"], 2, 2, id="fast-and-slow"),
            # Solves that are not done do not limit the averaging.
            pytest.param(["slow"], 6, 8, id="slow-only"),
            # Applied full-Jones solutions have 1 time slot and 1 channel intervals.
            pytest.param(["slow", "fulljones"], 1, 1, id="fulljones"),
        ],
    )
    def test_set_calibration_averaging_parameters_fits_solution_intervals(
        self,
        observation,
        calibration_parset,
        solution_intervals,
        expected_timestep,
        expected_freqstep,
    ):
        """The averaging divides the solution interval of every solve that is done."""
        self.set_averaging_test_parameters(
            observation,
            calibration_parset,
            timesteps={"fast": 4, "medium": 5, "slow": 6},
            freqsteps={"fast": 2, "medium": 1, "slow": 8},
        )

        # A radius of 0.05 degrees allows 12 time slots and 82 channels.
        observation.set_calibration_averaging_parameters(
            calibration_parset, 0.05, solution_intervals
        )

        assert observation.parameters["calibration_timestep"] == [expected_timestep]
        assert observation.parameters["calibration_freqstep"] == [expected_freqstep]

    def test_set_calibration_averaging_parameters_fits_the_observation(
        self, observation, calibration_parset
    ):
        """The averaging never exceeds the 6 time slots of the observation."""
        self.set_averaging_test_parameters(
            observation,
            calibration_parset,
            timesteps={"fast": 10, "medium": 10, "slow": 10},
            freqsteps={"fast": 8, "medium": 8, "slow": 8},
        )

        # A radius of 0.05 degrees allows 12 time slots, so 10 would be chosen
        observation.set_calibration_averaging_parameters(
            calibration_parset, 0.05, ["fast", "medium", "slow"]
        )

        assert observation.parameters["calibration_timestep"] == [5]

    def test_set_calibration_averaging_parameters_limits_bda_channels(
        self, observation, calibration_parset
    ):
        """BDA cannot keep more channels than the averaged data have."""
        self.set_averaging_test_parameters(
            observation,
            calibration_parset,
            timesteps={"fast": 6, "medium": 6, "slow": 6},
            freqsteps={"fast": 1, "medium": 8, "slow": 8},
        )
        # The 1-channel fast interval keeps all 8 channels in BDA
        assert observation.parameters["bda_minchannels"] == [8]

        observation.set_calibration_averaging_parameters(calibration_parset, 0.05, ["medium"])

        assert observation.parameters["calibration_freqstep"] == [8]
        assert observation.parameters["bda_minchannels"] == [1]

    def test_set_calibration_averaging_parameters_fits_direction_dependent_intervals(
        self, observation, calibration_parset
    ):
        """The averaging divides the shorter intervals of bright calibrators."""
        self.set_averaging_test_parameters(
            observation,
            calibration_parset,
            timesteps={"fast": 3, "medium": 3, "slow": 3},
            freqsteps={"fast": 8, "medium": 8, "slow": 8},
            calibrator_fluxes=(1.0, 2.0),
            dd_interval_factor=2,
        )
        # The 6 time slot interval has 2 solutions for the bright calibrator.
        assert observation.parameters["solint_fast_timestep"] == [6]
        assert observation.parameters["fast_solutions_per_direction"] == [[1, 2]]

        observation.set_calibration_averaging_parameters(calibration_parset, 0.05, ["fast"])

        assert observation.parameters["calibration_timestep"] == [3]

    @pytest.mark.parametrize(
        "target_fulljones_freqstep, expected_freqstep",
        [
            pytest.param(None, 8, id="parset-value"),
            pytest.param(2, 2, id="supplied-solutions"),
        ],
    )
    def test_set_calibration_averaging_parameters_fits_supplied_fulljones_solutions(
        self, observation, calibration_parset, target_fulljones_freqstep, expected_freqstep
    ):
        """The frequency grid of supplied full-Jones solutions limits the averaging."""
        settings = calibration_parset["calibration_specific"]
        settings.update(
            {
                "average_visibilities": True,
                "max_peak_smearing": 0.15,
                "fulljones_freqstep_hz": 8 * observation.channelwidth,
            }
        )
        calibration_parset["imaging_specific"] = {"cellsize_arcsec": 1.5}
        if target_fulljones_freqstep is not None:
            target_fulljones_freqstep *= observation.channelwidth
        observation.set_calibration_parameters(
            calibration_parset,
            -1,  # not used, since chunk_by_time is False
            [1.0],
            observation.timepersample,
            observation.timepersample,
            observation.timepersample,
            observation.timepersample,
            target_fulljones_freqstep=target_fulljones_freqstep,
        )

        # A radius of 0.05 degrees allows 82 channels, so the 8 channels of the
        # parset value would be chosen without the supplied solutions
        observation.set_calibration_averaging_parameters(calibration_parset, 0.05, ["fulljones"])

        assert observation.parameters["solint_fulljones_freqstep"] == [expected_freqstep]
        assert observation.parameters["calibration_freqstep"] == [expected_freqstep]

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
        "preapply_dd_solutions, expected_image_timestep, expected_image_freqstep",
        [(False, 2, 1), (True, 2, 8)],
    )
    def test_set_imaging_parameters(
        self,
        observation,
        test_ms,
        preapply_dd_solutions,
        expected_image_timestep,
        expected_image_freqstep,
    ):
        """Preapplied DD solutions allow more time/frequency averaging."""
        observation.set_imaging_parameters(
            sector_name="sector_1",
            cellsize_arcsec=3.0,
            max_peak_smearing=0.1,
            width_ra=1.2,
            width_dec=0.8,
            # Without preapplied DD solutions, imaging must preserve the fast-solve
            # cadence so solutions can still be applied accurately. That caps the
            # image timestep at the solution interval of 20 s / 10.0139008 s =
            # 1.997, which the solves round to 2 slots. With preapplied DD
            # solutions, the smearing limit of 27.0 s is the active constraint,
            # which is rounded down to 2 slots.
            min_solve_timestep=20.0,
            min_solve_timestep_short_baselines=60.0,
            min_solve_freqstep=observation.channelwidth,
            min_solve_freqstep_short_baselines=observation.channelwidth * 4,
            preapply_dd_solutions=preapply_dd_solutions,
        )

        assert observation.parameters["ms_filename"] == test_ms
        assert observation.parameters["ms_prep_filename"] == "test.sector_1_prep.ms"
        assert observation.parameters["image_freqstep"] == expected_image_freqstep
        assert observation.parameters["image_timestep"] == expected_image_timestep
        assert observation.parameters["image_bda_maxinterval"] == 6
        assert observation.parameters["image_bda_minchannels"] == 2

    def test_set_imaging_parameters_rounds_the_averaging_steps_down(self, observation):
        """The averaging steps never exceed the widths allowed by the smearing limit."""
        # At 150 MHz, 1.5 degrees from the center, with 1.5 arcsec pixels and 15%
        # smearing, the targets are 75.4 kHz and 6.7 s. The nearest steps of 2
        # channels (97.7 kHz) and 3 time slots (6 s) would smear the peak by 18%
        observation.referencefreq = 150e6
        observation.channelwidth = 48828.125
        observation.numchannels = 40
        observation.timepersample = 2.0
        observation.numsamples = 1800
        max_peak_smearing = 0.15
        cellsize_arcsec = 1.5
        radius_deg = 1.5

        observation.set_imaging_parameters(
            sector_name="sector_1",
            cellsize_arcsec=cellsize_arcsec,
            max_peak_smearing=max_peak_smearing,
            width_ra=2 * radius_deg,
            width_dec=2 * radius_deg,
            min_solve_timestep=1e9,
            min_solve_timestep_short_baselines=1e9,
            min_solve_freqstep=1e9,
            min_solve_freqstep_short_baselines=1e9,
            preapply_dd_solutions=True,
        )

        freqstep = observation.parameters["image_freqstep"]
        timestep = observation.parameters["image_timestep"]
        assert (freqstep, timestep) == (1, 3)
        # Each of time and bandwidth smearing gets half of the allowed reduction
        reduction_factor = np.sqrt(1.0 - max_peak_smearing)
        resolution = 3.0 * cellsize_arcsec / 3600.0
        assert (
            observation.get_bandwidth_smearing_factor(
                observation.referencefreq,
                freqstep * observation.channelwidth,
                radius_deg,
                resolution,
            )
            >= reduction_factor
        )
        assert timestep * observation.timepersample <= observation.get_target_timewidth(
            radius_deg, resolution, reduction_factor
        )

    def test_get_target_averaging_widths_splits_smearing(self, observation):
        """Time and bandwidth smearing each get half of the allowed peak reduction."""
        delta_theta = 0.6
        resolution = 3.0 * 3.0 / 3600.0  # three 3-arcsec pixels across the beam
        reduction_factor = np.sqrt(1.0 - 0.1)

        timewidth, bandwidth = observation.get_target_averaging_widths(delta_theta, 3.0, 0.1)

        assert timewidth == pytest.approx(
            observation.get_target_timewidth(delta_theta, resolution, reduction_factor)
        )
        assert bandwidth == pytest.approx(
            observation.get_target_bandwidth(
                observation.referencefreq, delta_theta, resolution, reduction_factor
            )
        )

    def test_get_target_averaging_widths_caps_small_fields(self, observation):
        """Very small fields are capped at 120 s and 2 MHz of averaging."""
        assert observation.get_target_averaging_widths(1e-4, 3.0, 0.1) == (120, 2e6)

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

    @pytest.mark.parametrize(
        "freq, delta_theta, resolution, reduction_factor",
        [
            pytest.param(150.0, 1.0, 0.01, 0.9, id="generic"),
            # HBA at 1, 1.5 and 2.5 degrees with 1.5-arcsec pixels: the widths
            # (about 114, 76 and 45 kHz) are below 0.1% of the frequency
            pytest.param(150e6, 1.0, 4.5 / 3600, np.sqrt(0.85), id="hba-1deg"),
            pytest.param(150e6, 1.5, 4.5 / 3600, np.sqrt(0.85), id="hba-1.5deg"),
            pytest.param(150e6, 2.5, 4.5 / 3600, np.sqrt(0.85), id="hba-2.5deg"),
        ],
    )
    def test_get_target_bandwidth(
        self, observation, freq, delta_theta, resolution, reduction_factor
    ):
        """The target bandwidth is the widest that gives the requested reduction."""
        target_bandwidth = observation.get_target_bandwidth(
            freq, delta_theta, resolution, reduction_factor
        )

        reduction = observation.get_bandwidth_smearing_factor(
            freq, target_bandwidth, delta_theta, resolution
        )
        assert reduction >= reduction_factor
        assert reduction == pytest.approx(reduction_factor, rel=1e-6)

    def test_get_target_bandwidth_at_one_degree_for_hba(self, observation):
        """The smearing limit at 1 degree is about 114 kHz at 150 MHz."""
        target_bandwidth = observation.get_target_bandwidth(150e6, 1.0, 4.5 / 3600, np.sqrt(0.85))

        assert target_bandwidth == pytest.approx(114e3, rel=0.01)

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


@pytest.fixture
def make_chunking_observation(field, monkeypatch):
    """Create evenly spaced sample midpoints in seconds, starting at zero.

    The field must initialize before MS reads are skipped.
    """

    def skip_ms_scan(self):
        self.startsat_startofms = False
        self.goesto_endofms = False

    monkeypatch.setattr(Observation, "scan_ms", skip_ms_scan)

    def make(num_samples, timepersample=10, ms_filename="test.ms"):
        sample_times = [index * timepersample for index in range(num_samples)]
        obs = Observation(ms_filename, starttime=sample_times[0], endtime=sample_times[-1])
        obs.timepersample = timepersample
        obs.numsamples = num_samples
        obs.high_el_starttime = obs.starttime
        obs.high_el_endtime = obs.endtime
        return obs, sample_times

    return make


def samples_in_chunks(sample_times, chunks):
    """List the original sample timestamps selected by each chunk."""
    return [
        [time for time in sample_times if chunk.starttime <= time <= chunk.endtime]
        for chunk in chunks
    ]


@pytest.mark.parametrize(
    "num_samples, max_nodes, expected_chunks",
    [
        pytest.param(6, 1, [[0, 10, 20, 30, 40, 50]], id="one_node_keeps_all_samples"),
        pytest.param(6, 2, [[0, 10, 20], [30, 40, 50]], id="two_equal_chunks"),
        pytest.param(7, 3, [[0, 10], [20, 30], [40, 50, 60]], id="uneven_split_keeps_every_sample"),
        pytest.param(
            8, 3, [[0, 10], [20, 30, 40], [50, 60, 70]], id="uneven_split_with_eight_samples"
        ),
        pytest.param(2, 19, [[0, 10]], id="two_samples_stay_together"),
        pytest.param(3, 19, [[0, 10, 20]], id="avoid_a_one_sample_chunk"),
        pytest.param(4, 19, [[0, 10], [20, 30]], id="at_least_two_samples_per_chunk"),
    ],
)
def test_chunking_full_data(
    field, make_chunking_observation, num_samples, max_nodes, expected_chunks
):
    """Use every sample, sharing work across nodes without making chunks too small."""
    obs, sample_times = make_chunking_observation(num_samples)
    field.full_observations = [obs]
    field.parset["cluster_specific"]["max_nodes"] = max_nodes

    chunk_observations(field, steps=[], data_fraction=1.0)

    assert samples_in_chunks(sample_times, field.observations) == expected_chunks


def test_chunking_full_data_limits_each_observation_to_node_count(field, make_chunking_observation):
    """Both short and long observations split into two chunks for two nodes."""
    short_obs, short_times = make_chunking_observation(6, ms_filename="short.ms")
    long_obs, long_times = make_chunking_observation(12, ms_filename="long.ms")
    field.full_observations = [short_obs, long_obs]
    field.parset["cluster_specific"]["max_nodes"] = 2

    chunk_observations(field, steps=[], data_fraction=1.0)

    assert len(field.observations) == 4
    short_chunks = [chunk for chunk in field.observations if chunk.ms_filename == "short.ms"]
    long_chunks = [chunk for chunk in field.observations if chunk.ms_filename == "long.ms"]
    assert samples_in_chunks(short_times, short_chunks) == [[0, 10, 20], [30, 40, 50]]
    assert samples_in_chunks(long_times, long_chunks) == [
        [0, 10, 20, 30, 40, 50],
        [60, 70, 80, 90, 100, 110],
    ]


@pytest.mark.parametrize(
    "num_samples, solve_time, expected_chunks",
    [
        pytest.param(5, 30, [[0, 10, 20, 30, 40]], id="too_short_for_two_solves"),
        pytest.param(6, 30, [[0, 10, 20], [30, 40, 50]], id="exactly_two_solves"),
        pytest.param(7, 30, [[0, 10, 20], [30, 40, 50, 60]], id="extra_sample_is_kept"),
        pytest.param(6, 31, [[0, 10, 20, 30, 40, 50]], id="solve_rounds_up_to_four_samples"),
        pytest.param(7, 31, [[0, 10, 20, 30, 40, 50, 60]], id="still_too_short_for_two_solves"),
        pytest.param(8, 31, [[0, 10, 20, 30], [40, 50, 60, 70]], id="two_rounded_up_solves"),
        pytest.param(
            12,
            30,
            [[0, 10, 20, 30], [40, 50, 60, 70], [80, 90, 100, 110]],
            id="node_count_limits_calibration_chunks",
        ),
    ],
)
def test_chunking_respects_calibration_duration(
    field, make_chunking_observation, num_samples, solve_time, expected_chunks
):
    """A 30-second solve needs three 10-second samples; a 31-second solve needs four."""
    obs, sample_times = make_chunking_observation(num_samples)
    field.full_observations = [obs]
    field.parset["cluster_specific"]["max_nodes"] = 3
    steps = [{"do_calibrate": True, "fulljones_timestep_sec": solve_time}]

    chunk_observations(field, steps, data_fraction=1.0)

    assert samples_in_chunks(sample_times, field.observations) == expected_chunks


def test_chunking_ignores_solve_duration_when_calibration_is_disabled(
    field, make_chunking_observation
):
    """An unused calibration interval must not prevent splitting across three nodes."""
    obs, sample_times = make_chunking_observation(6)
    field.full_observations = [obs]
    field.parset["cluster_specific"]["max_nodes"] = 3
    steps = [{"do_calibrate": False, "fulljones_timestep_sec": 600}]

    chunk_observations(field, steps, data_fraction=1.0)

    assert samples_in_chunks(sample_times, field.observations) == [[0, 10], [20, 30], [40, 50]]


def test_chunking_calibration_duration_applies_to_each_observation(
    field, make_chunking_observation
):
    """A 60-second solve needs six samples at 10 s/sample, but only three at 20 s/sample."""
    short_interval_obs, short_interval_times = make_chunking_observation(8, 10, "short_interval.ms")
    long_interval_obs, long_interval_times = make_chunking_observation(8, 20, "long_interval.ms")
    field.full_observations = [short_interval_obs, long_interval_obs]
    field.parset["cluster_specific"]["max_nodes"] = 3
    steps = [{"do_calibrate": True, "fulljones_timestep_sec": 60}]

    chunk_observations(field, steps, data_fraction=1.0)

    assert len(field.observations) == 3
    short_interval_chunks = [
        chunk for chunk in field.observations if chunk.ms_filename == "short_interval.ms"
    ]
    long_interval_chunks = [
        chunk for chunk in field.observations if chunk.ms_filename == "long_interval.ms"
    ]
    assert samples_in_chunks(short_interval_times, short_interval_chunks) == [
        [0, 10, 20, 30, 40, 50, 60, 70]
    ]
    assert samples_in_chunks(long_interval_times, long_interval_chunks) == [
        [0, 20, 40, 60],
        [80, 100, 120, 140],
    ]


def test_chunking_different_sample_intervals_without_calibration(field, make_chunking_observation):
    """Each observation keeps at least two samples per chunk, regardless of sample spacing."""
    short_interval_obs, short_interval_times = make_chunking_observation(4, 10, "short_interval.ms")
    long_interval_obs, long_interval_times = make_chunking_observation(5, 40, "long_interval.ms")
    field.full_observations = [short_interval_obs, long_interval_obs]
    field.parset["cluster_specific"]["max_nodes"] = 19

    chunk_observations(field, steps=[], data_fraction=1.0)

    assert len(field.observations) == 4
    short_interval_chunks = [
        chunk for chunk in field.observations if chunk.ms_filename == "short_interval.ms"
    ]
    long_interval_chunks = [
        chunk for chunk in field.observations if chunk.ms_filename == "long_interval.ms"
    ]
    assert samples_in_chunks(short_interval_times, short_interval_chunks) == [[0, 10], [20, 30]]
    assert samples_in_chunks(long_interval_times, long_interval_chunks) == [[0, 40], [80, 120, 160]]


@pytest.mark.parametrize("max_nodes", [1, 2])
@pytest.mark.parametrize("do_calibrate", [False, True])
def test_chunking_partial_data_is_independent_of_node_count(
    field, make_chunking_observation, max_nodes, do_calibrate
):
    """Keep six of ten samples in three 600-second chunks, even with fewer nodes."""
    obs, sample_times = make_chunking_observation(10, timepersample=300)
    field.full_observations = [obs]
    field.parset["cluster_specific"]["max_nodes"] = max_nodes
    steps = [{"do_calibrate": do_calibrate, "fulljones_timestep_sec": 600}]

    chunk_observations(field, steps, data_fraction=0.6)

    assert samples_in_chunks(sample_times, field.observations) == [
        [0, 300],
        [1200, 1500],
        [2400, 2700],
    ]


@pytest.mark.parametrize("num_samples", [10, 11], ids=["equal_gaps", "leftover_sample_at_end"])
def test_chunking_partial_data_leaves_equal_gaps(make_chunking_observation, num_samples):
    """Keep three pairs with two skipped samples per gap; leave any leftover at the end."""
    obs, sample_times = make_chunking_observation(num_samples)
    obs.data_fraction = 0.6

    chunks = obs.chunk_observation(mintime=20)

    assert samples_in_chunks(sample_times, chunks) == [[0, 10], [40, 50], [80, 90]]


def test_chunking_partial_data_does_not_create_an_extra_chunk(make_chunking_observation):
    """Keep six of seven samples as three pairs, leaving the last sample unused."""
    obs, sample_times = make_chunking_observation(7)
    obs.starttime = -0.1
    obs.endtime = 60.1
    obs.high_el_starttime = obs.starttime
    obs.high_el_endtime = obs.endtime
    obs.data_fraction = 0.9

    chunks = obs.chunk_observation(mintime=20)

    assert samples_in_chunks(sample_times, chunks) == [[0, 10], [20, 30], [40, 50]]


@pytest.mark.parametrize(
    "prefer_high_el_periods, expected_chunks",
    [
        pytest.param(True, [[20, 30], [60, 70]], id="select_within_high_elevation_period"),
        pytest.param(False, [[0, 10], [80, 90]], id="select_across_whole_observation"),
    ],
)
def test_chunking_partial_data_can_prefer_high_elevation(
    make_chunking_observation, prefer_high_el_periods, expected_chunks
):
    """Keep four of ten samples, optionally restricting them to the period from 20 to 70 s."""
    obs, sample_times = make_chunking_observation(10)
    obs.high_el_starttime = 20
    obs.high_el_endtime = 70
    obs.data_fraction = 0.4

    chunks = obs.chunk_observation(mintime=20, prefer_high_el_periods=prefer_high_el_periods)

    assert samples_in_chunks(sample_times, chunks) == expected_chunks


@pytest.mark.parametrize(
    "data_fraction, expected_chunks, expected_starts, expected_ends",
    [
        pytest.param(1.0, [[0, 10, 20], [30, 40, 50]], [-0.1, 29.9], [20.1, 50.1], id="full_data"),
        pytest.param(0.8, [[0, 10], [40, 50]], [-0.1, 39.9], [10.1, 50.1], id="partial_data"),
    ],
)
def test_chunking_preserves_timestamp_margin(
    make_chunking_observation, data_fraction, expected_chunks, expected_starts, expected_ends
):
    """Retain the 0.1-second margins used to avoid excluding samples through rounding."""
    obs, sample_times = make_chunking_observation(6)
    obs.starttime = -0.1
    obs.endtime = 50.1
    obs.high_el_starttime = obs.starttime
    obs.high_el_endtime = obs.endtime
    obs.data_fraction = data_fraction

    chunks = obs.chunk_observation(mintime=20, max_chunks=2)

    assert samples_in_chunks(sample_times, chunks) == expected_chunks
    assert [chunk.starttime for chunk in chunks] == pytest.approx(expected_starts)
    assert [chunk.endtime for chunk in chunks] == pytest.approx(expected_ends)


def test_chunking_increases_data_fraction_to_fit_a_calibration_solve(
    field, make_chunking_observation
):
    """A 40-second solve needs four central samples even when only 1% was requested."""
    obs, sample_times = make_chunking_observation(10)
    field.full_observations = [obs]
    field.parset["cluster_specific"]["max_nodes"] = 3
    steps = [{"do_calibrate": True, "fulljones_timestep_sec": 40}]

    chunk_observations(field, steps, data_fraction=0.01)

    assert samples_in_chunks(sample_times, field.observations) == [[30, 40, 50, 60]]


@pytest.mark.parametrize("data_fraction", [0.01, 0.2, 0.4])
def test_chunking_small_data_fraction(observation, field, data_fraction):
    """Retain the third and fourth of six real MS samples, even for tiny data fractions."""
    with pt.table(observation.ms_filename, ack=False) as table:
        sample_times = np.unique(table.getcol("TIME")).tolist()
    assert len(sample_times) == 6
    observation.high_el_starttime = observation.starttime
    observation.high_el_endtime = observation.endtime
    field.full_observations = [observation]
    field.parset["cluster_specific"]["max_nodes"] = 19

    chunk_observations(field, [], data_fraction)

    assert [chunk.numsamples for chunk in field.observations] == [2]
    assert samples_in_chunks(sample_times, field.observations) == [sample_times[2:4]]

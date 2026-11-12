"""
Test cases for the `rapthor.lib.observation` module.
"""

import math
from logging import Logger
from unittest import mock

import casacore.tables as pt
import numpy as np
import pytest

from rapthor.execution.pipeline.lifecycle import chunk_observations, rechunk_observations
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
        calibrator_fluxes = [1.0]
        observation.set_calibration_parameters(
            calibration_parset,
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

        calibrator_fluxes = [1.0]
        observation.set_calibration_parameters(
            parset,
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

        calibrator_fluxes = [1.0, 1.5, 2.5, 5.0]
        observation.set_calibration_parameters(
            parset,
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

    def test_set_calibration_parameters_smoothness_ref_frequency(
        self, observation, test_ms, calibration_parset
    ):
        """Test set_calibration_parameters() with custom smoothness reference frequencies."""
        fast_reference_frequency = 42e6
        medium_reference_frequency = 43e6

        parset = calibration_parset
        parset["calibration_specific"]["fast_smoothnessreffrequency"] = fast_reference_frequency
        parset["calibration_specific"]["medium_smoothnessreffrequency"] = medium_reference_frequency

        calibrator_fluxes = [1.0]
        observation.set_calibration_parameters(
            parset,
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
        "preapply_dd_solutions, expected_image_timestep, expected_image_freqstep",
        [(False, 2, 1), (True, 3, 8)],
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
            # image timestep at 20 s / 10.0139008 s ~= 2 slots. With preapplied DD
            # solutions, the smearing limit is the active constraint, giving 3 slots.
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


@pytest.fixture
def make_chunking_observation(field, monkeypatch):
    """Create evenly spaced sample midpoints in seconds, starting at zero by default.

    The field must initialize before MS reads are skipped.
    """

    def skip_ms_scan(self):
        self.startsat_startofms = False
        self.goesto_endofms = False

    monkeypatch.setattr(Observation, "scan_ms", skip_ms_scan)

    def make(num_samples, timepersample=10, ms_filename="test.ms", starttime=0):
        sample_times = [starttime + index * timepersample for index in range(num_samples)]
        obs = Observation(ms_filename, starttime=sample_times[0], endtime=sample_times[-1])
        obs.timepersample = timepersample
        obs.numsamples = num_samples
        obs.high_el_starttime = obs.starttime
        obs.high_el_endtime = obs.endtime
        return obs, sample_times

    return make


def samples_in_chunks(sample_times, chunks):
    """List the original sample timestamps selected by each chunk (as a scan of the
    chunk selects them, with >= and <=)."""
    return [
        [time for time in sample_times if chunk.starttime <= time <= chunk.endtime]
        for chunk in chunks
    ]


@pytest.mark.parametrize(
    "num_samples, max_nodes, expected_chunks",
    [
        pytest.param(6, 1, [[0, 10, 20, 30, 40, 50]], id="one_node_keeps_all_samples"),
        pytest.param(6, 2, [[0, 10, 20], [30, 40, 50]], id="two_equal_chunks"),
        pytest.param(7, 3, [[0, 10, 20], [30, 40], [50, 60]], id="uneven_split_keeps_every_sample"),
        pytest.param(
            8, 3, [[0, 10, 20], [30, 40, 50], [60, 70]], id="uneven_split_with_eight_samples"
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
        pytest.param(6, 31, [[0, 10, 20], [30, 40, 50]], id="solve_rounds_to_three_samples"),
        pytest.param(8, 31, [[0, 10, 20], [30, 40, 50, 60, 70]], id="partial_solve_at_the_end"),
        pytest.param(
            12,
            30,
            [[0, 10, 20, 30, 40, 50], [60, 70, 80], [90, 100, 110]],
            id="node_count_spreads_whole_solves",
        ),
        pytest.param(
            23,
            30,
            [
                [0, 10, 20, 30, 40, 50, 60, 70, 80],
                [90, 100, 110, 120, 130, 140],
                [150, 160, 170, 180, 190, 200, 210, 220],
            ],
            id="chunks_differ_by_at_most_one_solve",
        ),
    ],
)
def test_chunking_respects_calibration_duration(
    field, make_chunking_observation, num_samples, solve_time, expected_chunks
):
    """Chunks hold whole solution intervals, as DP3 counts them: a 30-second or a
    31-second solve spans three 10-second samples. Only the end of the observation
    holds a partial interval."""
    obs, sample_times = make_chunking_observation(num_samples)
    field.full_observations = [obs]
    field.parset["cluster_specific"]["max_nodes"] = 3
    steps = [
        {
            "do_calibrate": True,
            "calibration_strategy": {"di": ["full_jones"]},
            "fulljones_timestep_sec": solve_time,
        }
    ]

    chunk_observations(field, steps, data_fraction=1.0)

    assert samples_in_chunks(sample_times, field.observations) == expected_chunks


@pytest.mark.parametrize(
    "step, expected_chunk_count",
    [
        pytest.param(
            {
                "calibration_strategy": {"dd": ["fast_phase", "medium_phase"]},
                "fast_timestep_sec": 10,
                "medium_timestep_sec": 20,
                "slow_timestep_sec": 30,
                "fulljones_timestep_sec": 90,
            },
            3,
            id="phase_only_uses_the_medium_interval",
        ),
        pytest.param(
            {
                "calibration_strategy": {"dd": ["fast_phase", "medium_phase"]},
                "fast_timestep_sec": 10,
            },
            3,
            id="missing_interval_uses_the_field_setting",
        ),
        pytest.param(
            {
                # DP3 rounds 25 s to two samples before multiplying by the factor
                "calibration_strategy": {"dd": ["fast_phase", "medium_phase"]},
                "fast_timestep_sec": 10,
                "medium_timestep_sec": 25,
            },
            3,
            id="factor_multiplies_the_interval_in_samples",
        ),
        pytest.param(
            {
                "calibration_strategy": {"di": ["full_jones"]},
                "slow_timestep_sec": 30,
                "fulljones_timestep_sec": 30,
            },
            4,
            id="full_jones_interval_is_not_multiplied",
        ),
        pytest.param(
            {
                "fast_timestep_sec": 10,
                "medium_timestep_sec": 20,
                "slow_timestep_sec": 30,
                "fulljones_timestep_sec": 90,
            },
            2,
            id="missing_strategy_uses_the_default_solves",
        ),
    ],
)
def test_chunking_uses_the_solves_of_the_strategy(
    field, make_chunking_observation, step, expected_chunk_count
):
    """Chunk by the longest interval solved: 40 s (four samples) for a medium-phase
    solve with a DD interval factor of two, 30 s for full-Jones, 60 s for slow gains."""
    obs, _ = make_chunking_observation(12)
    field.full_observations = [obs]
    field.parset["cluster_specific"]["max_nodes"] = 19
    field.dd_interval_factor = 2
    field.medium_timestep_sec = 20

    chunk_observations(field, [{"do_calibrate": True, **step}], data_fraction=1.0)

    assert len(field.observations) == expected_chunk_count


def test_chunking_ignores_the_interval_factor_for_screens(field, make_chunking_observation):
    """Screens are solved without DD solution intervals, so a 20-second medium-phase
    solve spans two samples rather than four."""
    obs, _ = make_chunking_observation(12)
    field.full_observations = [obs]
    field.parset["cluster_specific"]["max_nodes"] = 19
    field.dd_interval_factor = 2
    field.generate_screens = True
    step = {
        "do_calibrate": True,
        "calibration_strategy": {"dd": ["fast_phase", "medium_phase"]},
        "fast_timestep_sec": 10,
        "medium_timestep_sec": 20,
    }

    chunk_observations(field, [step], data_fraction=1.0)

    assert len(field.observations) == 6


def _full_jones_step(fulljones_timestep_sec, **updates):
    return {
        "do_calibrate": True,
        "calibration_strategy": {"di": ["full_jones"]},
        "fulljones_timestep_sec": fulljones_timestep_sec,
        **updates,
    }


def test_selfcal_cycles_use_the_same_data_in_chunks_of_their_own_solves(
    field, make_chunking_observation
):
    """With 40% of 60 samples, four 6-sample ranges are selected for the longest
    (60-second) solve of any selfcal cycle. A phase-only cycle with 20-second solves
    splits each range into three chunks; a slow-gain cycle keeps whole ranges."""
    obs, sample_times = make_chunking_observation(60)
    field.full_observations = [obs]
    phase_only = {
        "do_calibrate": True,
        "calibration_strategy": {"dd": ["fast_phase", "medium_phase"]},
        "fast_timestep_sec": 10,
        "medium_timestep_sec": 20,
    }
    slow_gains = {
        **phase_only,
        "calibration_strategy": {"dd": ["fast_phase", "medium_phase", "slow_gains"]},
        "slow_timestep_sec": 60,
    }
    steps = [phase_only, slow_gains]
    selected = [time for start in (0, 180, 360, 540) for time in range(start, start + 60, 10)]

    chunk_observations(field, steps[:1], data_fraction=0.4, selection_steps=steps)
    phase_only_chunks = samples_in_chunks(sample_times, field.observations)
    rechunk_observations(field, steps, 1, data_fraction=0.4)
    slow_gain_chunks = samples_in_chunks(sample_times, field.observations)

    assert [len(chunk) for chunk in phase_only_chunks] == [2] * 12
    assert [len(chunk) for chunk in slow_gain_chunks] == [6] * 4
    assert sum(phase_only_chunks, []) == sum(slow_gain_chunks, []) == selected

    # A repeat of the cycle keeps its chunks
    observations = field.observations
    assert rechunk_observations(field, steps, 1, data_fraction=0.4) is steps[1]
    assert field.observations is observations


def test_rechunking_keeps_the_chunks_of_a_cycle_without_calibration(
    field, make_chunking_observation
):
    """An image-only cycle applies the solutions of the cycles before it, so it keeps
    their chunks rather than splitting into the six 2-sample chunks of no solves."""
    obs, _ = make_chunking_observation(12)
    field.full_observations = [obs]
    field.parset["cluster_specific"]["max_nodes"] = 19
    steps = [
        _full_jones_step(60, peel_outliers=True),
        {"do_calibrate": False, "peel_outliers": False},
    ]
    chunk_observations(field, steps[:1], data_fraction=1.0)
    observations = field.observations

    step = rechunk_observations(field, steps, 1, data_fraction=1.0)

    assert step is steps[-1]
    assert field.observations is observations
    assert len(observations) == 2


@pytest.mark.parametrize(
    "fulljones_timestep_sec, max_nodes, expected_chunk_count",
    [
        pytest.param(30, 19, 4, id="shorter_solves_split_into_more_chunks"),
        pytest.param(30, 2, 2, id="node_count_keeps_the_same_chunks"),
    ],
)
def test_rechunking_follows_the_chunks_of_the_cycle(
    field, make_chunking_observation, fulljones_timestep_sec, max_nodes, expected_chunk_count
):
    """A 60-second solve gives two 6-sample chunks; a 30-second solve gives four
    3-sample chunks unless the node count limits them to two again. Calibrating steps
    need not set the peeling flag."""
    obs, _ = make_chunking_observation(12)
    field.full_observations = [obs]
    field.parset["cluster_specific"]["max_nodes"] = max_nodes
    steps = [_full_jones_step(60), _full_jones_step(fulljones_timestep_sec)]
    chunk_observations(field, steps[:1], data_fraction=1.0)
    observations = field.observations

    step = rechunk_observations(field, steps, 1, data_fraction=1.0)

    assert len(field.observations) == expected_chunk_count
    if expected_chunk_count == len(observations):
        assert field.observations is observations
        assert step is steps[-1]
    else:
        assert step == {**steps[-1], "peel_outliers": False}
    assert "peel_outliers" not in steps[-1]


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
    """A 60-second solve spans six samples at 10 s/sample, but only three at 20 s/sample,
    where the two leftover samples join the last chunk."""
    short_interval_obs, short_interval_times = make_chunking_observation(8, 10, "short_interval.ms")
    long_interval_obs, long_interval_times = make_chunking_observation(8, 20, "long_interval.ms")
    field.full_observations = [short_interval_obs, long_interval_obs]
    field.parset["cluster_specific"]["max_nodes"] = 3
    steps = [
        {
            "do_calibrate": True,
            "calibration_strategy": {"di": ["full_jones"]},
            "fulljones_timestep_sec": 60,
        }
    ]

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
        [0, 20, 40],
        [60, 80, 100, 120, 140],
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
    assert samples_in_chunks(long_interval_times, long_interval_chunks) == [[0, 40, 80], [120, 160]]


@pytest.mark.parametrize("max_nodes", [1, 2])
@pytest.mark.parametrize("do_calibrate", [False, True])
def test_chunking_partial_data_is_independent_of_node_count(
    field, make_chunking_observation, max_nodes, do_calibrate
):
    """Keep six of ten samples in three 600-second chunks, even with fewer nodes."""
    obs, sample_times = make_chunking_observation(10, timepersample=300)
    field.full_observations = [obs]
    field.parset["cluster_specific"]["max_nodes"] = max_nodes
    steps = [
        {
            "do_calibrate": do_calibrate,
            "calibration_strategy": {"di": ["full_jones"]},
            "fulljones_timestep_sec": 600,
        }
    ]

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

    chunks = obs.chunk_observation(2)

    assert samples_in_chunks(sample_times, chunks) == [[0, 10], [40, 50], [80, 90]]


def test_chunking_partial_data_splits_each_selected_range(make_chunking_observation):
    """Two 7-sample ranges hold two 3-sample solution intervals each; the leftover
    sample of each range joins its last chunk."""
    obs, sample_times = make_chunking_observation(20)
    obs.data_fraction = 0.7

    chunks = obs.chunk_observation(3, selection_timesteps=7)

    assert samples_in_chunks(sample_times, chunks) == [
        [0, 10, 20],
        [30, 40, 50, 60],
        [130, 140, 150],
        [160, 170, 180, 190],
    ]


@pytest.mark.parametrize("chunk_timesteps", [2, 6])
def test_chunking_high_elevation_ranges_keep_every_sample(
    make_chunking_observation, chunk_timesteps
):
    """Four 6-sample ranges of the high-elevation samples 10 to 49 keep all of their
    samples at the MJD times of the test MS, in chunks of two or six samples."""
    timepersample = 10.0139008
    obs, sample_times = make_chunking_observation(60, timepersample, starttime=4871282393.00695)
    # As scanned: the observation extends a hundredth of a sample beyond its first and
    # last samples, and the high-elevation period is given by the times of samples
    obs.starttime -= timepersample / 100
    obs.endtime += timepersample / 100
    obs.high_el_starttime = sample_times[10]
    obs.high_el_endtime = sample_times[49]
    obs.data_fraction = 0.4

    chunks = obs.chunk_observation(chunk_timesteps, selection_timesteps=6)

    selected = [
        sample_times[index] for first in (10, 21, 32, 43) for index in range(first, first + 6)
    ]
    kept = samples_in_chunks(sample_times, chunks)
    assert [len(chunk) for chunk in kept] == [chunk_timesteps] * (24 // chunk_timesteps)
    assert sum(kept, []) == selected


def test_chunking_partial_data_does_not_create_an_extra_chunk(make_chunking_observation):
    """Keep six of seven samples as three pairs, leaving the last sample unused."""
    obs, sample_times = make_chunking_observation(7)
    obs.starttime = -0.1
    obs.endtime = 60.1
    obs.high_el_starttime = obs.starttime
    obs.high_el_endtime = obs.endtime
    obs.data_fraction = 0.9

    chunks = obs.chunk_observation(2)

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

    chunks = obs.chunk_observation(2, prefer_high_el_periods=prefer_high_el_periods)

    assert samples_in_chunks(sample_times, chunks) == expected_chunks


@pytest.mark.parametrize(
    "data_fraction, expected_chunks, expected_starts, expected_ends",
    [
        pytest.param(1.0, [[0, 10, 20, 30], [40, 50]], [-0.1, 39.9], [30.1, 50.1], id="full_data"),
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

    chunks = obs.chunk_observation(2, max_chunks=2)

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
    steps = [
        {
            "do_calibrate": True,
            "calibration_strategy": {"di": ["full_jones"]},
            "fulljones_timestep_sec": 40,
        }
    ]

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

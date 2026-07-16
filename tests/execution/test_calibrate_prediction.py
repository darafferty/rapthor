import pytest

import rapthor.execution.calibrate.prediction as calibrate_prediction


@pytest.mark.parametrize(
    ("frequencies", "fallback_bandwidth", "max_bandwidth", "expected_chunks"),
    [
        (
            [150_000_000.0],
            1_000_000.0,
            2_000_000.0,
            [
                {
                    "frequency_bandwidth": [150_000_000.0, 1_000_000.0],
                    "channel_range": (0, 1),
                }
            ],
        ),
        (
            [100.0, 110.0, 120.0, 130.0],
            40.0,
            20.0,
            [
                {"frequency_bandwidth": [105.0, 20.0], "channel_range": (0, 2)},
                {"frequency_bandwidth": [125.0, 20.0], "channel_range": (2, 4)},
            ],
        ),
        (
            [100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0],
            80.0,
            30.0,
            [
                {"frequency_bandwidth": [110.0, 30.0], "channel_range": (0, 3)},
                {"frequency_bandwidth": [140.0, 30.0], "channel_range": (3, 6)},
                {"frequency_bandwidth": [165.0, 20.0], "channel_range": (6, 8)},
            ],
        ),
    ],
    ids=["single-channel", "exact-division", "uneven-final-chunk"],
)
def test_wsclean_prediction_frequency_chunks_cover_each_channel_once(
    monkeypatch,
    frequencies,
    fallback_bandwidth,
    max_bandwidth,
    expected_chunks,
):
    """Frequency chunks must be complete, ordered, and non-overlapping."""
    monkeypatch.setattr(
        calibrate_prediction,
        "_measurement_set_channel_frequencies",
        lambda _msin: frequencies,
    )

    chunks = calibrate_prediction._frequency_chunks_for_ms(
        "input.ms",
        [frequencies[0], fallback_bandwidth],
        max_bandwidth_hz=max_bandwidth,
    )

    assert chunks == expected_chunks
    covered_channels = [channel for chunk in chunks for channel in range(*chunk["channel_range"])]
    assert covered_channels == list(range(len(frequencies)))

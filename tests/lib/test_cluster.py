"""
Test cases for the cluster module in `rapthor/lib`.
"""

import numpy
import pytest

from rapthor.lib.cluster import estimate_dp3_peak_memory, get_available_memory


def test_estimate_dp3_peak_memory_accounts_for_all_current_buffers():
    estimate = estimate_dp3_peak_memory(
        baselines=3,
        channels=8,
        solution_interval_seconds=12,
        sampling_interval_seconds=4,
        directions=4,
    )

    samples = 3 * 8 * 3 * (4 + 1)
    assert estimate.time_steps == 3
    assert estimate.visibility_copies_gb == pytest.approx(samples * 4 * 8 / 1e9)
    assert estimate.weights_gb == pytest.approx(samples * 4 * 4 / 1e9)
    assert estimate.weighted_data_gb == pytest.approx(samples * 4 * 8 / 1e9)
    assert estimate.peak_memory_gb == pytest.approx(samples * 80 / 1e9)


@pytest.mark.parametrize(
    "overrides",
    [
        {"baselines": 0},
        {"channels": 0},
        {"solution_interval_seconds": 0},
        {"sampling_interval_seconds": 0},
        {"directions": 0},
    ],
)
def test_estimate_dp3_peak_memory_requires_positive_inputs(overrides):
    inputs = {
        "baselines": 3,
        "channels": 8,
        "solution_interval_seconds": 12,
        "sampling_interval_seconds": 4,
        "directions": 4,
    }
    inputs.update(overrides)

    with pytest.raises(ValueError, match="must be positive"):
        estimate_dp3_peak_memory(**inputs)


def test_get_available_memory(monkeypatch):
    """
    Test the get_available_memory function to ensure it returns the expected value.
    """
    # Mock the subprocess.getoutput to return a fixed string
    monkeypatch.setattr(
        "subprocess.getoutput",
        lambda _: (
            "               total        used        free      shared  buff/cache   available\n"
            "Mem:              15           9           3           2           5           6\n"
            "Swap:             19           3          16\n"
            "Total:            35          12          20\n"
        ),
    )
    available_memory = get_available_memory()
    assert available_memory == 6, f"Expected 6 GB of available memory, got {available_memory} GB"

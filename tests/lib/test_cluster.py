"""
Test cases for the cluster module in `rapthor/lib`.
"""

import numpy
import pytest

from rapthor.lib.cluster import estimate_dp3_peak_memory, get_available_memory, get_chunk_size


def test_estimate_dp3_peak_memory():
    estimate = estimate_dp3_peak_memory(
        baselines=3,
        channels=2,
        solution_interval_seconds=10,
        sampling_interval_seconds=4,
        directions=2,
    )

    assert estimate.time_steps == 3
    assert estimate.visibility_copies_gb == pytest.approx(54 * 4 * 8 / 1e9)
    assert estimate.weights_gb == pytest.approx(54 * 4 * 4 / 1e9)
    assert estimate.weighted_data_gb == pytest.approx(54 * 4 * 8 / 1e9)
    assert estimate.peak_memory_gb == pytest.approx(54 * 80 / 1e9)


@pytest.mark.parametrize(
    "parameter",
    [
        "baselines",
        "channels",
        "solution_interval_seconds",
        "sampling_interval_seconds",
        "directions",
    ],
)
def test_estimate_dp3_peak_memory_requires_positive_inputs(parameter):
    inputs = {
        "baselines": 3,
        "channels": 2,
        "solution_interval_seconds": 10,
        "sampling_interval_seconds": 4,
        "directions": 2,
    }
    inputs[parameter] = 0

    with pytest.raises(ValueError, match=f"{parameter} must be positive"):
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


@pytest.mark.parametrize(
    "max_nodes, numsamples, numobs, solint, expected_chunk_size",
    [
        (4, 1000, 5, 10, 1000),
        (4, 1000, 3, 15, 495),
        (8, 1500, 4, 27, 729),
        (12, 4000, 6, 35, 1995),
        (8, 0, 5, 10, 10),       # Edge case: zero samples (should return solint)
        (8, 1000, 0, 10, None),  # Edge case: zero observations (raises ArithmeticError)
        (4, 1000, 5, 0, None),   # Edge case: zero solution interval (raises ArithmeticError)
        (0, 1000, 5, 10, None),  # Edge case: zero nodes (raises ArithmeticError)
    ], 
)  # fmt: skip
def test_get_chunk_size(max_nodes, numsamples, numobs, solint, expected_chunk_size):
    """
    Test the get_chunk_size function with various parameters to ensure it calculates
    the chunk size correctly.
    """
    cluster_parset = {"max_nodes": max_nodes}
    if numobs == 0 or solint == 0 or max_nodes == 0:
        with pytest.raises(ArithmeticError):
            with numpy.errstate(divide="ignore"):  # Suppress division by zero warnings
                get_chunk_size(cluster_parset, numsamples, numobs, solint)
    else:
        chunk_size = get_chunk_size(cluster_parset, numsamples, numobs, solint)
        assert expected_chunk_size == chunk_size, (
            f"Expected chunk size {expected_chunk_size}, got {chunk_size}"
        )

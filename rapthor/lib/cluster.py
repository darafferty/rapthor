"""Compute-cluster sizing and resource helpers."""

import logging
import math
import subprocess
from dataclasses import dataclass

log = logging.getLogger("rapthor:cluster")

_CORRELATIONS_PER_SAMPLE = 4
_COMPLEX64_BYTES = 8
_FLOAT32_BYTES = 4
_BYTES_PER_DECIMAL_GB = 1e9


@dataclass(frozen=True)
class DP3MemoryEstimate:
    """Peak-memory terms for one DP3 calibration solve."""

    time_steps: int
    visibility_copies_gb: float
    weights_gb: float
    weighted_data_gb: float
    peak_memory_gb: float


def estimate_dp3_peak_memory(
    *,
    baselines: int,
    channels: int,
    solution_interval_seconds: float,
    sampling_interval_seconds: float,
    directions: int,
) -> DP3MemoryEstimate:
    """Estimate current DP3 calibration peak memory in decimal gigabytes."""
    inputs = {
        "baselines": baselines,
        "channels": channels,
        "solution_interval_seconds": solution_interval_seconds,
        "sampling_interval_seconds": sampling_interval_seconds,
        "directions": directions,
    }
    for name, value in inputs.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")

    time_steps = math.ceil(solution_interval_seconds / sampling_interval_seconds)
    direction_buffers = directions + 1
    visibility_samples = baselines * channels * time_steps * direction_buffers
    correlation_values = visibility_samples * _CORRELATIONS_PER_SAMPLE
    visibility_copies_gb = correlation_values * _COMPLEX64_BYTES / _BYTES_PER_DECIMAL_GB
    weights_gb = correlation_values * _FLOAT32_BYTES / _BYTES_PER_DECIMAL_GB
    weighted_data_gb = correlation_values * _COMPLEX64_BYTES / _BYTES_PER_DECIMAL_GB

    return DP3MemoryEstimate(
        time_steps=time_steps,
        visibility_copies_gb=visibility_copies_gb,
        weights_gb=weights_gb,
        weighted_data_gb=weighted_data_gb,
        peak_memory_gb=visibility_copies_gb + weights_gb + weighted_data_gb,
    )


def get_available_memory():
    """
    Returns the available memory in GB

    Note: a call to 'free' is used, which is parsed for the "available" value,
    the last entry on the second line of output.

    Returns
    -------
    available_gb : int
        Available memory in GB
    """
    memstr = subprocess.getoutput("free -t -g").split("\n")[1]  # second line
    available_gb = list(map(int, memstr.split()[1:]))[-1]  # last entry

    return available_gb

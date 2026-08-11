"""
Module that holds all compute-cluster-related functions
"""

import logging
import math
import subprocess
from dataclasses import dataclass

log = logging.getLogger("rapthor:cluster")


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
    samples = baselines * channels * time_steps * (directions + 1)
    visibility_copies_gb = samples * 4 * 8 / 1e9
    weights_gb = samples * 4 * 4 / 1e9
    weighted_data_gb = samples * 4 * 8 / 1e9

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


def get_chunk_size(cluster_parset, numsamples, numobs, solint):
    """
    Returns the optimal chunk size to use during a solve

    Parameters
    ----------
    cluster_parset : dict
        Cluster-specific parset dictionary
    numsamples : int
        Total number of samples in the observation
    numobs : int
        Total number of observations
    solint : int
        Solution interval in number of samples to be used for the solve

    Returns
    -------
    samples_per_chunk : int
        Size of chunk in number of samples
    """
    # Determine the size of chunks to split the calibration into (to allow
    # parallelization over nodes).
    #
    # Try to make at least as many chunks (over all observations) as there are
    # nodes and ensure that the solint is a divisor of samples_per_chunk
    # (otherwise we could get a lot of solutions with less than the target size)
    target_numchunks = math.ceil(cluster_parset["max_nodes"] / numobs)
    samples_per_chunk = math.ceil(numsamples / target_numchunks)
    samples_per_chunk -= samples_per_chunk % solint
    if samples_per_chunk < solint:
        samples_per_chunk = solint

    return samples_per_chunk

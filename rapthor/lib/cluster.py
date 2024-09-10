"""
Module that holds all compute-cluster-related functions
"""
import subprocess
import logging
import numpy as np

log = logging.getLogger('rapthor:cluster')


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
    memstr = subprocess.getoutput('free -t -g').split('\n')[1]  # second line
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
    target_numchunks = np.ceil(cluster_parset['max_nodes'] / numobs)
    samples_per_chunk = int(np.ceil(numsamples / target_numchunks))
    samples_per_chunk -= samples_per_chunk % solint
    if samples_per_chunk < solint:
        samples_per_chunk = solint

    return samples_per_chunk

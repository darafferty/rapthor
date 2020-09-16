"""
Module that holds all compute-cluster-related functions
"""
import os
import logging
import sys
import re
import numpy as np
import shutil

log = logging.getLogger('rapthor:cluster')


def get_total_memory():
    """
    Returns the total memory in GB
    """
    tot_gb, used_gb, free_gb = list(map(int, os.popen('free -t -g').readlines()[-1].split()[1:]))

    return tot_gb


def get_time_chunksize(cluster_parset, timepersample, numsamples, solint_fast_timestep,
                       antenna, ndir):
    """
    Returns the target chunk size in seconds for an observation

    Parameters
    ----------
    cluster_parset : dict
        Cluster-specific parset dictionary
    timepersample : float
        Time in seconds per time sample
    numsamples : int
        Total number of time samples in the observation
    solint_fast_timestep : int
        Number of time samples in fast-phase solve
    antenna : str
        Antenna type: "HBA" or "LBA"
    ndir : int
        Number of directions/patches in the calibration
    """
    # TODO: check memory usage?
    mem_gb = get_total_memory()
    if antenna == 'HBA':
        # Memory usage in GB/timeslot/dir of a typical HBA observation
        mem_usage_gb = 0.2
    elif antenna == 'LBA':
        # Memory usage in GB/timeslot/dir of a typical LBA observation
        mem_usage_gb = 0.05
    gb_per_solint = mem_usage_gb * solint_fast_timestep * ndir

    # Try to make at least as many time chunks as there are nodes, but ensure that
    # solint_fast_timestep a divisor of samplesperchunk (otherwise we could get a lot
    # of solutions with less than the target time)
    n_nodes = cluster_parset['max_nodes']
    samplesperchunk = np.ceil(numsamples / n_nodes)
    if mem_gb / gb_per_solint < 1.0:
        old_solint_fast_timestep = solint_fast_timestep
        solint_fast_timestep *= mem_gb / gb_per_solint
        solint_fast_timestep = max(1, int(round(solint_fast_timestep)))
        log.warn('Not enough memory available for fast-phase solve. Reducing solution '
                 'time interval from {0} to {1}'.format(old_solint_fast_timestep,
                                                        solint_fast_timestep))
    while samplesperchunk % solint_fast_timestep:
        samplesperchunk -= 1
    if samplesperchunk < solint_fast_timestep:
        samplesperchunk = solint_fast_timestep
    target_time_chunksize = timepersample * samplesperchunk

    return target_time_chunksize, solint_fast_timestep


def get_frequency_chunksize(cluster_parset, channelwidth, solint_slow_freqstep,
                            solint_slow_timestep, antenna, ndir):
    """
    Returns the target chunk size in Hz for an observation (the maximum chunk size
    that will fit in memory)

    Parameters
    ----------
    cluster_parset : dict
        Cluster-specific parset dictionary
    channelwidth : float
        Bandwidth in Hz per frequency sample
    solint_slow_freqstep : int
        Number of frequency samples in slow-gain solve
    solint_slow_timestep : int
        Number of time samples in slow-gain solve
    antenna : str
        Antenna type: "HBA" or "LBA"
    ndir : int
        Number of directions/patches in the calibration
    """
    # Try to make at least as many time chunks as there are nodes
    mem_gb = get_total_memory()
    if antenna == 'HBA':
        # Memory usage in GB/chan/timeslot/dir of a typical HBA observation
        mem_usage_gb = 1e-3
    elif antenna == 'LBA':
        # Memory usage in GB/chan/timeslot/dir of a typical LBA observation
        mem_usage_gb = 2.5e-4
    gb_per_solint = mem_usage_gb * solint_slow_freqstep * solint_slow_timestep * ndir

    # Determine if we need to reduce solint_slow_timestep to fit in memory. We adjust
    # the time step rather than the frequency step, as it is less critical and usually
    # has a finer sampling. However, we ensure that the new time step is an even
    # divisor of the original one to avoid problems with irregular steps and IDG
    if mem_gb / gb_per_solint < 1.0:
        old_solint_slow_timestep = solint_slow_timestep
        solint_slow_timestep *= mem_gb / gb_per_solint
        solint_slow_timestep = max(1, int(round(solint_slow_timestep)))
        while old_solint_slow_timestep % solint_slow_timestep:
            solint_slow_timestep -= 1
        log.warn('Not enough memory available for slow-gain solve. Reducing solution '
                 'time interval from {0} to {1}'.format(old_solint_slow_timestep,
                                                        solint_slow_timestep))
    nsolints = max(1, int(round(mem_gb / gb_per_solint)))
    channelsperchunk = np.ceil(solint_slow_freqstep * nsolints)
    target_freq_chunksize = channelwidth * channelsperchunk

    return target_freq_chunksize, solint_slow_timestep

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


def get_fast_solve_intervals(cluster_parset, numsamples, numobs, target_timestep,
                             antenna, ndir):
    """
    Returns the optimal solution interval and chunk size (both in number of time
    slots) for an observation so that the fast-phase solves in DPPP will fit in
    memory

    Parameters
    ----------
    cluster_parset : dict
        Cluster-specific parset dictionary
    numsamples : int
        Total number of time samples in the observation
    numobs : int
        Total number of observations
    target_timestep : int
        Target number of time samples for the fast-phase solve
    antenna : str
        Antenna type: "HBA" or "LBA"
    ndir : int
        Number of directions/patches in the calibration

    Returns
    -------
    samples_per_chunk : int
        Size of chunk in number time slots
    solint : int
        Solution interval in number of time slots that will ensure the solve
        fits in the available memory
    """
    def get_mem_per_solution(target_timestep, ndir):
        # Estimate the memory usage per solution
        #
        # Note: the numbers below were determined empirically from typical (Dutch-only)
        # datasets
        if antenna == 'HBA':
            # Memory usage in GB/timeslot/dir of a typical HBA observation
            mem_usage_gb = 0.2
        elif antenna == 'LBA':
            # Memory usage in GB/timeslot/dir of a typical LBA observation
            mem_usage_gb = 0.05

        # Return total usage plus a 20% safety buffer
        return mem_usage_gb * target_timestep * ndir * 1.2

    # Determine whether we need to adjust the solution interval to fit the solve in
    # memory
    if 'mem_per_node_gb' in cluster_parset and cluster_parset['mem_per_node_gb'] != 0:
        # If the user has specified the memory to request, use that value
        mem_gb = cluster_parset['mem_per_node_gb']
    else:
        # Otherwise, get it from the machine Rapthor is running on
        mem_gb = get_available_memory()
    gb_per_sol = get_mem_per_solution(target_timestep, ndir)
    if mem_gb / gb_per_sol < 1.0:
        solint = target_timestep * mem_gb / gb_per_sol
        solint = max(1, int(round(solint)))
        if solint < target_timestep:
            log.warn('Not enough memory available for fast-phase solve. Reducing solution '
                     'time interval from {0} to {1} time slots'.format(target_timestep, solint))
    else:
        solint = target_timestep

    # Determine the size of chunks to split the calibration into (to allow
    # parallelization over nodes).
    #
    # Try to make at least as many chunks (over all observations) as there are
    # nodes and ensure that the solint is a divisor of samples_per_chunk
    # (otherwise we could get a lot of solutions with less than the target time)
    target_numchunks = np.ceil(cluster_parset['max_nodes'] / numobs)
    samples_per_chunk = int(np.ceil(numsamples / target_numchunks))
    samples_per_chunk -= samples_per_chunk % solint
    if samples_per_chunk < solint:
        samples_per_chunk = solint

    return samples_per_chunk, solint


def get_slow_solve_intervals(cluster_parset, numsamples, numobs, target_freqstep,
                             target_timestep, antenna, ndir):
    """
    Returns the optimal solution interval (in number of time slots) and chunk
    size (in number of frequency channels) for an observation so that the
    slow-gain solves in DPPP will fit in memory

    Parameters
    ----------
    cluster_parset : dict
        Cluster-specific parset dictionary
    numsamples : int
        Total number of frequency samples in the observation
    numobs : int
        Total number of observations
    target_freqstep : int
        Target number of frequency samples for the slow-gain solve
    target_timestep : int
        Target number of time samples for the slow-gain solve
    antenna : str
        Antenna type: "HBA" or "LBA"
    ndir : int
        Number of directions/patches in the calibration

    Returns
    -------
    samples_per_chunk : int
        Size of chunk in number of frequency channels
    solint : int
        Solution interval in number of frequency channels that will ensure the solve
        fits in the available memory
    """
    def get_gb_per_solution(target_freqstep, target_timestep, ndir):
        # Estimate the memory usage in GB per solution
        #
        # Note: the numbers below were determined empirically from typical (Dutch-only)
        # datasets by fitting a 2-D quadratic curve of the form:
        #    Z = C[4]*X**2. + C[5]*Y**2. + C[3]*X*Y + C[1]*X + C[2]*Y + C[0]
        # where X = ndir, Y = target_freqstep, and Z is the memory usage for 25
        # time slots (the usage as a function of number of time slots was observed
        # to be a simple linear scaling so was not included for simplicity)
        if antenna == 'HBA':
            # Best-fit coefficients for 25 time slots
            coef = [7.33333333e-01, 8.33333333e-02, 7.66666667e-03, 5.50000000e-03,
                    -3.20833714e-17, -4.39443416e-18]
        elif antenna == 'LBA':
            # TODO: determine best-fit coefficients
            # (for now set to HBA values)
            coef = [7.33333333e-01, 8.33333333e-02, 7.66666667e-03, 5.50000000e-03,
                    -3.20833714e-17, -4.39443416e-18]
        gb_per_25_timeslots = (coef[4]*ndir**2 + coef[5]*target_freqstep**2 +
                               coef[3]*ndir*target_freqstep + coef[1]*ndir +
                               coef[2]*target_freqstep + coef[0])

        # Return total usage, scaled to the requested time interval, plus a 20% safety
        # buffer
        return gb_per_25_timeslots * target_timestep / 25 * 1.2

    # Determine whether we need to adjust the solution interval in time to fit
    # the solve in memory. We adjust the time interval rather than the frequency
    # one, as it is less critical and usually has a finer sampling
    if 'mem_per_node_gb' in cluster_parset and cluster_parset['mem_per_node_gb'] != 0:
        # If the user has specified the memory to request, use that value
        mem_gb = cluster_parset['mem_per_node_gb']
    else:
        # Otherwise, get it from the machine Rapthor is running on
        mem_gb = get_available_memory()
    gb_per_sol = get_gb_per_solution(target_freqstep, target_timestep, ndir)
    if mem_gb / gb_per_sol < 1.0:
        solint = target_timestep * mem_gb / gb_per_sol
        solint = max(1, int(round(solint)))
        if solint < target_timestep:
            log.warn('Not enough memory available for slow-gain solve. Reducing solution '
                     'time interval from {0} to {1} time slots'.format(target_timestep, solint))
    else:
        solint = target_timestep

    # Determine the size of the frequency chunks into which to split the calibration.
    #
    # Try to make at least as many total chunks (over all observations) as there
    # are nodes to maximize the parallelization.
    #
    # Also, unlike for the fast-phase solves, the memory usage scales with the
    # chunk size (since the solve is parallelized over the channels of a chunk, so
    # larger chunks require more memory). Therefore, we also need to ensure that
    # the chunk size works with the available memory.
    target_numchunks = np.ceil(cluster_parset['max_nodes'] / numobs)
    samples_per_chunk = int(np.ceil(numsamples / target_numchunks))
    gb_per_sol = get_gb_per_solution(samples_per_chunk, target_timestep, ndir)
    if mem_gb / gb_per_sol < 1.0:
        samples_per_chunk = samples_per_chunk * mem_gb / gb_per_sol
        samples_per_chunk = max(1, int(round(samples_per_chunk)))
    if samples_per_chunk < target_freqstep:
        samples_per_chunk = target_freqstep

    return samples_per_chunk, solint

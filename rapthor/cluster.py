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


def get_pbs_nodes():
    """
    Get nodes from the PBS_NODEFILE

    Returns
    -------
    node_list
        List of nodes
    """
    nodes = []
    try:
        filename = os.environ['PBS_NODEFILE']
    except KeyError:
        log.error('PBS_NODEFILE not found. You must have a reservation to '
                  'use clusterdesc = PBS.')
        sys.exit(1)

    with open(filename, 'r') as file:
        for line in file:
            node_name = line.split()[0]
            if node_name not in nodes:
                nodes.append(node_name)
    log.info('Using {0} node(s)'.format(len(nodes)))

    return nodes


def expand_part(s):
    """Expand a part (e.g. "x[1-2]y[1-3][1-3]") (no outer level commas).

    Note: Adapted from git://www.nsc.liu.se/~kent/python-hostlist.git
    """
    # Base case: the empty part expand to the singleton list of ""
    if s == "":
        return [""]

    # Split into:
    # 1) prefix string (may be empty)
    # 2) rangelist in brackets (may be missing)
    # 3) the rest

    m = re.match(r'([^,\[]*)(\[[^\]]*\])?(.*)', s)
    (prefix, rangelist, rest) = m.group(1, 2, 3)

    # Expand the rest first (here is where we recurse!)
    rest_expanded = expand_part(rest)

    # Expand our own part
    if not rangelist:
        # If there is no rangelist, our own contribution is the prefix only
        us_expanded = [prefix]
    else:
        # Otherwise expand the rangelist (adding the prefix before)
        us_expanded = expand_rangelist(prefix, rangelist[1:-1])

    return [us_part + rest_part
            for us_part in us_expanded
            for rest_part in rest_expanded]


def expand_range(prefix, range_):
    """ Expand a range (e.g. 1-10 or 14), putting a prefix before.

    Note: Adapted from git://www.nsc.liu.se/~kent/python-hostlist.git
    """
    # Check for a single number first
    m = re.match(r'^[0-9]+$', range_)
    if m:
        return ["%s%s" % (prefix, range_)]

    # Otherwise split low-high
    m = re.match(r'^([0-9]+)-([0-9]+)$', range_)

    (s_low, s_high) = m.group(1, 2)
    low = int(s_low)
    high = int(s_high)
    width = len(s_low)

    results = []
    for i in range(low, high+1):
        results.append("%s%0*d" % (prefix, width, i))
    return results


def expand_rangelist(prefix, rangelist):
    """ Expand a rangelist (e.g. "1-10,14"), putting a prefix before.

    Note: Adapted from git://www.nsc.liu.se/~kent/python-hostlist.git
    """
    # Split at commas and expand each range separately
    results = []
    for range_ in rangelist.split(","):
        results.extend(expand_range(prefix, range_))
    return results


def expand_hostlist(hostlist, allow_duplicates=False, sort=False):
    """Expand a hostlist expression string to a Python list.

    Example: expand_hostlist("n[9-11],d[01-02]") ==>
             ['n9', 'n10', 'n11', 'd01', 'd02']

    Unless allow_duplicates is true, duplicates will be purged
    from the results. If sort is true, the output will be sorted.

    Note: Adapted from git://www.nsc.liu.se/~kent/python-hostlist.git
    """
    results = []
    bracket_level = 0
    part = ""

    for c in hostlist + ",":
        if c == "," and bracket_level == 0:
            # Comma at top level, split!
            if part:
                results.extend(expand_part(part))
            part = ""
        else:
            part += c

        if c == "[":
            bracket_level += 1
        elif c == "]":
            bracket_level -= 1

    seen = set()
    results_nodup = []
    for e in results:
        if e not in seen:
            results_nodup.append(e)
            seen.add(e)
    return results_nodup


def get_slurm_nodes():
    """
    Get nodes from the SLURM_JOB_NODELIST

    Returns
    -------
    node_list
        List of nodes
    """
    nodes = []
    try:
        hostlist = os.environ['SLURM_JOB_NODELIST']
    except KeyError:
        log.error('SLURM_JOB_NODELIST not found. You must have a reservation to '
                  'use clusterdesc = SLURM.')
        sys.exit(1)
    nodes = expand_hostlist(hostlist)
    log.info('Using {0} node(s)'.format(len(nodes)))

    return nodes


def get_compute_nodes(cluster_type):
    """
    Returns list of nodes

    Parameters
    ----------
    cluster_type : str
        One of 'pbs' or 'slurm'; other values are treated as 'localhost'

    Returns
    -------
    result : list
        Sorted list of node names
    """
    if cluster_type == 'pbs':
        nodes = get_pbs_nodes()
    elif cluster_type == 'slurm' or cluster_type == 'juropa_slurm':
        nodes = get_slurm_nodes()
    else:
        nodes = ['localhost']

    return sorted(nodes)


def find_executables(cluster_parset):
    """
    Adds the paths to required executables to parset dict

    Parameters
    ----------
    cluster_parset : dict
        Cluster-specific parset dictionary
    """
    executables = {'genericpipeline_executable': ['genericpipeline.py'],
                   'wsclean_executable': ['wsclean'],
                   'h5collector_executable': ['H5parm_collector.py']}
    for key, names in executables.items():
        for name in names:
            try:
                path = shutil.which(name)
            except AttributeError:
                from distutils import spawn
                path = spawn.find_executable(name)
            if path is not None:
                cluster_parset[key] = path
                break
        if path is None:
            log.error('The path to the {0} executable could not be determined. '
                      'Please make sure it is in your PATH.'.format(name))
            sys.exit(1)

    return cluster_parset


def check_ulimit(cluster_parset):
    """
    Checks the limit on number of open files

    Parameters
    ----------
    cluster_parset : dict
        Cluster-specific parset dictionary
    """
    try:
        import resource
        nof_files_limits = resource.getrlimit(resource.RLIMIT_NOFILE)
        if cluster_parset['batch_system'] == 'singleMachine' and nof_files_limits[0] < nof_files_limits[1]:
            log.debug('Setting limit for number of open files to: {}.'.format(nof_files_limits[1]))
            resource.setrlimit(resource.RLIMIT_NOFILE, (nof_files_limits[1], nof_files_limits[1]))
            nof_files_limits = resource.getrlimit(resource.RLIMIT_NOFILE)
        log.debug('Active limit for number of open files is {0}, maximum limit '
                  'is {1}.'.format(nof_files_limits[0], nof_files_limits[1]))
        if nof_files_limits[0] < 2048:
            log.warn('The limit for number of open files is small, this could '
                     'result in a "Too many open files" problem when running rapthor.')
            log.warn('The active limit can be increased to the maximum for the '
                     'user with: "ulimit -Sn <number>" (bash) or "limit descriptors 1024" (csh).')
    except resource.error:
        log.warn('Cannot check limits for number of open files, what kind of system is this?')


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
    Returns the target chunk size in seconds for an observation

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

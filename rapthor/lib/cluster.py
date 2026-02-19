"""
Module that holds all compute-cluster-related functions
"""
import subprocess
import logging

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

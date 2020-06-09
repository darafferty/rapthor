"""
Definition of context manager classes
"""
from timeit import default_timer as timer
import datetime
import logging
import sys


class Timer(object):
    """
    Context manager used to time operations

    Parameters
    ----------
    log : logging instance
        The logging instance to use. If None, root is used
    type : str, optional
        Type of operation
    """
    def __init__(self, log=None, type='operation'):
        if log is None:
            self.log = logging
        else:
            self.log = log
        self.type = type

    def __enter__(self):
        self.start = timer()

    def __exit__(self, *args):
        elapsed = timer() - self.start
        self.log.debug('Time for {0}: {1}'.format(self.type,
                       datetime.timedelta(seconds=elapsed)))


class RedirectStdStreams(object):
    """
    Context manager used to redirect streams

    Parameters
    ----------
    stdout : file or stream object
        stdout stream
    stderr : file or stream object
        stderr stream
    """
    def __init__(self, stdout=None, stderr=None):
        self._stdout = stdout or sys.stdout
        self._stderr = stderr or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush()
        self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush()
        self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

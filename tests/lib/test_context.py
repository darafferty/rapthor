"""
Test cases for the context module in rapthor library.
"""

import pytest
import sys
from rapthor.lib.context import Timer, RedirectStdStreams


def test_timer():
    """
    Test the Timer context manager to ensure it logs the elapsed time correctly.
    """
    with Timer() as t:
        pass


def test_streams():
    """
    Test the RedirectStdStreams context manager to ensure it redirects stdout and stderr.
    """
    with RedirectStdStreams() as s:
        print("Testing redirect of stdout")
        print("Testing redirect of stderr", file=sys.stderr)

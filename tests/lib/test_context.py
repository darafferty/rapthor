"""
Test cases for the context module in `rapthor/lib`.
"""

import io
import logging
import sys

from rapthor.lib.context import RedirectStdStreams, Timer


def test_timer_logs_elapsed_time(caplog):
    logger = logging.getLogger("rapthor:test-context")

    with caplog.at_level(logging.DEBUG, logger=logger.name):
        with Timer(log=logger, type="calibration"):
            sum(range(3))

    assert "Time for calibration:" in caplog.text


def test_redirect_std_streams_captures_and_restores_stdout_and_stderr():
    stdout = io.StringIO()
    stderr = io.StringIO()
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    with RedirectStdStreams(stdout=stdout, stderr=stderr):
        print("Testing redirect of stdout")
        print("Testing redirect of stderr", file=sys.stderr)

    assert stdout.getvalue() == "Testing redirect of stdout\n"
    assert stderr.getvalue() == "Testing redirect of stderr\n"
    assert sys.stdout is original_stdout
    assert sys.stderr is original_stderr

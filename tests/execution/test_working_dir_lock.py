"""Exclusive use of a working directory by one Rapthor process."""

import errno
import logging
import subprocess
import sys

import pytest

from rapthor.execution import working_dir_lock
from rapthor.execution.working_dir_lock import (
    WorkingDirectoryInUse,
    WorkingDirectoryLockError,
    working_directory_lock,
)

# Holds the lock until it is killed, reporting once it is held
_HOLD_LOCK = """
import sys, time
from rapthor.execution.working_dir_lock import working_directory_lock
with working_directory_lock(sys.argv[1], "a Rapthor run"):
    print("locked", flush=True)
    time.sleep(120)
"""


def test_second_lock_is_refused_even_in_the_same_process(tmp_path):
    with working_directory_lock(str(tmp_path), "a Rapthor run"):
        with pytest.raises(
            WorkingDirectoryInUse, match=r"in use by a Rapthor run \(process \d+ on "
        ):
            with working_directory_lock(str(tmp_path), "a Rapthor reset", required=True):
                pytest.fail("a second holder must be refused")

    with working_directory_lock(str(tmp_path), "a Rapthor reset", required=True):
        pass


def test_lock_of_a_killed_process_does_not_block_a_later_run(tmp_path):
    holder = subprocess.Popen(
        [sys.executable, "-c", _HOLD_LOCK, str(tmp_path)], stdout=subprocess.PIPE, text=True
    )
    try:
        assert holder.stdout.readline().strip() == "locked"
        with pytest.raises(WorkingDirectoryInUse):
            with working_directory_lock(str(tmp_path), "a Rapthor reset", required=True):
                pytest.fail("the lock of a running process must be honoured")
    finally:
        holder.kill()
        holder.wait(timeout=30)
        holder.stdout.close()

    with working_directory_lock(str(tmp_path), "a Rapthor reset", required=True):
        pass


@pytest.mark.parametrize("cause", ["flock-unsupported", "lock-file-is-a-directory"])
def test_unavailable_lock_stops_a_reset_but_not_a_run(tmp_path, monkeypatch, caplog, cause):
    if cause == "flock-unsupported":

        def flock(file, operation):
            raise OSError(errno.ENOLCK, "No locks available")

        monkeypatch.setattr(working_dir_lock.fcntl, "flock", flock)
    else:
        (tmp_path / working_dir_lock.LOCK_FILENAME).mkdir()

    # A run continues without the lock, but a reset refuses to go ahead
    with caplog.at_level(logging.WARNING, logger="rapthor:lock"):
        with working_directory_lock(str(tmp_path), "a Rapthor run"):
            with pytest.raises(WorkingDirectoryLockError, match="Could not lock the working"):
                with working_directory_lock(str(tmp_path), "a Rapthor reset", required=True):
                    pytest.fail("a reset must not go ahead without the lock")
    assert "reset protection is off" in caplog.text

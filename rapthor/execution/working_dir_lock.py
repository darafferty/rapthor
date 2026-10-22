"""Exclusive use of a working directory by one Rapthor process."""

import fcntl
import logging
import os
import socket
from collections.abc import Generator
from contextlib import contextmanager, suppress

log = logging.getLogger("rapthor:lock")
LOCK_FILENAME = ".rapthor.lock"


class WorkingDirectoryLockError(RuntimeError):
    """Raised when the working directory cannot be locked."""


class WorkingDirectoryInUse(WorkingDirectoryLockError):
    """Raised when another Rapthor process is using the working directory."""


@contextmanager
def working_directory_lock(
    dir_working: str, purpose: str, *, required: bool = False
) -> Generator[None, None, None]:
    """Hold the working directory for this process, or raise WorkingDirectoryInUse.

    Runs and resets take this advisory ``flock``, so that a reset cannot change
    the files of a running pipeline and two runs cannot share a working
    directory. The operating system releases it when the process ends, even if
    it is killed. The lock file names the holder for the error message. A second
    call in the same process is refused too, except where ``flock`` is emulated
    with POSIX locks (NFS on Linux).

    If the lock cannot be taken for another reason, for example because the file
    system does not support it, a required lock (for a reset) raises
    WorkingDirectoryLockError, while a run logs a warning and continues without it.
    """
    lock_file = None
    try:
        os.makedirs(dir_working, exist_ok=True)
        lock_file = open(os.path.join(dir_working, LOCK_FILENAME), "a+", encoding="utf-8")
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        # Only flock can block: another process holds the lock
        holder = ""
        with lock_file, suppress(OSError):
            lock_file.seek(0)
            holder = lock_file.read().strip()
        raise WorkingDirectoryInUse(
            f"The working directory {dir_working} is in use by "
            f"{holder or 'another Rapthor process'}. Wait for it to finish, or stop it, "
            "before trying again."
        ) from None
    except OSError as error:
        if lock_file is not None:
            lock_file.close()
        if required:
            raise WorkingDirectoryLockError(
                f"Could not lock the working directory {dir_working} ({error}). The lock "
                "is needed to make sure that no Rapthor run is using it."
            ) from error
        log.warning(
            "Could not lock the working directory %s (%s). The run continues without "
            "the lock, so reset protection is off: nothing stops a reset of this working "
            "directory while the run is active. Do not reset until the run has stopped.",
            dir_working,
            error,
        )
        lock_file = None
    else:
        try:
            lock_file.seek(0)
            lock_file.truncate()
            lock_file.write(f"{purpose} (process {os.getpid()} on {socket.gethostname()})\n")
            lock_file.flush()
        except OSError as error:
            # The lock is held; only the description for other processes is missing
            log.warning("Could not describe the holder of the lock in %s: %s", dir_working, error)
    try:
        yield
    finally:
        if lock_file is not None:
            lock_file.close()  # releases the lock

"""Worker-owned temporary directories for external commands.

Temporary files used within one task may be node-local. MPI tasks require
shared storage so every rank can access the same temporary files. Products
passed between tasks are handled separately by the operation workspace.
"""

import logging
import os
import re
import shutil
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from typing import Optional

from rapthor.execution.config import ExecutionConfig

log = logging.getLogger("rapthor:scratch")


@contextmanager
def task_temporary_directory(
    execution_config: ExecutionConfig,
    *,
    name: str = "command",
    use_mpi: bool = False,
    fallback_path: Optional[str] = None,
) -> Generator[Optional[str], None, None]:
    """Create an isolated directory on the worker and clean only that directory.

    A configured scratch root receives a unique child per invocation. MPI uses
    global scratch, falling back to the shared operation directory. With no
    local scratch configured, ordinary commands retain their inherited temp
    environment; callers such as WSClean can supply their existing fallback.
    """
    root = (
        execution_config.global_scratch_dir
        if use_mpi
        else execution_config.resolved_local_scratch_dir()
    )
    if root is not None:
        os.makedirs(root, exist_ok=True)
        prefix = re.sub(r"[^A-Za-z0-9_.-]", "_", name)[:64]
        temporary_directory = tempfile.mkdtemp(prefix=f"rapthor-{prefix}-", dir=root)
    elif fallback_path is not None:
        temporary_directory = os.path.abspath(fallback_path)
        os.makedirs(temporary_directory, exist_ok=True)
    elif use_mpi:
        raise ValueError("MPI temporary files require a shared fallback path")
    else:
        yield None
        return

    try:
        yield temporary_directory
    finally:
        if not execution_config.keep_temporary_files:
            try:
                shutil.rmtree(temporary_directory)
            except OSError as error:
                if os.path.lexists(temporary_directory):
                    log.warning(
                        "Could not remove command scratch %s: %s", temporary_directory, error
                    )


def temporary_environment(temporary_directory: str) -> dict[str, str]:
    """Return temp overrides for the child process without changing worker state."""
    return dict.fromkeys(("TMPDIR", "TMP", "TEMP"), temporary_directory)

"""Leave an operation on shared scratch as a run that stops abruptly does."""

import subprocess
import sys

# Runs a flow on shared scratch in a child process that exits abruptly at one of
# the directory transitions, so that no exception or finally handler can run.
_CHILD = r"""
import errno
import os
import sys
from pathlib import Path
from rapthor.execution import workspace as module

workdir, scratch = map(Path, sys.argv[1:3])
stage = sys.argv[3]
rename, unlink, symlink_to = Path.rename, Path.unlink, Path.symlink_to
copy_workspace = module._copy_workspace

def copy(source, destination):
    copy_workspace(source, destination)
    if (stage == "copy" and source == workdir) or (
        stage == "cross-copy" and source.parent.parent == scratch
    ):
        os._exit(73)

def move(source, destination):
    if source.parent.parent == scratch and stage.startswith("cross-"):
        raise OSError(errno.EXDEV, "different filesystems")
    result = rename(source, destination)
    if (stage == "backup" and source == workdir) or (
        stage == "rename" and source.parent.parent == scratch
    ) or (stage == "cross-rename" and source.name == "promoted"):
        os._exit(73)
    return result

def remove(path, *args, **kwargs):
    result = unlink(path, *args, **kwargs)
    if stage == "unlink" and path == workdir:
        os._exit(73)
    return result

def link(path, *args, **kwargs):
    result = symlink_to(path, *args, **kwargs)
    if stage == "link" and path == workdir:
        os._exit(73)
    return result

module._copy_workspace = copy
Path.rename, Path.unlink, Path.symlink_to = move, remove, link
with module.shared_scratch_workspace(str(workdir), str(scratch)):
    (workdir / "result.fits").write_text("result")
    (workdir / "result-link.fits").symlink_to((workdir / "result.fits").resolve())
    if stage == "flow":
        os._exit(73)
"""


def exit_at_scratch_transition(workdir, scratch, stage):
    result = subprocess.run(
        [sys.executable, "-c", _CHILD, str(workdir), str(scratch), stage],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 73, result.stderr

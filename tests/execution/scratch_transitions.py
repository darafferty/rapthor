"""
Stop a run part-way through staging an operation on ``global_scratch_dir``.

When ``global_scratch_dir`` is set, an operation works on the scratch disk: its
directory in ``dir_working/pipelines`` is copied to a workspace there, moved
aside as a backup and replaced by a link to the workspace. When the operation
ends, the workspace is moved back into place; if the scratch disk is another
filesystem, it is first copied to a "promoted" directory next to the operation
directory. See :func:`rapthor.execution.workspace.shared_scratch_workspace`.

A run that is killed between two of these steps leaves some mix of directory,
link, backup and workspace behind, which a restart or a reset has to cope with.
:func:`exit_at_scratch_transition` reproduces this. It runs a small flow in a
child process (this module as a script) that exits with ``os._exit`` right after
the chosen step, so that no exception handler or ``finally`` clause runs, just as
when a process is killed.
"""

import errno
import os
import subprocess
import sys
from pathlib import Path

import pytest

from rapthor.execution import workspace

# Exit status of the child, so that a planned exit is told apart from a crash
EXIT_CODE = 73

# The step after which the child exits, by the name of the stage
STAGES = {
    "copy": "the operation directory has been copied to the workspace",
    "backup": "the operation directory has been renamed to the backup",
    "link": "the link to the workspace has been created",
    "flow": "the flow has written its products in the workspace",
    "unlink": "the link has been removed, before the workspace is moved back",
    "rename": "the workspace has been moved back to the operation directory",
    "cross-copy": "the workspace has been copied to the promoted directory (other filesystem)",
    "cross-rename": "the promoted directory has been renamed to the operation directory",
}


def exit_at_scratch_transition(workdir, scratch, stage):
    """Run a flow on ``workdir``, staged on ``scratch``, that is killed at ``stage``."""
    if stage not in STAGES:
        raise ValueError(f"Unknown stage {stage!r}; choose from {sorted(STAGES)}")
    result = subprocess.run(
        [sys.executable, __file__, str(workdir), str(scratch), stage],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == EXIT_CODE, result.stderr


def _run_flow_and_exit(workdir, scratch, stage):
    """Run the flow in this process and exit after the step named by ``stage``."""
    cross_device = stage.startswith("cross-")

    # Copies of the functions that are patched below, which the wrappers call
    original_copy_workspace = workspace._copy_workspace
    original_rename = Path.rename
    original_unlink = Path.unlink
    original_symlink_to = Path.symlink_to

    def on_scratch(path):
        # The workspace is scratch/<temporary directory>/workspace
        return path.parent.parent == scratch

    def exit_if(step_reached):
        if step_reached:
            os._exit(EXIT_CODE)

    def copy_workspace(source, destination):
        original_copy_workspace(source, destination)
        exit_if(stage == "copy" and source == workdir)
        exit_if(stage == "cross-copy" and on_scratch(source))

    def rename(source, destination):
        if cross_device and on_scratch(source):
            raise OSError(errno.EXDEV, "different filesystems")
        result = original_rename(source, destination)
        exit_if(stage == "backup" and source == workdir)
        exit_if(stage == "rename" and on_scratch(source))
        exit_if(stage == "cross-rename" and source.name == "promoted")
        return result

    def unlink(path, *args, **kwargs):
        result = original_unlink(path, *args, **kwargs)
        exit_if(stage == "unlink" and path == workdir)
        return result

    def symlink_to(path, *args, **kwargs):
        result = original_symlink_to(path, *args, **kwargs)
        exit_if(stage == "link" and path == workdir)
        return result

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(workspace, "_copy_workspace", copy_workspace)
        monkeypatch.setattr(Path, "rename", rename)
        monkeypatch.setattr(Path, "unlink", unlink)
        monkeypatch.setattr(Path, "symlink_to", symlink_to)
        with workspace.shared_scratch_workspace(str(workdir), str(scratch)):
            (workdir / "result.fits").write_text("result")
            (workdir / "result-link.fits").symlink_to((workdir / "result.fits").resolve())
            exit_if(stage == "flow")


if __name__ == "__main__":
    workdir = Path(sys.argv[1])
    scratch = Path(sys.argv[2])
    stage = sys.argv[3]
    _run_flow_and_exit(workdir, scratch, stage)

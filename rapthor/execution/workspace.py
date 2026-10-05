"""Shared scratch workspaces with durable operation paths."""

import errno
import json
import logging
import os
import shutil
import stat
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

log = logging.getLogger("rapthor:workspace")
WORKSPACE_MARKER = ".rapthor-workspace.json"


@contextmanager
def shared_scratch_workspace(
    pipeline_working_dir: Optional[str], global_scratch_dir: Optional[str]
) -> Generator[None, None, None]:
    """Stage a flow on shared scratch while keeping every output path stable.

    Workers access the ordinary operation path through a shared symlink. On
    success or failure, its contents return to that path before finalization.
    This also preserves partial products for the flow's existing restart checks.
    A sibling recovery record also covers interrupted directory moves when
    the operation link is missing.
    """
    if not pipeline_working_dir:
        yield
        return

    workdir = Path(os.path.abspath(pipeline_working_dir))
    recover_scratch_workspace(str(workdir))
    if workdir.is_symlink():
        workspace = workdir.resolve()
        owner = _workspace_owner(workspace)
        if owner != str(workdir):
            if not global_scratch_dir:
                yield
                return
            raise ValueError(f"Cannot stage an unowned operation symlink on scratch: {workdir}")
        log.info("Recovering shared scratch workspace %s", workspace)
        _record_workspace(workdir, workspace)
    else:
        if not global_scratch_dir:
            yield
            return
        scratch_root = Path(global_scratch_dir).expanduser().resolve()
        if scratch_root.is_relative_to(workdir.resolve()):
            raise ValueError("global_scratch_dir must be outside the operation working directory")
        scratch_root.mkdir(parents=True, exist_ok=True)
        workdir.mkdir(parents=True, exist_ok=True)
        scratch_dir = Path(tempfile.mkdtemp(prefix=f"rapthor-{workdir.name}-", dir=scratch_root))
        workspace = scratch_dir / "workspace"
        (scratch_dir / WORKSPACE_MARKER).write_text(json.dumps({"working_directory": str(workdir)}))
        recovery_dir = _record_workspace(workdir, workspace)
        backup = recovery_dir / "backup"
        try:
            _copy_workspace(workdir, workspace)
            workdir.rename(backup)
            workdir.symlink_to(workspace, target_is_directory=True)
        except BaseException:
            recover_scratch_workspace(str(workdir))
            raise
        else:
            _remove_workspace(backup)
        log.info("Shared scratch workspace for %s is %s", workdir.name, workspace)

    try:
        yield
    finally:
        _restore_workspace(workdir, workspace)


def _recovery_directory(workdir: Path) -> Path:
    return workdir.with_name(f".{workdir.name}.scratch")


def _record_workspace(workdir: Path, workspace: Path) -> Path:
    """Record recovery paths before moving the operation directory."""
    recovery_dir = _recovery_directory(workdir)
    if not recovery_dir.exists():
        recovery_dir.mkdir()
        (recovery_dir / WORKSPACE_MARKER).write_text(
            json.dumps({"working_directory": str(workdir), "workspace": str(workspace)})
        )
    return recovery_dir


def recover_scratch_workspace(pipeline_working_dir: str) -> None:
    """Repair an interrupted directory move before setup can recreate the path.

    The sibling recovery record survives both staging and promotion. An intact
    real operation directory is authoritative; otherwise the original backup
    or the scratch workspace still owns the data.
    """
    workdir = Path(os.path.abspath(pipeline_working_dir))
    recovery_dir = _recovery_directory(workdir)
    if not recovery_dir.exists():
        return
    record = json.loads((recovery_dir / WORKSPACE_MARKER).read_text())
    workspace = Path(record["workspace"])
    if record["working_directory"] != str(workdir) or (
        workspace.parent.exists() and _workspace_owner(workspace) != str(workdir)
    ):
        raise RuntimeError(f"Unowned scratch recovery record: {recovery_dir}")
    backup = recovery_dir / "backup"
    if workdir.is_symlink():
        if workdir.resolve() != workspace.resolve():
            raise RuntimeError(f"Operation scratch link changed: {workdir}")
    elif backup.exists():
        if workdir.exists():
            raise RuntimeError(f"Cannot restore {backup}; operation directory already exists")
        backup.rename(workdir)
    elif not workdir.exists():
        if not workspace.is_dir():
            raise FileNotFoundError(f"Scratch workspace is unavailable: {workspace}")
        workdir.symlink_to(workspace, target_is_directory=True)

    if workdir.is_symlink():
        if not workspace.is_dir():
            raise FileNotFoundError(f"Scratch workspace is unavailable: {workspace}")
        # The flow can resume on scratch. Discard only incomplete copies.
        for path in (backup, recovery_dir / "promoted"):
            if path.exists():
                _remove_workspace(path)
    else:
        # Staging had not started, was rolled back, or promotion had finished.
        if workspace.parent.exists():
            _remove_workspace(workspace.parent)
        _remove_workspace(recovery_dir)


def _workspace_owner(workspace: Path) -> Optional[str]:
    if workspace.name != "workspace":
        return None
    try:
        return json.loads((workspace.parent / WORKSPACE_MARKER).read_text())["working_directory"]
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _restore_workspace(workdir: Path, workspace: Path) -> None:
    """Return products to the durable path, preserving scratch on promotion failure."""
    if not workdir.is_symlink() or workdir.resolve() != workspace.resolve():
        raise RuntimeError(f"Operation scratch link changed while the flow was running: {workdir}")
    _rewrite_workspace_links(workspace, workspace)
    recovery_dir = _recovery_directory(workdir)
    try:
        workdir.unlink()
        workspace.rename(workdir)
    except BaseException as error:
        if not workdir.exists() and not workdir.is_symlink():
            workdir.symlink_to(workspace, target_is_directory=True)
        if not isinstance(error, OSError) or error.errno != errno.EXDEV:
            raise
    else:
        _remove_workspace(workspace.parent)
        _remove_workspace(recovery_dir)
        return
    # Keep the scratch source until promotion succeeds, including when copying
    # across filesystems fails. Every published path still resolves to it.
    promoted = recovery_dir / "promoted"
    try:
        _copy_workspace(workspace, promoted)
        workdir.unlink()
        try:
            promoted.rename(workdir)
        except BaseException:
            workdir.symlink_to(workspace, target_is_directory=True)
            raise
    finally:
        if promoted.exists():
            _remove_workspace(promoted)
    _remove_workspace(workspace.parent)
    _remove_workspace(recovery_dir)


def _copy_workspace(source: Path, destination: Path) -> None:
    shutil.copytree(source, destination, symlinks=True)
    _rewrite_workspace_links(source, destination)


def _rewrite_workspace_links(source: Path, destination: Path) -> None:
    # Internal links must move with the workspace; external links must keep
    # their original targets regardless of the workspace's parent directory.
    for link in source.rglob("*"):
        if not link.is_symlink():
            continue
        target = Path(os.readlink(link))
        absolute_target = Path(os.path.abspath(link.parent / target))
        if absolute_target.is_relative_to(source):
            if not target.is_absolute():
                continue
            target = Path(os.path.relpath(absolute_target, link.parent))
        else:
            if target.is_absolute():
                continue
            target = absolute_target
        copied_link = destination / link.relative_to(source)
        mode = stat.S_IMODE(copied_link.parent.stat().st_mode)
        try:
            copied_link.parent.chmod(mode | stat.S_IWUSR)
            with tempfile.TemporaryDirectory(
                prefix=".rapthor-link-", dir=copied_link.parent
            ) as root:
                replacement = Path(root) / "link"
                replacement.symlink_to(target, target_is_directory=link.is_dir())
                replacement.replace(copied_link)
        finally:
            copied_link.parent.chmod(mode)


def _remove_workspace(directory: Path) -> None:
    """Remove an owned tree, including read-only copied input directories."""
    try:
        shutil.rmtree(directory)
    except PermissionError:
        for path, _, _ in os.walk(directory):
            parent = Path(path)
            parent.chmod(parent.stat().st_mode | stat.S_IWUSR)
        shutil.rmtree(directory)

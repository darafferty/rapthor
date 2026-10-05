"""Scratch workspace routing, promotion, and restart contracts."""

import errno
import json
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from rapthor.execution.workspace import (
    WORKSPACE_MARKER,
    recover_scratch_workspace,
    shared_scratch_workspace,
)


def _force_cross_device_promotion(monkeypatch, scratch):
    rename = Path.rename

    def cross_device_rename(source, destination):
        if source.parent.parent == scratch:
            raise OSError(errno.EXDEV, "different filesystems")
        return rename(source, destination)

    monkeypatch.setattr(Path, "rename", cross_device_rename)


def test_unconfigured_scratch_keeps_operation_directory(tmp_path):
    workdir = tmp_path / "pipelines" / "image_1"
    with shared_scratch_workspace(str(workdir), None):
        assert not workdir.exists()


@pytest.mark.parametrize("cross_device", [False, True])
def test_shared_scratch_preserves_paths_and_promotes_products(tmp_path, monkeypatch, cross_device):
    workdir = tmp_path / "working" / "pipelines" / "image_1"
    scratch = tmp_path / "scratch"
    workdir.mkdir(parents=True)
    (workdir / "pipeline_inputs.json").write_text("{}")
    model = workdir / "model.fits"
    model.write_text("model")
    (workdir / "internal.fits").symlink_to(model.name)
    (workdir / "absolute-internal.fits").symlink_to(model)
    external = workdir.parent / "external.fits"
    external.write_text("external")
    (workdir / "external.fits").symlink_to("../external.fits")
    read_only = workdir / "read-only"
    read_only.mkdir()
    (read_only / "external.fits").symlink_to("../../external.fits")
    read_only.chmod(0o555)
    if cross_device:
        _force_cross_device_promotion(monkeypatch, scratch)

    with shared_scratch_workspace(str(workdir), str(scratch)):
        assert workdir.is_symlink()
        assert workdir.resolve().is_relative_to(scratch)
        assert (workdir / "pipeline_inputs.json").read_text() == "{}"
        assert (workdir / "internal.fits").read_text() == "model"
        assert (workdir / "absolute-internal.fits").read_text() == "model"
        (workdir / "resolved-internal.fits").symlink_to(model.resolve())
        assert (workdir / "external.fits").read_text() == "external"
        assert (read_only / "external.fits").read_text() == "external"
        shared_model = scratch / "shared-model.fits"
        shared_model.write_text("shared")
        generated_link = workdir / "generated-link.fits"
        generated_link.symlink_to("../../shared-model.fits")
        assert generated_link.read_text() == "shared"
        output = workdir / "result.ms"
        output.mkdir()
        (output / "table.dat").write_text("data")
        record = {"class": "Directory", "path": str(output)}

    assert not workdir.is_symlink()
    assert Path(record["path"]).is_dir()
    assert (output / "table.dat").read_text() == "data"
    assert (workdir / "internal.fits").read_text() == "model"
    assert (workdir / "absolute-internal.fits").read_text() == "model"
    assert (workdir / "resolved-internal.fits").read_text() == "model"
    assert (workdir / "external.fits").read_text() == "external"
    assert (read_only / "external.fits").read_text() == "external"
    assert generated_link.read_text() == "shared"
    assert stat.S_IMODE(read_only.stat().st_mode) == 0o555
    read_only.chmod(0o755)
    assert list(scratch.iterdir()) == [shared_model]
    assert not (workdir.parent / ".image_1.scratch").exists()


@pytest.mark.parametrize(
    "stage",
    ["copy", "backup", "link", "flow", "unlink", "cross-copy", "cross-rename", "rename"],
)
def test_restart_recovers_after_process_exit_at_directory_transitions(tmp_path, stage):
    """Exercise actual process death, which cannot run exception/finally handlers."""
    workdir = tmp_path / "pipelines" / "image_1"
    scratch = tmp_path / "scratch"
    workdir.mkdir(parents=True)
    original = workdir / "input.fits"
    original.write_text("input")
    original_mtime = original.stat().st_mtime_ns
    child = r"""
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
    result = subprocess.run(
        [sys.executable, "-c", child, str(workdir), str(scratch), stage],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 73, result.stderr

    # Recovery must also work when the next parset disables global scratch.
    recover_scratch_workspace(str(workdir))
    with shared_scratch_workspace(str(workdir), None):
        assert original.read_text() == "input"
        assert original.stat().st_mtime_ns == original_mtime
        if stage not in {"copy", "backup", "link"}:
            assert (workdir / "result-link.fits").read_text() == "result"

    assert not workdir.is_symlink()
    assert list(scratch.iterdir()) == []
    assert list(workdir.parent.iterdir()) == [workdir]


def test_recovery_refuses_a_record_for_another_operation(tmp_path):
    workdir = tmp_path / "image_1"
    workspace = tmp_path / "another-run" / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "input.fits").write_text("another run")
    recovery_dir = tmp_path / ".image_1.scratch"
    recovery_dir.mkdir()
    (recovery_dir / WORKSPACE_MARKER).write_text(
        json.dumps({"working_directory": str(tmp_path / "other"), "workspace": str(workspace)})
    )

    with pytest.raises(RuntimeError, match="Unowned scratch recovery record"):
        recover_scratch_workspace(str(workdir))

    assert (workspace / "input.fits").read_text() == "another run"


def test_failure_preserves_partial_products_for_restart(tmp_path):
    workdir = tmp_path / "pipelines" / "calibrate_1"
    scratch = tmp_path / "scratch"
    with pytest.raises(RuntimeError, match="solve failed"):
        with shared_scratch_workspace(str(workdir), str(scratch)):
            (workdir / "solve.h5").write_text("completed solve")
            raise RuntimeError("solve failed")

    assert not workdir.is_symlink()
    assert (workdir / "solve.h5").read_text() == "completed solve"
    assert list(scratch.iterdir()) == []


def test_independent_runs_share_scratch_without_cleanup_collisions(tmp_path):
    scratch = tmp_path / "scratch"
    first = tmp_path / "first" / "pipelines" / "image_1"
    second = tmp_path / "second" / "pipelines" / "image_1"
    scratch.mkdir()
    unrelated = scratch / "unrelated.txt"
    unrelated.write_text("other job")

    with shared_scratch_workspace(str(first), str(scratch)):
        first_workspace = first.resolve()
        (first / "first.fits").write_text("first")
        with shared_scratch_workspace(str(second), str(scratch)):
            assert second.resolve() != first_workspace
            (second / "second.fits").write_text("second")
        assert first_workspace.is_dir()
        assert (first / "first.fits").read_text() == "first"

    assert (second / "second.fits").read_text() == "second"
    assert list(scratch.iterdir()) == [unrelated]


@pytest.mark.parametrize("configured", [False, True])
def test_recovers_workspace_after_interrupted_flow(tmp_path, configured):
    workdir = tmp_path / "pipelines" / "predict_1"
    scratch = tmp_path / "scratch"
    workspace = scratch / "rapthor-interrupted" / "workspace"
    workspace.mkdir(parents=True)
    (workspace.parent / WORKSPACE_MARKER).write_text(
        json.dumps({"working_directory": str(workdir)})
    )
    (workspace / "model.ms").mkdir()
    workdir.parent.mkdir()
    workdir.symlink_to(workspace, target_is_directory=True)

    with shared_scratch_workspace(str(workdir), str(scratch) if configured else None):
        assert workdir.resolve() == workspace
        assert (workdir / "model.ms").is_dir()

    assert not workdir.is_symlink()
    assert (workdir / "model.ms").is_dir()
    assert list(scratch.iterdir()) == []


def test_rejects_unowned_operation_symlink(tmp_path):
    workdir = tmp_path / "pipelines" / "image_1"
    original = tmp_path / "user-workspace"
    original.mkdir()
    workdir.parent.mkdir()
    workdir.symlink_to(original, target_is_directory=True)

    with pytest.raises(ValueError, match="unowned operation symlink"):
        with shared_scratch_workspace(str(workdir), str(tmp_path / "scratch")):
            pytest.fail("unowned workspace must not be replaced")

    assert workdir.resolve() == original


@pytest.mark.parametrize("suffix", ["", "scratch"])
def test_rejects_scratch_inside_operation_directory(tmp_path, suffix):
    workdir = tmp_path / "pipelines" / "image_1"
    with pytest.raises(ValueError, match="outside the operation"):
        with shared_scratch_workspace(str(workdir), str(workdir / suffix)):
            pytest.fail("nested scratch must be rejected")


def test_scratch_root_may_be_a_symlink(tmp_path):
    scratch = tmp_path / "real-scratch"
    scratch.mkdir()
    alias = tmp_path / "scratch"
    alias.symlink_to(scratch, target_is_directory=True)
    workdir = tmp_path / "pipelines" / "image_1"

    with shared_scratch_workspace(str(workdir), str(alias)):
        (workdir / "result.fits").write_text("image")

    assert (workdir / "result.fits").read_text() == "image"
    assert not workdir.is_symlink()
    assert list(scratch.iterdir()) == []


def test_failed_promotion_keeps_source_discoverable(tmp_path, monkeypatch):
    from rapthor.execution import workspace as module

    workdir = tmp_path / "pipelines" / "image_1"
    scratch = tmp_path / "scratch"
    copy_workspace = module._copy_workspace
    _force_cross_device_promotion(monkeypatch, scratch)

    def fail_promotion(source, destination):
        if source.parent.parent == scratch:
            destination.mkdir()
            raise OSError("disk full")
        return copy_workspace(source, destination)

    monkeypatch.setattr(module, "_copy_workspace", fail_promotion)
    with pytest.raises(OSError, match="disk full"):
        with shared_scratch_workspace(str(workdir), str(scratch)):
            (workdir / "result.fits").write_text("image")

    assert workdir.is_symlink()
    assert (workdir / "result.fits").read_text() == "image"
    monkeypatch.setattr(module, "_copy_workspace", copy_workspace)
    with shared_scratch_workspace(str(workdir), str(scratch)):
        assert (workdir / "result.fits").read_text() == "image"
    assert not workdir.is_symlink()
    assert list(scratch.iterdir()) == []


@pytest.mark.parametrize("error", [PermissionError, KeyboardInterrupt])
def test_failed_rename_keeps_source_discoverable(tmp_path, monkeypatch, error):
    workdir = tmp_path / "pipelines" / "image_1"
    scratch = tmp_path / "scratch"
    rename = Path.rename

    def fail_promotion(source, destination):
        if source.parent.parent == scratch:
            raise error("promotion interrupted")
        return rename(source, destination)

    monkeypatch.setattr(Path, "rename", fail_promotion)
    with pytest.raises(error, match="promotion interrupted"):
        with shared_scratch_workspace(str(workdir), str(scratch)):
            (workdir / "result.fits").write_text("image")

    assert workdir.is_symlink()
    assert (workdir / "result.fits").read_text() == "image"
    monkeypatch.setattr(Path, "rename", rename)
    with shared_scratch_workspace(str(workdir), None):
        assert (workdir / "result.fits").read_text() == "image"
    assert not workdir.is_symlink()
    assert list(scratch.iterdir()) == []


def test_failed_link_rewrite_preserves_original_link(tmp_path, monkeypatch):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    shared_model = scratch / "model.fits"
    shared_model.write_text("model")
    workdir = tmp_path / "pipelines" / "image_1"
    symlink_to = Path.symlink_to

    def fail_replacement(link, *args, **kwargs):
        if link.parent.name.startswith(".rapthor-link-"):
            raise OSError("disk full")
        return symlink_to(link, *args, **kwargs)

    monkeypatch.setattr(Path, "symlink_to", fail_replacement)
    with pytest.raises(OSError, match="disk full"):
        with shared_scratch_workspace(str(workdir), str(scratch)):
            (workdir / "model.fits").symlink_to("../../model.fits")

    assert (workdir / "model.fits").read_text() == "model"
    monkeypatch.setattr(Path, "symlink_to", symlink_to)
    with shared_scratch_workspace(str(workdir), None):
        assert (workdir / "model.fits").read_text() == "model"
    assert (workdir / "model.fits").read_text() == "model"


@pytest.mark.parametrize("cross_device", [False, True])
def test_measurement_set_tables_survive_scratch_promotion(tmp_path, monkeypatch, cross_device):
    import casacore.tables as pt

    workdir = tmp_path / "pipelines" / "predict_1"
    scratch = tmp_path / "scratch"
    if cross_device:
        _force_cross_device_promotion(monkeypatch, scratch)
    with shared_scratch_workspace(str(workdir), str(scratch)):
        ms_path = str(workdir / "input.ms")
        with pt.default_ms(ms_path) as table:
            table.addrows(1)
            table.putcol("TIME", [42.0])
            selection = table.query("TIME >= 0", name=str(workdir / "reference.ms"))
            selection.close()

    for path in (workdir / "input.ms", workdir / "reference.ms"):
        with pt.table(str(path), ack=False) as table:
            assert list(table.getcol("TIME")) == [42.0]
            with pt.table(table.getkeyword("ANTENNA"), ack=False) as antenna:
                assert Path(antenna.name()).is_relative_to(workdir)

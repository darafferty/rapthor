from pathlib import Path

import pytest

from rapthor.execution.config import ExecutionConfig
from rapthor.execution.scratch import task_temporary_directory


@pytest.mark.parametrize(
    "local, global_, mpi, expected",
    [
        (True, True, False, "local"),
        (False, True, False, None),
        (True, True, True, "global"),
        (True, False, True, "work"),
    ],
)
def test_task_scratch_matches_master_directory_precedence(tmp_path, local, global_, mpi, expected):
    config = ExecutionConfig(
        local_scratch_dir=str(tmp_path / "local") if local else None,
        global_scratch_dir=str(tmp_path / "global") if global_ else None,
    )
    fallback = str(tmp_path / "work" / "wsclean_tmp") if mpi else None
    with task_temporary_directory(config, use_mpi=mpi, fallback_path=fallback) as temp:
        if expected is None:
            assert temp is None
            return
        directory = Path(temp)
        assert directory.parent == tmp_path / expected
        assert directory.is_dir()
    assert not directory.exists()
    assert directory.parent.is_dir()


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("keep", [False, True])
def test_task_scratch_is_isolated_and_retained_only_when_requested(tmp_path, fail, keep):
    root = tmp_path / "scratch"
    root.mkdir()
    unrelated = root / "another-run"
    unrelated.mkdir()
    config = ExecutionConfig(local_scratch_dir=str(root), keep_temporary_files=keep)
    directories = []

    def run_task():
        with task_temporary_directory(config, name="sector_1") as first:
            with task_temporary_directory(config, name="sector_1") as second:
                directories.extend([Path(first), Path(second)])
                assert first != second
                (Path(first) / "temporary").write_text("data")
                if fail:
                    raise RuntimeError("command failed")

    if fail:
        with pytest.raises(RuntimeError, match="command failed"):
            run_task()
    else:
        run_task()

    assert unrelated.is_dir()
    assert all(directory.exists() is keep for directory in directories)


def test_unconfigured_wsclean_scratch_preserves_existing_fallback(tmp_path):
    fallback = tmp_path / "sector_1_wsclean_tmp"
    with task_temporary_directory(ExecutionConfig(), fallback_path=str(fallback)) as directory:
        assert directory == str(fallback)
        assert fallback.is_dir()
    assert not fallback.exists()


def test_local_scratch_expands_variables_on_worker(tmp_path, monkeypatch):
    monkeypatch.delenv("RAPTHOR_WORKER_SCRATCH", raising=False)
    config = ExecutionConfig.from_parset(
        {"cluster_specific": {"local_scratch_dir": "$RAPTHOR_WORKER_SCRATCH"}}
    )
    worker_root = tmp_path / "node-local"
    monkeypatch.setenv("RAPTHOR_WORKER_SCRATCH", str(worker_root))

    with task_temporary_directory(config) as temporary:
        assert Path(temporary).parent == worker_root
        assert Path(temporary).is_dir()
    assert list(worker_root.iterdir()) == []


def test_mpi_requires_shared_temporary_storage():
    with pytest.raises(ValueError, match="shared fallback path"):
        with task_temporary_directory(ExecutionConfig(), use_mpi=True):
            pytest.fail("MPI must not use system or node-local temporary storage")


@pytest.mark.parametrize("command_fails", [False, True])
def test_cleanup_failure_is_reported_without_masking_command_failure(
    tmp_path, monkeypatch, caplog, command_fails
):
    from rapthor.execution import scratch

    def fail_cleanup(path):
        raise PermissionError("permission denied")

    monkeypatch.setattr(scratch.shutil, "rmtree", fail_cleanup)
    config = ExecutionConfig(local_scratch_dir=str(tmp_path))

    def run():
        with scratch.task_temporary_directory(config) as directory:
            if command_fails:
                raise RuntimeError("command failed")
            return directory

    if command_fails:
        with pytest.raises(RuntimeError, match="command failed"):
            run()
    else:
        assert Path(run()).is_dir()
    assert "Could not remove command scratch" in caplog.text
    assert str(tmp_path) in caplog.text
    assert "permission denied" in caplog.text

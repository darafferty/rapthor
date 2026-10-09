from datetime import datetime, timezone

from rapthor.execution.task_metrics import task_log_path, write_task_log_record


def test_task_log_path_uses_pipeline_working_directory():
    assert task_log_path("/work/pipelines/image_1").as_posix() == "/work/logs/tasks.jsonl"
    assert task_log_path("/work/image_1") is None


def test_write_task_log_record_appends_prefect_task_metadata(tmp_path):
    pipeline_dir = tmp_path / "pipelines" / "calibrate_1"
    pipeline_dir.mkdir(parents=True)
    timestamp = datetime(2026, 7, 11, tzinfo=timezone.utc)

    log_path = write_task_log_record(
        pipeline_dir,
        started_at=timestamp,
        finished_at=timestamp,
        duration_seconds=1.25,
        status="completed",
        task_metadata={
            "task_name": "chunk",
            "task_run_name": "solve_chunk_1",
            "task_tags": ["dp3"],
        },
    )

    assert log_path == tmp_path / "logs" / "tasks.jsonl"
    assert '"operation": "calibrate_1"' in log_path.read_text(encoding="utf-8")
    assert '"task_run_name": "solve_chunk_1"' in log_path.read_text(encoding="utf-8")
    assert '"task_tags": ["dp3"]' in log_path.read_text(encoding="utf-8")

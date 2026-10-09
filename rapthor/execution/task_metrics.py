"""Lightweight task runtime records for Prefect/Dask observability."""

import json
import logging
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Mapping, Optional

TASK_LOG_FILENAME = "tasks.jsonl"

log = logging.getLogger("rapthor:task-metrics")


def task_log_path(pipeline_working_dir: Optional[object]) -> Optional[Path]:
    """Return the task-runtime log path for an operation workdir."""
    if pipeline_working_dir is None:
        return None
    workdir = Path(str(pipeline_working_dir))
    if workdir.parent.name != "pipelines":
        return None
    return workdir.parent.parent / "logs" / TASK_LOG_FILENAME


def current_prefect_task_metadata() -> dict[str, object]:
    """Return task-run metadata from Prefect when running inside a task."""
    try:
        from prefect.context import get_run_context

        context = get_run_context()
    except Exception:
        return {}

    task_run = getattr(context, "task_run", None)
    task = getattr(context, "task", None)
    metadata: dict[str, object] = {}
    if task_run is not None:
        task_run_name = getattr(task_run, "name", None)
        task_run_id = getattr(task_run, "id", None)
        if task_run_name:
            metadata["task_run_name"] = str(task_run_name)
        if task_run_id:
            metadata["task_run_id"] = str(task_run_id)
        tags = getattr(task_run, "tags", None)
        if tags:
            metadata["task_tags"] = sorted(str(tag) for tag in tags)
    if task is not None:
        task_name = getattr(task, "name", None)
        if task_name:
            metadata["task_name"] = str(task_name)
    return metadata


def write_task_log_record(
    pipeline_working_dir: Optional[object],
    *,
    started_at: datetime,
    finished_at: datetime,
    duration_seconds: float,
    status: str,
    error: Optional[str] = None,
    task_metadata: Optional[Mapping[str, object]] = None,
) -> Optional[Path]:
    """Append a structured task-runtime record beside command records."""
    log_path = task_log_path(pipeline_working_dir)
    if log_path is None:
        return None

    record = _task_log_record(
        pipeline_working_dir,
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=duration_seconds,
        status=status,
        error=error,
        task_metadata=task_metadata or current_prefect_task_metadata(),
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return log_path


@contextmanager
def record_task_runtime(pipeline_working_dir: Optional[object]) -> Iterator[None]:
    """Record wall-clock runtime for the current Prefect task, when possible."""
    metadata = current_prefect_task_metadata()
    started_at = datetime.now(timezone.utc)
    started_monotonic = time.monotonic()
    try:
        yield
    except Exception as err:
        finished_at = datetime.now(timezone.utc)
        write_task_log_record(
            pipeline_working_dir,
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=time.monotonic() - started_monotonic,
            status="failed",
            error=str(err),
            task_metadata=metadata,
        )
        raise
    else:
        finished_at = datetime.now(timezone.utc)
        write_task_log_record(
            pipeline_working_dir,
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=time.monotonic() - started_monotonic,
            status="completed",
            task_metadata=metadata,
        )


def _task_log_record(
    pipeline_working_dir: object,
    *,
    started_at: datetime,
    finished_at: datetime,
    duration_seconds: float,
    status: str,
    error: Optional[str] = None,
    task_metadata: Optional[Mapping[str, object]] = None,
) -> dict[str, object]:
    workdir = Path(str(pipeline_working_dir))
    record: dict[str, object] = {
        "backend": "prefect",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "operation": workdir.name,
        "pipeline_working_dir": str(workdir),
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": round(duration_seconds, 6),
        "status": status,
    }
    if task_metadata:
        record.update(
            {key: value for key, value in task_metadata.items() if value not in (None, "", [])}
        )
    if error:
        record["error"] = error
    return record

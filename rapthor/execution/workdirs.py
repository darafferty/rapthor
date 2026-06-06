"""Working-directory helpers for command tasks."""

import contextlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _safe_part(value: object) -> str:
    text = str(value).strip().replace(os.sep, "_")
    return text or "task"


@dataclass(frozen=True)
class WorkDirectoryLayout:
    """Deterministic working-directory layout for operation tasks."""

    root: Path
    operation_name: str

    @classmethod
    def from_paths(cls, root: object, operation_name: str) -> "WorkDirectoryLayout":
        return cls(root=Path(root), operation_name=operation_name)

    @property
    def operation_dir(self) -> Path:
        return self.root / "pipelines" / _safe_part(self.operation_name)

    def task_dir(self, task_name: str, *parts: object) -> Path:
        path = self.operation_dir / "tasks" / _safe_part(task_name)
        for part in parts:
            path = path / _safe_part(part)
        return path

    def ensure_task_dir(self, task_name: str, *parts: object) -> Path:
        path = self.task_dir(task_name, *parts)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def cleanup_task_dir(self, task_name: str, *parts: object) -> None:
        path = self.task_dir(task_name, *parts)
        if path.is_dir():
            shutil.rmtree(path)

    @contextlib.contextmanager
    def task_scope(
        self,
        task_name: str,
        *parts: object,
        cleanup: bool = True,
        cleanup_on_failure: bool = False,
    ):
        path = self.ensure_task_dir(task_name, *parts)
        try:
            yield path
        except Exception:
            if cleanup and cleanup_on_failure:
                self.cleanup_task_dir(task_name, *parts)
            raise
        else:
            if cleanup:
                self.cleanup_task_dir(task_name, *parts)

    def temp_path(self, final_path: object) -> Path:
        final = Path(final_path)
        return final.with_name(f".{final.name}.tmp")

    def atomic_write_text(self, final_path: object, content: str) -> Path:
        final = Path(final_path)
        final.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.temp_path(final)
        tmp.write_text(content)
        tmp.replace(final)
        return final

    def atomic_write_json(self, final_path: object, content: Any) -> Path:
        return self.atomic_write_text(final_path, json.dumps(content, indent=4, sort_keys=True))

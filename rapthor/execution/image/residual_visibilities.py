"""Residual visibility products for image sectors."""

import os
from typing import Mapping, Optional

from rapthor.execution.config import ExecutionConfig
from rapthor.execution.image.commands import build_make_residual_visibilities_command
from rapthor.execution.image.contracts import ImageSectorPayload
from rapthor.execution.outputs import require_directory
from rapthor.execution.shell import run_external_command


def make_residual_visibility_record(
    sector: ImageSectorPayload,
    concat_record: Mapping[str, str],
    pipeline_working_dir: str,
    execution_config: ExecutionConfig,
    *,
    force: bool = False,
    shell_operation_cls=None,
) -> Optional[dict]:
    """Create a residual Measurement Set from DATA minus WSClean MODEL_DATA."""
    if not sector["make_residual_visibilities"]:
        return None

    if sector["residual_filename"] is None or sector["residual_path"] is None:
        raise ValueError("Residual visibility filename is required when residuals are requested")

    residual_path = str(sector["residual_path"])
    if force or not os.path.isdir(residual_path):
        command = build_make_residual_visibilities_command(
            msin=concat_record["path"],
            msout=str(sector["residual_filename"]),
            numthreads=int(sector["max_threads"]),
        )
        run_external_command(
            command,
            pipeline_working_dir,
            execution_config,
            shell_operation_cls=shell_operation_cls,
        )

    return require_directory(residual_path, "Residual visibility MS")

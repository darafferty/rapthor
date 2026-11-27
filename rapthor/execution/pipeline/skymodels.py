"""Prefect task for the sky-model processing of a cycle."""

import os
from typing import Optional

from prefect import task
from prefect.cache_policies import NO_CACHE

from rapthor.execution.prefect_context import in_prefect_run_context
from rapthor.execution.prefect_logging import publish_python_logs_to_prefect
from rapthor.execution.run_names import task_run_options
from rapthor.execution.task_metrics import record_task_runtime

SKYMODEL_TASK_NAME = "process_skymodels"


def skymodel_records_dir(working_dir: object, index: int) -> str:
    """Return the directory name under which the task records its runtime.

    The directory is not created: it only names the records in
    ``logs/tasks.jsonl``, as ``skymodels_<cycle>``.
    """
    return os.path.join(str(working_dir), "pipelines", f"skymodels_{index}")


# The task changes the Field in place, so it is called directly from the
# pipeline flow and runs in the rapthor process, never on a Dask worker. It is
# not cached, since its inputs are live run state
@task(name=SKYMODEL_TASK_NAME, cache_policy=NO_CACHE)
def process_skymodels_task(
    field: object,
    index: int,
    regroup: bool,
    target_flux: Optional[float],
    target_number: Optional[int],
    calibrator_max_dist_deg: Optional[float],
    combine_current_and_initial: bool,
    pipeline_working_dir: str,
) -> None:
    """Prefect task wrapper for Field.process_skymodels()."""
    with publish_python_logs_to_prefect(), record_task_runtime(pipeline_working_dir):
        field.process_skymodels(
            index,
            regroup,
            target_flux=target_flux,
            target_number=target_number,
            calibrator_max_dist_deg=calibrator_max_dist_deg,
            combine_current_and_initial=combine_current_and_initial,
        )


def run_skymodel_processing(field: object, **parameters) -> None:
    """Run the sky-model processing of a cycle as a Prefect task.

    Used as the ``skymodel_runner`` of Field.update(). Outside a Prefect run (for
    example, when the pipeline steps are called directly in tests), the
    processing is run without a task.
    """
    if not in_prefect_run_context():
        field.process_skymodels(**parameters)
        return

    index = parameters["index"]
    process_skymodels_task.with_options(
        **task_run_options(SKYMODEL_TASK_NAME, index, tags=["python"])
    )(
        field,
        pipeline_working_dir=skymodel_records_dir(field.working_dir, index),
        **parameters,
    )

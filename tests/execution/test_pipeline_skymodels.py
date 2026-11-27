import json
import threading

import pytest
from prefect import flow

from rapthor.execution.pipeline.skymodels import run_skymodel_processing

SKYMODEL_PARAMETERS = {
    "index": 2,
    "regroup": True,
    "target_flux": 0.3,
    "target_number": 10,
    "calibrator_max_dist_deg": 3.0,
    "combine_current_and_initial": False,
}


class RecordingField:
    """Field stand-in that records the sky-model processing it is asked to do."""

    def __init__(self, working_dir, error=None):
        self.working_dir = str(working_dir)
        self.error = error
        self.calls = []
        # A lock cannot be pickled, so the task fails if the field is ever
        # serialized for a worker
        self.lock = threading.Lock()

    def process_skymodels(self, index, regroup, **kwargs):
        if self.error is not None:
            raise self.error
        self.calls.append({"index": index, "regroup": regroup, **kwargs})
        self.num_patches = 4


def _task_records(working_dir):
    path = working_dir / "logs" / "tasks.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_run_skymodel_processing_outside_prefect_calls_the_field_directly(tmp_path):
    field = RecordingField(tmp_path)

    run_skymodel_processing(field, **SKYMODEL_PARAMETERS)

    assert field.calls == [SKYMODEL_PARAMETERS]
    assert not (tmp_path / "logs" / "tasks.jsonl").exists()


def test_run_skymodel_processing_runs_a_recorded_task_on_the_same_field(tmp_path, prefect_server):
    field = RecordingField(tmp_path)

    @flow
    def pipeline():
        run_skymodel_processing(field, **SKYMODEL_PARAMETERS)

    pipeline()

    # The task changes the field of the pipeline, not a copy of it
    assert field.calls == [SKYMODEL_PARAMETERS]
    assert field.num_patches == 4
    [record] = _task_records(tmp_path)
    assert record["operation"] == "skymodels_2"
    assert record["task_name"] == "process_skymodels"
    assert record["task_run_name"] == "process_skymodels_2"
    assert record["task_tags"] == ["python"]
    assert record["status"] == "completed"
    assert not (tmp_path / "pipelines").exists()


def test_run_skymodel_processing_records_a_failed_task(tmp_path, prefect_server):
    field = RecordingField(tmp_path, error=ValueError("bad sky model"))

    @flow
    def pipeline():
        run_skymodel_processing(field, **SKYMODEL_PARAMETERS)

    with pytest.raises(ValueError, match="bad sky model"):
        pipeline()

    [record] = _task_records(tmp_path)
    assert record["status"] == "failed"
    assert record["error"] == "bad sky model"

from copy import deepcopy

import rapthor.execution.calibrate.flow as calibrate_module
import rapthor.execution.concatenate.flow as concatenate_module
import rapthor.execution.image.flow as image_module
import rapthor.execution.mosaic.flow as mosaic_module
import rapthor.execution.predict.flow as predict_module
from rapthor.execution.config import ExecutionConfig
from rapthor.execution.payloads import assert_serializable_payload
from rapthor.lib.records import directory_record, file_record
from tests.execution.payload_factories import (
    representative_calibrate_payload,
    representative_concatenate_payload,
    representative_image_payload,
    representative_mosaic_payload,
    representative_predict_payload,
)


class _CapturedFuture:
    def __init__(self, result):
        self._result = result

    def result(self):
        return deepcopy(self._result)


class _CapturedTask:
    def __init__(self, result_factory):
        self._result_factory = result_factory
        self.submissions = []

    def with_options(self, **options):
        return _ConfiguredCapturedTask(self, options)


class _ConfiguredCapturedTask:
    def __init__(self, task: _CapturedTask, options: dict):
        self._task = task
        self._options = options

    def submit(self, *args, **kwargs):
        submission = {
            "args": args,
            "kwargs": kwargs,
            "options": self._options,
        }
        self._task.submissions.append(submission)
        return _CapturedFuture(self._task._result_factory(len(self._task.submissions) - 1))


def _assert_worker_submission_is_serializable(
    submission: dict,
    execution_config: ExecutionConfig,
) -> None:
    """Worker task inputs stay plain; runtime config travels as a separate object."""
    for arg in submission["args"]:
        assert_serializable_payload(arg)
    for name, value in submission["kwargs"].items():
        if name == "execution_config":
            assert value == execution_config
        else:
            assert_serializable_payload(value)


def _image_sector_result(index: int) -> dict:
    root = f"/work/image_1/sector_{index + 1}"
    return {
        "filtered_skymodel_true_sky": file_record(f"{root}.true_sky.txt"),
        "filtered_skymodel_apparent_sky": file_record(f"{root}.apparent_sky.txt"),
        "pybdsf_catalog": file_record(f"{root}.catalog.fits"),
        "sector_diagnostics": file_record(f"{root}.image_diagnostics.json"),
        "sector_offsets": file_record(f"{root}.astrometry_offsets.json"),
        "sector_diagnostic_plots": [file_record(f"{root}.photometry.pdf")],
        "visibilities": [directory_record(f"{root}.prep.ms")],
        "sector_I_images": [file_record(f"{root}-I-image.fits")],
        "sector_extra_images": [],
        "source_filtering_mask": file_record(f"{root}.mask.fits"),
        "sector_skymodels": None,
    }


def test_image_flow_submits_one_plain_payload_per_sector(monkeypatch):
    config = ExecutionConfig(task_runner="sync")
    payload = representative_image_payload()
    image_task = _CapturedTask(_image_sector_result)
    monkeypatch.setattr(image_module, "image_sector_task", image_task)

    result = image_module._run_image_prefect_tasks(payload, execution_config=config)

    assert list(result)[:3] == [
        "filtered_skymodel_true_sky",
        "filtered_skymodel_apparent_sky",
        "pybdsf_catalog",
    ]
    assert [submission["options"]["task_run_name"] for submission in image_task.submissions] == [
        "image_1_sector_1",
    ]
    _assert_worker_submission_is_serializable(image_task.submissions[0], config)


def test_calibrate_flow_submits_plain_payloads_and_readable_chunk_names(monkeypatch):
    config = ExecutionConfig(task_runner="sync")
    payload = representative_calibrate_payload()
    calibrate_task = _CapturedTask(
        lambda index: {"solve1": file_record(f"/work/calibrate_1/solve{index + 1}.h5")}
    )
    collected = {}

    def fake_collect(payload_arg, solve_records, execution_config, shell_operation_cls=None):
        collected["payload"] = payload_arg
        collected["solve_records"] = solve_records
        collected["execution_config"] = execution_config
        collected["shell_operation_cls"] = shell_operation_cls
        return {"combined_solutions": file_record("/work/calibrate_1/combined.h5")}

    monkeypatch.setattr(calibrate_module, "calibrate_chunk_task", calibrate_task)
    monkeypatch.setattr(calibrate_module, "collect_plot_and_combine", fake_collect)

    result = calibrate_module._run_calibrate_prefect_tasks(payload, execution_config=config)

    assert result == {"combined_solutions": file_record("/work/calibrate_1/combined.h5")}
    assert collected == {
        "payload": payload,
        "solve_records": [{"solve1": file_record("/work/calibrate_1/solve1.h5")}],
        "execution_config": config,
        "shell_operation_cls": None,
    }
    assert [
        submission["options"]["task_run_name"] for submission in calibrate_task.submissions
    ] == [
        "calibrate_dd_1_chunk_1",
    ]
    _assert_worker_submission_is_serializable(calibrate_task.submissions[0], config)


def test_predict_flow_keeps_model_and_postprocess_worker_payloads_plain(monkeypatch):
    config = ExecutionConfig(task_runner="sync")
    payload = representative_predict_payload()
    model_task = _CapturedTask(
        lambda index: directory_record(f"/work/predict_1/model_{index + 1}.ms")
    )
    postprocess_task = _CapturedTask(
        lambda index: [directory_record(f"/work/predict_1/postprocess_{index + 1}.ms")]
    )
    monkeypatch.setattr(predict_module, "predict_model_data_task", model_task)
    monkeypatch.setattr(predict_module, "predict_postprocess_task", postprocess_task)

    result = predict_module._run_predict_prefect_tasks(payload, execution_config=config)

    assert result == {"msfiles_di_cal": [[directory_record("/work/predict_1/postprocess_1.ms")]]}
    assert [submission["options"]["task_run_name"] for submission in model_task.submissions] == [
        "predict_di_1_model_1",
    ]
    assert [
        submission["options"]["task_run_name"] for submission in postprocess_task.submissions
    ] == [
        "predict_di_1_postprocess_1",
    ]
    _assert_worker_submission_is_serializable(model_task.submissions[0], config)
    _assert_worker_submission_is_serializable(postprocess_task.submissions[0], config)


def test_mosaic_flow_submits_plain_payloads_with_image_type_names(monkeypatch):
    config = ExecutionConfig(task_runner="sync")
    payload = representative_mosaic_payload()
    mosaic_task = _CapturedTask(
        lambda index: file_record(f"/work/mosaic_1/mosaic_{index + 1}-I-image.fits")
    )
    monkeypatch.setattr(mosaic_module, "mosaic_image_type_task", mosaic_task)

    result = mosaic_module._run_mosaic_prefect_tasks(payload, execution_config=config)

    assert result == {"mosaic_image": [file_record("/work/mosaic_1/mosaic_1-I-image.fits")]}
    assert [submission["options"]["task_run_name"] for submission in mosaic_task.submissions] == [
        "mosaic_1_image_type_1",
    ]
    _assert_worker_submission_is_serializable(mosaic_task.submissions[0], config)


def test_concatenate_flow_submits_plain_payloads_with_epoch_names(monkeypatch):
    config = ExecutionConfig(task_runner="sync")
    payload = representative_concatenate_payload()
    concatenate_task = _CapturedTask(
        lambda index: directory_record(f"/work/concatenate_1/epoch_{index}.ms")
    )
    monkeypatch.setattr(concatenate_module, "concatenate_epoch_task", concatenate_task)

    result = concatenate_module._run_concatenate_prefect_tasks(payload, execution_config=config)

    assert result == {
        "concatenated_filenames": [directory_record("/work/concatenate_1/epoch_0.ms")]
    }
    assert [
        submission["options"]["task_run_name"] for submission in concatenate_task.submissions
    ] == [
        "concatenate_1_epoch_1",
    ]
    _assert_worker_submission_is_serializable(concatenate_task.submissions[0], config)

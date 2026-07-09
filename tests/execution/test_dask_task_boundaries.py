from copy import deepcopy

import pytest

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
        self.futures = []

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
        future = _CapturedFuture(self._task._result_factory(len(self._task.submissions) - 1))
        self._task.futures.append(future)
        return future


def _assert_worker_submission_is_serializable(
    submission: dict,
    execution_config: ExecutionConfig,
) -> None:
    """Worker task inputs stay plain; runtime config travels as a separate object."""
    for arg in submission["args"]:
        if isinstance(arg, _CapturedFuture):
            assert_serializable_payload(arg.result())
        else:
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
        "sector_I_images": [
            file_record(f"{root}-I-image.fits"),
            file_record(f"{root}-I-image-pb.fits"),
            file_record(f"{root}-I-image-pb-ast.fits"),
        ],
        "sector_extra_images": [],
        "source_filtering_mask": file_record(f"{root}.mask.fits"),
        "sector_skymodels": None,
    }


def _image_sector_preparation_result(index: int) -> dict:
    root = f"/work/image_1/sector_{index + 1}"
    return {
        "prepared_records": [directory_record(f"{root}.prep.ms")],
        "concat_record": directory_record(f"{root}.concat.ms"),
        "mask_record": file_record(f"{root}.mask.fits"),
        "region_record": None,
        "nonpb_image": file_record(f"{root}-I-image.fits"),
        "pb_image": file_record(f"{root}-I-image-pb.fits"),
        "wsclean_ran": True,
        "extra_images": [],
        "residual_visibilities": None,
        "sector_images": [
            file_record(f"{root}-I-image.fits"),
            file_record(f"{root}-I-image-pb.fits"),
        ],
        "image_cubes": [],
        "image_cube_beams": [],
        "image_cube_frequencies": [],
        "skymodel_nonpb": None,
        "skymodel_pb": None,
    }


def _image_sector_filter_result(index: int) -> dict:
    root = f"/work/image_1/sector_{index + 1}"
    return {
        "filtered_true_sky": file_record(f"{root}.true_sky.txt"),
        "filtered_apparent_sky": file_record(f"{root}.apparent_sky.txt"),
        "diagnostics": file_record(f"{root}.image_diagnostics.json"),
        "flat_noise_rms": file_record(f"{root}.flat_noise_rms.fits"),
        "true_sky_rms": file_record(f"{root}.true_sky_rms.fits"),
        "source_catalog": file_record(f"{root}.source_catalog.fits"),
        "source_filtering_mask": file_record(f"{root}.mask.fits"),
    }


def _image_sector_diagnostics_result(index: int) -> dict:
    root = f"/work/image_1/sector_{index + 1}"
    return {
        "diagnostics": file_record(f"{root}.image_diagnostics.json"),
        "offsets": file_record(f"{root}.astrometry_offsets.json"),
        "diagnostic_plots": [file_record(f"{root}.photometry.pdf")],
    }


@pytest.mark.parametrize("task_runner", ["sync", "local_dask", "external_dask"])
def test_image_flow_submits_plain_sector_task_payloads(monkeypatch, task_runner):
    config = ExecutionConfig(task_runner=task_runner)
    payload = representative_image_payload()
    prepare_task = _CapturedTask(_image_sector_preparation_result)
    filter_task = _CapturedTask(_image_sector_filter_result)
    diagnostics_task = _CapturedTask(_image_sector_diagnostics_result)
    finalize_task = _CapturedTask(_image_sector_result)
    monkeypatch.setattr(image_module, "image_sector_prepare_task", prepare_task)
    monkeypatch.setattr(image_module, "image_sector_filter_skymodel_task", filter_task)
    monkeypatch.setattr(image_module, "image_sector_diagnostics_task", diagnostics_task)
    monkeypatch.setattr(image_module, "image_sector_finalize_task", finalize_task)

    result = image_module._run_image_prefect_tasks(payload, execution_config=config)

    assert list(result)[:3] == [
        "filtered_skymodel_true_sky",
        "filtered_skymodel_apparent_sky",
        "pybdsf_catalog",
    ]
    assert [submission["options"]["task_run_name"] for submission in prepare_task.submissions] == [
        "prepare",
    ]
    assert [submission["options"]["task_run_name"] for submission in filter_task.submissions] == [
        "filter_skymodel",
    ]
    assert [
        submission["options"]["task_run_name"] for submission in diagnostics_task.submissions
    ] == [
        "calculate_image_diagnostics",
    ]
    assert [submission["options"]["task_run_name"] for submission in finalize_task.submissions] == [
        "finalize",
    ]
    _assert_worker_submission_is_serializable(prepare_task.submissions[0], config)
    _assert_worker_submission_is_serializable(filter_task.submissions[0], config)
    _assert_worker_submission_is_serializable(diagnostics_task.submissions[0], config)
    _assert_worker_submission_is_serializable(finalize_task.submissions[0], config)
    assert filter_task.submissions[0]["args"][1] is prepare_task.futures[0]
    assert filter_task.submissions[0]["args"][1].result() == _image_sector_preparation_result(0)
    assert diagnostics_task.submissions[0]["args"][1] is prepare_task.futures[0]
    assert diagnostics_task.submissions[0]["args"][1].result() == _image_sector_preparation_result(
        0
    )
    assert diagnostics_task.submissions[0]["args"][2] is filter_task.futures[0]
    assert diagnostics_task.submissions[0]["args"][2].result() == _image_sector_filter_result(0)
    assert finalize_task.submissions[0]["args"][1] is prepare_task.futures[0]
    assert finalize_task.submissions[0]["args"][1].result() == _image_sector_preparation_result(0)
    assert finalize_task.submissions[0]["args"][2] is filter_task.futures[0]
    assert finalize_task.submissions[0]["args"][2].result() == _image_sector_filter_result(0)
    assert finalize_task.submissions[0]["args"][3] is diagnostics_task.futures[0]
    assert finalize_task.submissions[0]["args"][3].result() == _image_sector_diagnostics_result(0)


def test_image_flow_disambiguates_sector_task_names_for_multiple_sectors(monkeypatch):
    config = ExecutionConfig(task_runner="sync")
    payload = representative_image_payload()
    second_sector = deepcopy(payload["sectors"][0])
    second_sector["image_name"] = "sector_2"
    payload["sectors"].append(second_sector)
    prepare_task = _CapturedTask(_image_sector_preparation_result)
    filter_task = _CapturedTask(_image_sector_filter_result)
    diagnostics_task = _CapturedTask(_image_sector_diagnostics_result)
    finalize_task = _CapturedTask(_image_sector_result)
    monkeypatch.setattr(image_module, "image_sector_prepare_task", prepare_task)
    monkeypatch.setattr(image_module, "image_sector_filter_skymodel_task", filter_task)
    monkeypatch.setattr(image_module, "image_sector_diagnostics_task", diagnostics_task)
    monkeypatch.setattr(image_module, "image_sector_finalize_task", finalize_task)

    image_module._run_image_prefect_tasks(payload, execution_config=config)

    assert [submission["options"]["task_run_name"] for submission in prepare_task.submissions] == [
        "sector_1_prepare",
        "sector_2_prepare",
    ]
    assert [submission["options"]["task_run_name"] for submission in filter_task.submissions] == [
        "sector_1_filter_skymodel",
        "sector_2_filter_skymodel",
    ]
    assert [
        submission["options"]["task_run_name"] for submission in diagnostics_task.submissions
    ] == [
        "sector_1_calculate_image_diagnostics",
        "sector_2_calculate_image_diagnostics",
    ]
    assert [submission["options"]["task_run_name"] for submission in finalize_task.submissions] == [
        "sector_1_finalize",
        "sector_2_finalize",
    ]


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
        "chunk_1",
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
        "model_1",
    ]
    assert [
        submission["options"]["task_run_name"] for submission in postprocess_task.submissions
    ] == [
        "postprocess_1",
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
        "image_type_I",
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
        "epoch_1",
    ]
    _assert_worker_submission_is_serializable(concatenate_task.submissions[0], config)

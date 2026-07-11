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

    def resolved(value):
        if isinstance(value, _CapturedFuture):
            return value.result()
        if isinstance(value, list):
            return [resolved(item) for item in value]
        if isinstance(value, tuple):
            return tuple(resolved(item) for item in value)
        if isinstance(value, dict):
            return {key: resolved(item) for key, item in value.items()}
        return value

    for arg in submission["args"]:
        assert_serializable_payload(resolved(arg))
    for name, value in submission["kwargs"].items():
        if name == "execution_config":
            assert value == execution_config
        else:
            assert_serializable_payload(resolved(value))


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
        "skymodel_nonpb": None,
        "skymodel_pb": None,
    }


def _image_sector_prepared_visibility_result(index: int) -> dict:
    return directory_record(f"/work/image_1/obs_{index + 1}.prep.ms")


def _image_sector_wsclean_result(index: int) -> dict:
    root = f"/work/image_1/sector_{index + 1}"
    return {
        "mask_record": file_record(f"{root}.mask.fits"),
        "region_record": None,
        "nonpb_image": file_record(f"{root}-I-image.fits"),
        "pb_image": file_record(f"{root}-I-image-pb.fits"),
        "wsclean_ran": True,
        "extra_images": [],
        "sector_images": [
            file_record(f"{root}-I-image.fits"),
            file_record(f"{root}-I-image-pb.fits"),
        ],
        "skymodel_nonpb": None,
        "skymodel_pb": None,
    }


def _image_sector_no_residual_result(_index: int) -> dict:
    return {"residual_visibilities": None}


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


def _image_sector_cube_result(index: int) -> dict:
    root = f"/work/image_1/sector_{index + 1}"
    return {
        "image_cubes": [file_record(f"{root}_I_freq_cube.fits")],
        "image_cube_beams": [file_record(f"{root}_I_freq_cube.fits_beams.txt")],
        "image_cube_frequencies": [file_record(f"{root}_I_freq_cube.fits_frequencies.txt")],
    }


def _image_sector_normalization_result(index: int) -> dict:
    root = f"/work/image_1/sector_{index + 1}"
    return {
        "normalize_h5parm": file_record(f"{root}_normalize.h5parm"),
    }


def _image_sector_catalog_result(index: int) -> dict:
    root = f"/work/image_1/sector_{index + 1}"
    return {"source_catalog": file_record(f"{root}_source_catalog.fits")}


def _image_sector_restored_model_result(index: int) -> dict:
    root = f"/work/image_1/sector_{index + 1}"
    return {"skymodel_image": file_record(f"{root}_filtered-model.fits.fz")}


def _image_sector_compression_result(index: int) -> dict:
    root = f"/work/image_1/sector_{index + 1}"
    return {
        "sector_images": [
            file_record(f"{root}-I-image.fits.fz"),
            file_record(f"{root}-I-image-pb.fits.fz"),
            file_record(f"{root}-I-image-pb-ast.fits.fz"),
        ],
        "extra_images": [file_record(f"{root}-I-residual.fits.fz")],
    }


def _patch_image_preparation_tasks(monkeypatch):
    prepare_visibility_task = _CapturedTask(_image_sector_prepared_visibility_result)
    concatenate_task = _CapturedTask(
        lambda index: directory_record(f"/work/image_1/sector_{index + 1}.concat.ms")
    )
    wsclean_task = _CapturedTask(_image_sector_wsclean_result)
    finish_wsclean_task = _CapturedTask(_image_sector_wsclean_result)
    residual_task = _CapturedTask(_image_sector_no_residual_result)
    prepare_outputs_task = _CapturedTask(_image_sector_preparation_result)
    monkeypatch.setattr(
        image_module,
        "image_sector_prepare_visibility_task",
        prepare_visibility_task,
    )
    monkeypatch.setattr(image_module, "image_sector_concatenate_task", concatenate_task)
    monkeypatch.setattr(image_module, "image_sector_wsclean_task", wsclean_task)
    monkeypatch.setattr(image_module, "image_sector_finish_wsclean_task", finish_wsclean_task)
    monkeypatch.setattr(
        image_module,
        "image_sector_residual_visibilities_task",
        residual_task,
    )
    monkeypatch.setattr(image_module, "image_sector_prepare_outputs_task", prepare_outputs_task)
    return {
        "prepare_visibility": prepare_visibility_task,
        "concatenate": concatenate_task,
        "wsclean": wsclean_task,
        "finish_wsclean": finish_wsclean_task,
        "residual": residual_task,
        "prepare_outputs": prepare_outputs_task,
    }


@pytest.mark.parametrize("task_runner", ["sync", "local_dask", "external_dask"])
def test_image_flow_submits_plain_sector_task_payloads(monkeypatch, task_runner):
    config = ExecutionConfig(task_runner=task_runner)
    payload = representative_image_payload()
    preparation_tasks = _patch_image_preparation_tasks(monkeypatch)
    prepare_task = preparation_tasks["prepare_outputs"]
    filter_task = _CapturedTask(_image_sector_filter_result)
    diagnostics_task = _CapturedTask(_image_sector_diagnostics_result)
    cube_task = _CapturedTask(_image_sector_cube_result)
    catalog_task = _CapturedTask(_image_sector_catalog_result)
    restore_task = _CapturedTask(_image_sector_restored_model_result)
    compression_task = _CapturedTask(_image_sector_compression_result)
    finalize_task = _CapturedTask(_image_sector_result)
    monkeypatch.setattr(image_module, "image_sector_filter_skymodel_task", filter_task)
    monkeypatch.setattr(image_module, "image_sector_diagnostics_task", diagnostics_task)
    monkeypatch.setattr(image_module, "image_sector_make_image_cube_task", cube_task)
    monkeypatch.setattr(
        image_module, "image_sector_make_catalog_from_image_cube_task", catalog_task
    )
    monkeypatch.setattr(image_module, "image_sector_restore_skymodel_task", restore_task)
    monkeypatch.setattr(image_module, "image_sector_compress_images_task", compression_task)
    monkeypatch.setattr(image_module, "image_sector_finalize_task", finalize_task)

    result = image_module._run_image_prefect_tasks(payload, execution_config=config)

    assert list(result)[:3] == [
        "filtered_skymodel_true_sky",
        "filtered_skymodel_apparent_sky",
        "pybdsf_catalog",
    ]
    assert [
        submission["options"]["task_run_name"]
        for submission in preparation_tasks["prepare_visibility"].submissions
    ] == ["prepare_imaging_data_1"]
    assert preparation_tasks["prepare_visibility"].submissions[0]["options"]["tags"] == ["dp3"]
    assert [
        submission["options"]["task_run_name"]
        for submission in preparation_tasks["concatenate"].submissions
    ] == ["concatenate_visibilities"]
    assert preparation_tasks["concatenate"].submissions[0]["options"]["tags"] == ["casacore"]
    assert [
        submission["options"]["task_run_name"]
        for submission in preparation_tasks["wsclean"].submissions
    ] == ["wsclean_image"]
    assert preparation_tasks["wsclean"].submissions[0]["options"]["tags"] == ["wsclean"]
    assert [
        submission["options"]["task_run_name"]
        for submission in preparation_tasks["finish_wsclean"].submissions
    ] == ["finish_wsclean_images"]
    assert preparation_tasks["finish_wsclean"].submissions[0]["options"]["tags"] == ["python"]
    assert preparation_tasks["residual"].submissions == []
    assert [submission["options"]["task_run_name"] for submission in prepare_task.submissions] == [
        "prepare_outputs",
    ]
    assert prepare_task.submissions[0]["options"]["tags"] == ["python"]
    assert [submission["options"]["task_run_name"] for submission in filter_task.submissions] == [
        "filter_skymodel",
    ]
    assert filter_task.submissions[0]["options"]["tags"] == ["python"]
    assert [
        submission["options"]["task_run_name"] for submission in diagnostics_task.submissions
    ] == [
        "calculate_image_diagnostics",
    ]
    assert [submission["options"]["task_run_name"] for submission in finalize_task.submissions] == [
        "finalize",
    ]
    assert cube_task.submissions == []
    assert catalog_task.submissions == []
    assert restore_task.submissions == []
    assert compression_task.submissions == []
    _assert_worker_submission_is_serializable(
        preparation_tasks["prepare_visibility"].submissions[0], config
    )
    _assert_worker_submission_is_serializable(
        preparation_tasks["concatenate"].submissions[0], config
    )
    _assert_worker_submission_is_serializable(preparation_tasks["wsclean"].submissions[0], config)
    _assert_worker_submission_is_serializable(
        preparation_tasks["finish_wsclean"].submissions[0], config
    )
    _assert_worker_submission_is_serializable(prepare_task.submissions[0], config)
    assert preparation_tasks["concatenate"].submissions[0]["args"][1] == [
        preparation_tasks["prepare_visibility"].futures[0]
    ]
    assert (
        preparation_tasks["wsclean"].submissions[0]["args"][1]
        is preparation_tasks["concatenate"].futures[0]
    )
    assert (
        preparation_tasks["finish_wsclean"].submissions[0]["args"][1]
        is preparation_tasks["wsclean"].futures[0]
    )
    assert prepare_task.submissions[0]["args"][0] == [
        preparation_tasks["prepare_visibility"].futures[0]
    ]
    assert prepare_task.submissions[0]["args"][1] is preparation_tasks["concatenate"].futures[0]
    assert prepare_task.submissions[0]["args"][2] is preparation_tasks["finish_wsclean"].futures[0]
    assert prepare_task.submissions[0]["args"][4] is None
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
    assert finalize_task.submissions[0]["args"][5] is None
    assert finalize_task.submissions[0]["args"][6] is None
    assert finalize_task.submissions[0]["args"][7] is None
    assert finalize_task.submissions[0]["args"][8] is None
    assert finalize_task.submissions[0]["args"][9] is None


def test_image_flow_submits_image_cube_task_when_requested(monkeypatch):
    config = ExecutionConfig(task_runner="sync")
    payload = representative_image_payload()
    payload["sectors"][0]["make_image_cube"] = True
    preparation_tasks = _patch_image_preparation_tasks(monkeypatch)
    prepare_task = preparation_tasks["prepare_outputs"]
    filter_task = _CapturedTask(_image_sector_filter_result)
    diagnostics_task = _CapturedTask(_image_sector_diagnostics_result)
    cube_task = _CapturedTask(_image_sector_cube_result)
    catalog_task = _CapturedTask(_image_sector_catalog_result)
    restore_task = _CapturedTask(_image_sector_restored_model_result)
    compression_task = _CapturedTask(_image_sector_compression_result)
    finalize_task = _CapturedTask(
        lambda index: (
            _image_sector_result(index)
            | {
                "sector_image_cubes": _image_sector_cube_result(index)["image_cubes"],
                "sector_image_cube_beams": _image_sector_cube_result(index)["image_cube_beams"],
                "sector_image_cube_frequencies": _image_sector_cube_result(index)[
                    "image_cube_frequencies"
                ],
            }
        )
    )
    monkeypatch.setattr(image_module, "image_sector_filter_skymodel_task", filter_task)
    monkeypatch.setattr(image_module, "image_sector_diagnostics_task", diagnostics_task)
    monkeypatch.setattr(image_module, "image_sector_make_image_cube_task", cube_task)
    monkeypatch.setattr(
        image_module, "image_sector_make_catalog_from_image_cube_task", catalog_task
    )
    monkeypatch.setattr(image_module, "image_sector_restore_skymodel_task", restore_task)
    monkeypatch.setattr(image_module, "image_sector_compress_images_task", compression_task)
    monkeypatch.setattr(image_module, "image_sector_finalize_task", finalize_task)

    result = image_module._run_image_prefect_tasks(payload, execution_config=config)

    assert [submission["options"]["task_run_name"] for submission in cube_task.submissions] == [
        "make_image_cube",
    ]
    assert result["sector_image_cubes"] == [
        [file_record("/work/image_1/sector_1_I_freq_cube.fits")]
    ]
    _assert_worker_submission_is_serializable(cube_task.submissions[0], config)
    assert catalog_task.submissions == []
    assert restore_task.submissions == []
    assert compression_task.submissions == []
    assert cube_task.submissions[0]["args"][1] is prepare_task.futures[0]
    assert finalize_task.submissions[0]["args"][5] is cube_task.futures[0]
    assert finalize_task.submissions[0]["args"][5].result() == _image_sector_cube_result(0)
    assert finalize_task.submissions[0]["args"][6] is None
    assert finalize_task.submissions[0]["args"][7] is None
    assert finalize_task.submissions[0]["args"][8] is None
    assert finalize_task.submissions[0]["args"][9] is None


def test_image_flow_submits_normalization_task_when_requested(monkeypatch):
    config = ExecutionConfig(task_runner="sync")
    payload = representative_image_payload()
    payload["sectors"][0]["make_image_cube"] = True
    payload["sectors"][0]["normalize_flux_scale"] = True
    preparation_tasks = _patch_image_preparation_tasks(monkeypatch)
    prepare_task = preparation_tasks["prepare_outputs"]
    filter_task = _CapturedTask(_image_sector_filter_result)
    diagnostics_task = _CapturedTask(_image_sector_diagnostics_result)
    cube_task = _CapturedTask(_image_sector_cube_result)
    catalog_task = _CapturedTask(_image_sector_catalog_result)
    normalization_task = _CapturedTask(_image_sector_normalization_result)
    restore_task = _CapturedTask(_image_sector_restored_model_result)
    compression_task = _CapturedTask(_image_sector_compression_result)
    finalize_task = _CapturedTask(
        lambda index: (
            _image_sector_result(index)
            | {
                "sector_image_cubes": _image_sector_cube_result(index)["image_cubes"],
                "sector_image_cube_beams": _image_sector_cube_result(index)["image_cube_beams"],
                "sector_image_cube_frequencies": _image_sector_cube_result(index)[
                    "image_cube_frequencies"
                ],
                "sector_source_catalog": file_record(
                    f"/work/image_1/sector_{index + 1}_source_catalog.fits"
                ),
                "sector_normalize_h5parm": file_record(
                    f"/work/image_1/sector_{index + 1}_normalize.h5parm"
                ),
            }
        )
    )
    monkeypatch.setattr(image_module, "image_sector_filter_skymodel_task", filter_task)
    monkeypatch.setattr(image_module, "image_sector_diagnostics_task", diagnostics_task)
    monkeypatch.setattr(image_module, "image_sector_make_image_cube_task", cube_task)
    monkeypatch.setattr(
        image_module, "image_sector_make_catalog_from_image_cube_task", catalog_task
    )
    monkeypatch.setattr(
        image_module,
        "image_sector_normalize_flux_scale_task",
        normalization_task,
    )
    monkeypatch.setattr(image_module, "image_sector_restore_skymodel_task", restore_task)
    monkeypatch.setattr(image_module, "image_sector_compress_images_task", compression_task)
    monkeypatch.setattr(image_module, "image_sector_finalize_task", finalize_task)

    result = image_module._run_image_prefect_tasks(payload, execution_config=config)

    assert [
        submission["options"]["task_run_name"] for submission in normalization_task.submissions
    ] == [
        "normalize_flux_scale",
    ]
    assert [submission["options"]["task_run_name"] for submission in catalog_task.submissions] == [
        "make_catalog_from_image_cube",
    ]
    assert result["sector_normalize_h5parm"] == [
        file_record("/work/image_1/sector_1_normalize.h5parm")
    ]
    _assert_worker_submission_is_serializable(cube_task.submissions[0], config)
    _assert_worker_submission_is_serializable(catalog_task.submissions[0], config)
    _assert_worker_submission_is_serializable(normalization_task.submissions[0], config)
    assert cube_task.submissions[0]["args"][1] is prepare_task.futures[0]
    assert catalog_task.submissions[0]["args"][1] is cube_task.futures[0]
    assert normalization_task.submissions[0]["args"][1] is prepare_task.futures[0]
    assert normalization_task.submissions[0]["args"][2] is catalog_task.futures[0]
    assert restore_task.submissions == []
    assert compression_task.submissions == []
    assert finalize_task.submissions[0]["args"][5] is cube_task.futures[0]
    assert finalize_task.submissions[0]["args"][6] is catalog_task.futures[0]
    assert finalize_task.submissions[0]["args"][6].result() == _image_sector_catalog_result(0)
    assert finalize_task.submissions[0]["args"][7] is normalization_task.futures[0]
    assert finalize_task.submissions[0]["args"][7].result() == _image_sector_normalization_result(0)
    assert finalize_task.submissions[0]["args"][8] is None
    assert finalize_task.submissions[0]["args"][9] is None


def test_image_flow_submits_restoration_and_compression_tasks_when_requested(monkeypatch):
    config = ExecutionConfig(task_runner="sync")
    payload = representative_image_payload()
    payload["sectors"][0]["save_filtered_model_image"] = True
    payload["sectors"][0]["compress_images"] = True
    preparation_tasks = _patch_image_preparation_tasks(monkeypatch)
    prepare_task = preparation_tasks["prepare_outputs"]
    filter_task = _CapturedTask(_image_sector_filter_result)
    diagnostics_task = _CapturedTask(_image_sector_diagnostics_result)
    cube_task = _CapturedTask(_image_sector_cube_result)
    catalog_task = _CapturedTask(_image_sector_catalog_result)
    restore_task = _CapturedTask(_image_sector_restored_model_result)
    compression_task = _CapturedTask(_image_sector_compression_result)
    finalize_task = _CapturedTask(
        lambda index: (
            _image_sector_result(index)
            | {
                "sector_skymodel_image_fits": _image_sector_restored_model_result(index)[
                    "skymodel_image"
                ],
                "sector_I_images": _image_sector_compression_result(index)["sector_images"],
                "sector_extra_images": _image_sector_compression_result(index)["extra_images"],
            }
        )
    )
    monkeypatch.setattr(image_module, "image_sector_filter_skymodel_task", filter_task)
    monkeypatch.setattr(image_module, "image_sector_diagnostics_task", diagnostics_task)
    monkeypatch.setattr(image_module, "image_sector_make_image_cube_task", cube_task)
    monkeypatch.setattr(
        image_module, "image_sector_make_catalog_from_image_cube_task", catalog_task
    )
    monkeypatch.setattr(image_module, "image_sector_restore_skymodel_task", restore_task)
    monkeypatch.setattr(image_module, "image_sector_compress_images_task", compression_task)
    monkeypatch.setattr(image_module, "image_sector_finalize_task", finalize_task)

    result = image_module._run_image_prefect_tasks(payload, execution_config=config)

    assert [submission["options"]["task_run_name"] for submission in restore_task.submissions] == [
        "restore_skymodel",
    ]
    assert [
        submission["options"]["task_run_name"] for submission in compression_task.submissions
    ] == [
        "compress_images",
    ]
    assert result["sector_skymodel_image_fits"] == [
        file_record("/work/image_1/sector_1_filtered-model.fits.fz")
    ]
    assert result["sector_I_images"] == [
        [
            file_record("/work/image_1/sector_1-I-image.fits.fz"),
            file_record("/work/image_1/sector_1-I-image-pb.fits.fz"),
            file_record("/work/image_1/sector_1-I-image-pb-ast.fits.fz"),
        ]
    ]
    assert result["sector_extra_images"] == [
        [file_record("/work/image_1/sector_1-I-residual.fits.fz")]
    ]
    assert cube_task.submissions == []
    assert catalog_task.submissions == []
    _assert_worker_submission_is_serializable(restore_task.submissions[0], config)
    _assert_worker_submission_is_serializable(compression_task.submissions[0], config)
    assert restore_task.submissions[0]["args"][1] is prepare_task.futures[0]
    assert restore_task.submissions[0]["args"][2] is filter_task.futures[0]
    assert compression_task.submissions[0]["args"][1] is prepare_task.futures[0]
    assert compression_task.submissions[0]["args"][2] is diagnostics_task.futures[0]
    assert finalize_task.submissions[0]["args"][8] is restore_task.futures[0]
    assert finalize_task.submissions[0]["args"][8].result() == _image_sector_restored_model_result(
        0
    )
    assert finalize_task.submissions[0]["args"][9] is compression_task.futures[0]
    assert finalize_task.submissions[0]["args"][9].result() == _image_sector_compression_result(0)


def test_image_flow_disambiguates_sector_task_names_for_multiple_sectors(monkeypatch):
    config = ExecutionConfig(task_runner="sync")
    payload = representative_image_payload()
    second_sector = deepcopy(payload["sectors"][0])
    second_sector["image_name"] = "sector_2"
    payload["sectors"].append(second_sector)
    preparation_tasks = _patch_image_preparation_tasks(monkeypatch)
    prepare_task = preparation_tasks["prepare_outputs"]
    filter_task = _CapturedTask(_image_sector_filter_result)
    diagnostics_task = _CapturedTask(_image_sector_diagnostics_result)
    finalize_task = _CapturedTask(_image_sector_result)
    monkeypatch.setattr(image_module, "image_sector_filter_skymodel_task", filter_task)
    monkeypatch.setattr(image_module, "image_sector_diagnostics_task", diagnostics_task)
    monkeypatch.setattr(image_module, "image_sector_finalize_task", finalize_task)

    image_module._run_image_prefect_tasks(payload, execution_config=config)

    assert [
        submission["options"]["task_run_name"]
        for submission in preparation_tasks["prepare_visibility"].submissions
    ] == [
        "sector_1_prepare_imaging_data_1",
        "sector_2_prepare_imaging_data_1",
    ]
    assert [
        submission["options"]["task_run_name"]
        for submission in preparation_tasks["concatenate"].submissions
    ] == [
        "sector_1_concatenate_visibilities",
        "sector_2_concatenate_visibilities",
    ]
    assert [
        submission["options"]["task_run_name"]
        for submission in preparation_tasks["wsclean"].submissions
    ] == [
        "sector_1_wsclean_image",
        "sector_2_wsclean_image",
    ]
    assert [
        submission["options"]["task_run_name"]
        for submission in preparation_tasks["finish_wsclean"].submissions
    ] == [
        "sector_1_finish_wsclean_images",
        "sector_2_finish_wsclean_images",
    ]
    assert [submission["options"]["task_run_name"] for submission in prepare_task.submissions] == [
        "sector_1_prepare_outputs",
        "sector_2_prepare_outputs",
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
    payload["chunks"][0]["solve_slots"].append(
        payload["chunks"][0]["solve_slots"][0]
        | {
            "slot": 2,
            "solve_type": "medium_phase",
            "solution_label": "medium1",
            "h5parm": "solve2.h5",
            "h5parm_path": "/work/calibrate_1/solve2.h5",
        }
    )
    calibrate_task = _CapturedTask(
        lambda index: {
            "solve1": file_record(f"/work/calibrate_1/chunk{index + 1}-solve1.h5"),
            "solve2": file_record(f"/work/calibrate_1/chunk{index + 1}-solve2.h5"),
        }
    )
    collect_task = _CapturedTask(
        lambda index: {
            "solve_key": f"solve{index + 1}",
            "solve_slot": payload["chunks"][0]["solve_slots"][index],
            "collected_record": file_record(f"/work/calibrate_1/collected_solve{index + 1}.h5"),
        }
    )
    process_task = _CapturedTask(
        lambda index: {
            "solve_key": f"solve{index + 1}",
            "solve_slot": payload["chunks"][0]["solve_slots"][index],
            "solution_record": file_record(f"/work/calibrate_1/collected_solve{index + 1}.h5"),
            "combine_record": file_record(f"/work/calibrate_1/processed_solve{index + 1}.h5"),
        }
    )
    plot_task = _CapturedTask(
        lambda index: {
            "solve_key": f"solve{index + 1}",
            "plots": {
                ("fast_phase_plots" if index == 0 else "medium1_phase_plots"): [
                    file_record(f"/work/calibrate_1/solve{index + 1}-phase.png")
                ]
            },
        }
    )
    combine_task = _CapturedTask(lambda index: file_record("/work/calibrate_1/combined_phase.h5"))
    finalize_task = _CapturedTask(
        lambda index: {"combined_solutions": file_record("/work/calibrate_1/combined.h5")}
    )

    monkeypatch.setattr(calibrate_module, "calibrate_chunk_task", calibrate_task)
    monkeypatch.setattr(calibrate_module, "collect_h5parms_task", collect_task)
    monkeypatch.setattr(calibrate_module, "process_solutions_task", process_task)
    monkeypatch.setattr(calibrate_module, "plot_solutions_task", plot_task)
    monkeypatch.setattr(calibrate_module, "combine_h5parms_task", combine_task)
    monkeypatch.setattr(calibrate_module, "finalize_solutions_task", finalize_task)

    result = calibrate_module._run_calibrate_prefect_tasks(payload, execution_config=config)

    assert result == {"combined_solutions": file_record("/work/calibrate_1/combined.h5")}
    assert [
        submission["options"]["task_run_name"] for submission in calibrate_task.submissions
    ] == [
        "solve_chunk_1",
    ]
    assert calibrate_task.submissions[0]["options"]["tags"] == ["dp3"]
    assert [submission["options"]["task_run_name"] for submission in collect_task.submissions] == [
        "collect_h5parms_1",
        "collect_h5parms_2",
    ]
    assert [submission["options"]["task_run_name"] for submission in process_task.submissions] == [
        "process_fast_phase",
        "process_medium1_phase",
    ]
    assert [submission["options"]["task_run_name"] for submission in plot_task.submissions] == [
        "plot_fast_phase",
        "plot_medium1_phase",
    ]
    assert [submission["options"]["task_run_name"] for submission in combine_task.submissions] == [
        "combine_h5parms",
    ]
    assert [submission["options"]["task_run_name"] for submission in finalize_task.submissions] == [
        "finalize_solutions",
    ]
    _assert_worker_submission_is_serializable(calibrate_task.submissions[0], config)
    _assert_worker_submission_is_serializable(collect_task.submissions[0], config)
    _assert_worker_submission_is_serializable(process_task.submissions[0], config)
    _assert_worker_submission_is_serializable(plot_task.submissions[0], config)
    _assert_worker_submission_is_serializable(combine_task.submissions[0], config)
    _assert_worker_submission_is_serializable(finalize_task.submissions[0], config)
    assert collect_task.submissions[0]["args"][0] is payload
    assert collect_task.submissions[0]["args"][1] == [
        {
            "solve1": file_record("/work/calibrate_1/chunk1-solve1.h5"),
            "solve2": file_record("/work/calibrate_1/chunk1-solve2.h5"),
        }
    ]
    assert collect_task.submissions[0]["args"][2] is payload["chunks"][0]["solve_slots"][0]
    assert collect_task.submissions[1]["args"][2] is payload["chunks"][0]["solve_slots"][1]
    assert process_task.submissions[0]["args"][1] is collect_task.futures[0]
    assert process_task.submissions[1]["args"][1] is collect_task.futures[1]
    assert plot_task.submissions[0]["args"][1] is process_task.futures[0]
    assert plot_task.submissions[1]["args"][1] is process_task.futures[1]
    assert combine_task.submissions[0]["args"][1] == process_task.futures
    assert finalize_task.submissions[0]["args"][0] is payload
    assert finalize_task.submissions[0]["args"][1] == process_task.futures
    assert finalize_task.submissions[0]["args"][2] == plot_task.futures
    assert finalize_task.submissions[0]["args"][3] is combine_task.futures[0]


def _calibrate_prediction_payload(payload: dict) -> dict:
    payload["image_predict"] = {
        "skymodel": "/data/calibration.skymodel",
        "model_image_root": "calibration_model",
        "model_image_ra_dec": ["12:00:00.0", "+45.00.00.0"],
        "model_image_imsize": [1024, 1024],
        "model_image_cellsize": 0.001,
        "model_image_frequency_bandwidth": [150000000.0, 1000000.0],
        "num_spectral_terms": 1,
        "model_images": ["/work/calibrate_1/calibration_model-term-0.fits"],
        "ra_mid": 123.0,
        "dec_mid": 45.0,
        "facet_region_width_ra": 2.0,
        "facet_region_width_dec": 2.5,
        "facet_region_file": "field_facets_ds9.reg",
        "facet_region_path": "/work/calibrate_1/field_facets_ds9.reg",
    }
    return payload


def _minimal_calibrate_task_graph(monkeypatch, payload):
    calibrate_task = _CapturedTask(
        lambda index: {"solve1": file_record(f"/work/calibrate_1/chunk{index + 1}-solve1.h5")}
    )
    collect_task = _CapturedTask(
        lambda index: {
            "solve_key": "solve1",
            "solve_slot": payload["chunks"][0]["solve_slots"][0],
            "collected_record": file_record("/work/calibrate_1/collected_solve1.h5"),
        }
    )
    process_task = _CapturedTask(
        lambda index: {
            "solve_key": "solve1",
            "solve_slot": payload["chunks"][0]["solve_slots"][0],
            "solution_record": file_record("/work/calibrate_1/collected_solve1.h5"),
            "combine_record": file_record("/work/calibrate_1/processed_solve1.h5"),
        }
    )
    plot_task = _CapturedTask(
        lambda index: {
            "solve_key": "solve1",
            "plots": {"fast_phase_plots": [file_record("/work/calibrate_1/solve1-phase.png")]},
        }
    )
    combine_task = _CapturedTask(lambda index: file_record("/work/calibrate_1/combined.h5"))
    finalize_task = _CapturedTask(
        lambda index: {"fast_phase_solutions": file_record("/work/calibrate_1/processed_solve1.h5")}
    )

    monkeypatch.setattr(calibrate_module, "calibrate_chunk_task", calibrate_task)
    monkeypatch.setattr(calibrate_module, "collect_h5parms_task", collect_task)
    monkeypatch.setattr(calibrate_module, "process_solutions_task", process_task)
    monkeypatch.setattr(calibrate_module, "plot_solutions_task", plot_task)
    monkeypatch.setattr(calibrate_module, "combine_h5parms_task", combine_task)
    monkeypatch.setattr(calibrate_module, "finalize_solutions_task", finalize_task)
    return calibrate_task, collect_task, process_task, plot_task, combine_task, finalize_task


def test_calibrate_flow_submits_image_predict_preparation_tasks(monkeypatch):
    config = ExecutionConfig(task_runner="sync")
    payload = _calibrate_prediction_payload(representative_calibrate_payload())
    payload["image_based_predict"] = True
    region_task = _CapturedTask(lambda index: file_record("/work/calibrate_1/field_facets_ds9.reg"))
    draw_task = _CapturedTask(
        lambda index: [file_record("/work/calibrate_1/calibration_model-term-0.fits")]
    )
    facet_task = _CapturedTask(lambda index: {"patch_names": ["patch1"], "modeldatacolumn": ""})
    wsclean_task = _CapturedTask(lambda index: payload["chunks"][index])
    normalization_task = _CapturedTask(lambda index: "/work/calibrate_1/normalize.h5")
    calibrate_task, _, _, _, _, finalize_task = _minimal_calibrate_task_graph(monkeypatch, payload)
    monkeypatch.setattr(calibrate_module, "make_predict_region_task", region_task)
    monkeypatch.setattr(calibrate_module, "draw_predict_model_task", draw_task)
    monkeypatch.setattr(calibrate_module, "wsclean_predict_facet_info_task", facet_task)
    monkeypatch.setattr(calibrate_module, "wsclean_predict_chunk_task", wsclean_task)
    monkeypatch.setattr(
        calibrate_module,
        "adjust_prediction_normalization_h5parm_task",
        normalization_task,
    )

    result = calibrate_module._run_calibrate_prefect_tasks(payload, execution_config=config)

    assert result == {"fast_phase_solutions": file_record("/work/calibrate_1/processed_solve1.h5")}
    assert [submission["options"]["task_run_name"] for submission in region_task.submissions] == [
        "make_predict_region",
    ]
    assert region_task.submissions[0]["options"]["tags"] == ["python"]
    assert [submission["options"]["task_run_name"] for submission in draw_task.submissions] == [
        "draw_model",
    ]
    assert draw_task.submissions[0]["options"]["tags"] == ["wsclean"]
    assert facet_task.submissions == []
    assert wsclean_task.submissions == []
    assert normalization_task.submissions == []
    prepared_payload = calibrate_task.submissions[0]["args"][0]
    assert prepared_payload["predict_regions"] == "/work/calibrate_1/field_facets_ds9.reg"
    assert prepared_payload["predict_images"] == ["/work/calibrate_1/calibration_model-term-0.fits"]
    _assert_worker_submission_is_serializable(region_task.submissions[0], config)
    _assert_worker_submission_is_serializable(draw_task.submissions[0], config)
    _assert_worker_submission_is_serializable(calibrate_task.submissions[0], config)
    assert finalize_task.submissions[0]["args"][0] is prepared_payload


def test_calibrate_flow_submits_wsclean_predict_chunk_tasks(monkeypatch):
    config = ExecutionConfig(task_runner="sync")
    payload = _calibrate_prediction_payload(representative_calibrate_payload())
    payload["wsclean_predict"] = True
    second_chunk = deepcopy(payload["chunks"][0])
    second_chunk["msin"] = "/data/obs_2.ms"
    second_chunk["starttime"] = "59100.0"
    second_chunk["output_h5parm"] = "solve1_chunk2.h5"
    second_chunk["output_h5parm_path"] = "/work/calibrate_1/solve1_chunk2.h5"
    payload["chunks"].append(second_chunk)
    region_task = _CapturedTask(
        lambda index: file_record("/work/calibrate_1/predict_field_facets_ds9.reg")
    )
    draw_task = _CapturedTask(lambda index: [])
    facet_task = _CapturedTask(
        lambda index: {"patch_names": ["patch1", "patch2"], "modeldatacolumn": "[patch1,patch2]"}
    )
    wsclean_task = _CapturedTask(
        lambda index: payload["chunks"][index] | {"msin": f"/work/calibrate_1/predict_{index}.ms"}
    )
    normalization_task = _CapturedTask(lambda index: "/work/calibrate_1/normalize.h5")
    calibrate_task, _, _, _, _, _ = _minimal_calibrate_task_graph(monkeypatch, payload)
    monkeypatch.setattr(calibrate_module, "make_predict_region_task", region_task)
    monkeypatch.setattr(calibrate_module, "draw_predict_model_task", draw_task)
    monkeypatch.setattr(calibrate_module, "wsclean_predict_facet_info_task", facet_task)
    monkeypatch.setattr(calibrate_module, "wsclean_predict_chunk_task", wsclean_task)
    monkeypatch.setattr(
        calibrate_module,
        "adjust_prediction_normalization_h5parm_task",
        normalization_task,
    )

    calibrate_module._run_calibrate_prefect_tasks(payload, execution_config=config)

    assert [submission["options"]["task_run_name"] for submission in region_task.submissions] == [
        "make_predict_region",
    ]
    assert [submission["options"]["task_run_name"] for submission in facet_task.submissions] == [
        "read_predict_facets",
    ]
    assert [submission["options"]["task_run_name"] for submission in wsclean_task.submissions] == [
        "wsclean_predict_1",
        "wsclean_predict_2",
    ]
    assert draw_task.submissions == []
    assert normalization_task.submissions == []
    prepared_payload = calibrate_task.submissions[0]["args"][0]
    assert prepared_payload["predict_regions"] == "/work/calibrate_1/predict_field_facets_ds9.reg"
    assert prepared_payload["modeldatacolumn"] == "[patch1,patch2]"
    assert [chunk["msin"] for chunk in prepared_payload["chunks"]] == [
        "/work/calibrate_1/predict_0.ms",
        "/work/calibrate_1/predict_1.ms",
    ]
    _assert_worker_submission_is_serializable(region_task.submissions[0], config)
    _assert_worker_submission_is_serializable(facet_task.submissions[0], config)
    _assert_worker_submission_is_serializable(wsclean_task.submissions[0], config)
    assert facet_task.submissions[0]["args"][0] is region_task.futures[0]
    assert facet_task.submissions[0]["args"][1] == payload["pipeline_working_dir"]
    assert wsclean_task.submissions[0]["args"][3] is facet_task.futures[0]


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
        "predict_model_data_1",
    ]
    assert model_task.submissions[0]["options"]["tags"] == ["dp3"]
    assert [
        submission["options"]["task_run_name"] for submission in postprocess_task.submissions
    ] == [
        "postprocess_1",
    ]
    assert postprocess_task.submissions[0]["options"]["tags"] == ["casacore", "python"]
    _assert_worker_submission_is_serializable(model_task.submissions[0], config)
    _assert_worker_submission_is_serializable(postprocess_task.submissions[0], config)


def test_mosaic_flow_submits_plain_payloads_with_mosaic_product_names(monkeypatch):
    config = ExecutionConfig(task_runner="sync")
    payload = representative_mosaic_payload()
    template_task = _CapturedTask(
        lambda index: file_record("/work/mosaic_1/mosaic_1_template.fits")
    )
    mosaic_task = _CapturedTask(
        lambda index: file_record(f"/work/mosaic_1/mosaic_{index + 1}-I-image.fits")
    )
    monkeypatch.setattr(mosaic_module, "mosaic_template_task", template_task)
    monkeypatch.setattr(mosaic_module, "mosaic_task", mosaic_task)

    result = mosaic_module._run_mosaic_prefect_tasks(payload, execution_config=config)

    assert result == {"mosaic_image": [file_record("/work/mosaic_1/mosaic_1-I-image.fits")]}
    assert [submission["options"]["task_run_name"] for submission in template_task.submissions] == [
        "make_mosaic_template",
    ]
    assert template_task.submissions[0]["options"]["tags"] == ["python"]
    assert [submission["options"]["task_run_name"] for submission in mosaic_task.submissions] == [
        "mosaic_I_image",
    ]
    assert mosaic_task.submissions[0]["options"]["tags"] == ["python"]
    _assert_worker_submission_is_serializable(template_task.submissions[0], config)
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
        "concatenate_epoch_1",
    ]
    assert concatenate_task.submissions[0]["options"]["tags"] == ["casacore"]
    _assert_worker_submission_is_serializable(concatenate_task.submissions[0], config)

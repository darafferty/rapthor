"""Image-based prediction preparation for calibration execution."""

from typing import Mapping

from rapthor.execution.calibrate.collection import adjust_h5parm_sources
from rapthor.execution.calibrate.commands import (
    build_draw_model_command,
    build_make_region_file_command,
)
from rapthor.execution.calibrate.payloads import CalibrateImagePredictPayload, CalibratePayload
from rapthor.execution.config import ExecutionConfig
from rapthor.execution.outputs import require_file
from rapthor.execution.shell import run_external_command
from rapthor.lib.records import file_record


def _run_draw_model(
    payload: CalibratePayload,
    image_predict: CalibrateImagePredictPayload,
    execution_config: ExecutionConfig,
    shell_operation_cls=None,
) -> list[dict]:
    pipeline_working_dir = str(payload["pipeline_working_dir"])
    command = build_draw_model_command(
        skymodel=str(image_predict["skymodel"]),
        numterms=int(image_predict["num_spectral_terms"]),
        name=str(image_predict["model_image_root"]),
        ra_dec=list(image_predict["model_image_ra_dec"]),
        frequency_bandwidth=list(image_predict["model_image_frequency_bandwidth"]),
        cellsize_deg=image_predict["model_image_cellsize"],
        imsize=list(image_predict["model_image_imsize"]),
        numthreads=int(payload["max_threads"]),
    )
    run_external_command(
        command,
        pipeline_working_dir,
        execution_config,
        shell_operation_cls=shell_operation_cls,
    )
    return [require_file(path, "Calibration model image") for path in image_predict["model_images"]]


def _run_make_region_file(
    image_predict: CalibrateImagePredictPayload,
    payload: CalibratePayload,
    execution_config: ExecutionConfig,
    shell_operation_cls=None,
) -> dict:
    pipeline_working_dir = str(payload["pipeline_working_dir"])
    command = build_make_region_file_command(
        skymodel=str(image_predict["skymodel"]),
        ra_mid=image_predict["ra_mid"],
        dec_mid=image_predict["dec_mid"],
        width_ra=image_predict["facet_region_width_ra"],
        width_dec=image_predict["facet_region_width_dec"],
        outfile=str(image_predict["facet_region_file"]),
        enclose_names=False,
    )
    run_external_command(
        command,
        pipeline_working_dir,
        execution_config,
        shell_operation_cls=shell_operation_cls,
    )
    return require_file(str(image_predict["facet_region_path"]), "Calibration region file")


def prepare_image_based_predict(
    payload: CalibratePayload,
    execution_config: ExecutionConfig,
    shell_operation_cls=None,
) -> CalibratePayload:
    """Prepare model-image inputs when calibration uses image-based prediction."""
    if not payload.get("image_based_predict"):
        return payload

    image_predict = payload.get("image_predict")
    if not isinstance(image_predict, Mapping):
        raise ValueError("Image-based prediction payload is missing")

    model_images = _run_draw_model(
        payload,
        image_predict,
        execution_config,
        shell_operation_cls=shell_operation_cls,
    )
    region_file = _run_make_region_file(
        image_predict,
        payload,
        execution_config,
        shell_operation_cls=shell_operation_cls,
    )
    prepared_payload = dict(payload)
    prepared_payload["predict_images"] = [record["path"] for record in model_images]
    prepared_payload["predict_regions"] = region_file["path"]
    if payload.get("normalize_h5parm"):
        normalize_record = adjust_h5parm_sources(
            file_record(str(payload["normalize_h5parm"])),
            payload,
            str(payload["pipeline_working_dir"]),
            execution_config,
            "Adjusted normalization h5parm",
            shell_operation_cls=shell_operation_cls,
        )
        prepared_payload["normalize_h5parm"] = normalize_record["path"]
    return prepared_payload

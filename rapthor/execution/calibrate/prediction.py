"""Image-based prediction preparation for calibration execution."""

import math
import os
import shutil
import stat
from pathlib import Path
from typing import Mapping

from rapthor.execution.calibrate.collection import adjust_h5parm_sources
from rapthor.execution.calibrate.commands import (
    DrawModelOptions,
    WscleanPredictOptions,
    build_draw_model_command,
    build_wsclean_predict_command,
)
from rapthor.execution.calibrate.payloads import CalibrateImagePredictPayload, CalibratePayload
from rapthor.execution.config import ExecutionConfig
from rapthor.execution.outputs import require_file
from rapthor.execution.regions import make_ds9_region_from_skymodel
from rapthor.execution.shell import run_external_command
from rapthor.lib.records import file_record

WSCLEAN_PREDICT_MAX_BANDWIDTH_HZ = 2.0e6
WSCLEAN_MODEL_STORAGE_MANAGER = "default"


def prepare_image_based_predict(
    payload: CalibratePayload,
    execution_config: ExecutionConfig,
    shell_operation_cls=None,
) -> CalibratePayload:
    """Prepare model-image inputs when calibration uses image-based prediction."""
    if not (payload.get("image_based_predict") or payload.get("wsclean_predict")):
        return payload

    region_file = make_predict_region_file(payload)
    prepared_payload = dict(payload)
    prepared_payload["predict_regions"] = region_file["path"]

    if payload.get("image_based_predict"):
        model_images = draw_predict_model_images(
            payload,
            execution_config,
            shell_operation_cls=shell_operation_cls,
        )
        prepared_payload["predict_images"] = [record["path"] for record in model_images]

    if payload.get("wsclean_predict"):
        facet_info = wsclean_predict_facet_info(region_file)
        prepared_chunks = []
        for chunk_index, chunk in enumerate(payload["chunks"]):
            prepared_chunks.append(
                prepare_wsclean_predict_chunk(
                    payload,
                    chunk,
                    chunk_index,
                    facet_info,
                    execution_config,
                    shell_operation_cls=shell_operation_cls,
                )
            )
        prepared_payload["chunks"] = prepared_chunks
        prepared_payload["modeldatacolumn"] = facet_info["modeldatacolumn"]

    if payload.get("normalize_h5parm"):
        prepared_payload["normalize_h5parm"] = adjust_prediction_normalization_h5parm(payload)
    return prepared_payload


def make_predict_region_file(payload: CalibratePayload) -> dict:
    """Create the facet region file used by image-based calibration prediction."""
    return _run_make_region_file(_image_predict_payload(payload))


def draw_predict_model_images(
    payload: CalibratePayload,
    execution_config: ExecutionConfig,
    shell_operation_cls=None,
) -> list[dict]:
    """Draw model images for DP3 image-based calibration prediction."""
    return _run_draw_model(
        payload,
        _image_predict_payload(payload),
        execution_config,
        shell_operation_cls=shell_operation_cls,
    )


def wsclean_predict_facet_info(region_record: Mapping[str, object]) -> dict:
    """Return facet names and DP3 model-column token for WSClean prediction."""
    patch_names = _patch_names_from_region(str(region_record["path"]))
    return {
        "patch_names": patch_names,
        "modeldatacolumn": _patch_list_token(patch_names),
    }


def prepare_wsclean_predict_chunk(
    payload: CalibratePayload,
    chunk: Mapping[str, object],
    chunk_index: int,
    facet_info: Mapping[str, object],
    execution_config: ExecutionConfig,
    shell_operation_cls=None,
) -> dict:
    """Prepare one copied Measurement Set with WSClean model columns."""
    copied_msin = _run_wsclean_predict_for_chunk(
        payload,
        _image_predict_payload(payload),
        chunk,
        chunk_index,
        [str(name) for name in facet_info["patch_names"]],
        execution_config,
        shell_operation_cls=shell_operation_cls,
    )
    prepared_chunk = dict(chunk)
    prepared_chunk["msin"] = copied_msin
    return prepared_chunk


def adjust_prediction_normalization_h5parm(payload: CalibratePayload) -> str:
    """Adjust a normalization h5parm to the prediction source coordinates."""
    normalize_h5parm = payload.get("normalize_h5parm")
    if not normalize_h5parm:
        raise ValueError("normalize_h5parm is required for normalization adjustment")
    normalize_record = adjust_h5parm_sources(
        file_record(str(normalize_h5parm)),
        payload,
        "Adjusted normalization h5parm",
    )
    return str(normalize_record["path"])


def _run_make_region_file(
    image_predict: CalibrateImagePredictPayload,
) -> dict:
    make_ds9_region_from_skymodel(
        str(image_predict["skymodel"]),
        float(image_predict["ra_mid"]),
        float(image_predict["dec_mid"]),
        float(image_predict["facet_region_width_ra"]),
        float(image_predict["facet_region_width_dec"]),
        str(image_predict["facet_region_path"]),
        enclose_names=False,
    )
    return require_file(str(image_predict["facet_region_path"]), "Calibration region file")


def _run_draw_model(
    payload: CalibratePayload,
    image_predict: CalibrateImagePredictPayload,
    execution_config: ExecutionConfig,
    shell_operation_cls=None,
) -> list[dict]:
    pipeline_working_dir = str(payload["pipeline_working_dir"])
    command = build_draw_model_command(_draw_model_options(payload, image_predict))
    run_external_command(
        command,
        pipeline_working_dir,
        execution_config,
        shell_operation_cls=shell_operation_cls,
    )
    return [require_file(path, "Calibration model image") for path in image_predict["model_images"]]


def _run_wsclean_predict_for_chunk(
    payload: CalibratePayload,
    image_predict: CalibrateImagePredictPayload,
    chunk: Mapping[str, object],
    chunk_index: int,
    patch_names: list[str],
    execution_config: ExecutionConfig,
    shell_operation_cls=None,
) -> str:
    """Draw narrow-band WSClean models and predict facets into a copied MS."""
    pipeline_dir = str(payload["pipeline_working_dir"])
    copied_msin = _copy_measurement_set_for_wsclean_predict(
        str(chunk["msin"]),
        pipeline_dir,
        chunk_index,
    )
    frequency_chunks = _frequency_chunks_for_ms(
        copied_msin,
        list(image_predict["model_image_frequency_bandwidth"]),
    )
    time_frequency_smearing = bool(
        payload.get("correctfreqsmearing") or payload.get("correcttimesmearing")
    )

    for frequency_index, frequency_chunk in enumerate(frequency_chunks):
        model_root = os.path.join(
            pipeline_dir,
            f"wsclean_predict_chunk_{chunk_index + 1}_band_{frequency_index + 1}",
        )
        draw_command = build_draw_model_command(
            _wsclean_draw_model_options(
                payload,
                image_predict,
                model_root=model_root,
                frequency_bandwidth=list(frequency_chunk["frequency_bandwidth"]),
            )
        )
        run_external_command(
            draw_command,
            pipeline_dir,
            execution_config,
            shell_operation_cls=shell_operation_cls,
        )
        _ensure_wsclean_model_fits(model_root)

        for patch_name in patch_names:
            predict_command = build_wsclean_predict_command(
                WscleanPredictOptions(
                    msin=copied_msin,
                    region_file=str(image_predict["facet_region_path"]),
                    model_column=str(patch_name),
                    facet=str(patch_name),
                    model_root=model_root,
                    channel_range=frequency_chunk["channel_range"],
                    model_storage_manager=WSCLEAN_MODEL_STORAGE_MANAGER,
                    num_threads=int(payload["max_threads"]),
                    apply_time_frequency_smearing=time_frequency_smearing,
                )
            )
            run_external_command(
                predict_command,
                pipeline_dir,
                execution_config,
                shell_operation_cls=shell_operation_cls,
            )

    return copied_msin


def _image_predict_payload(payload: CalibratePayload) -> CalibrateImagePredictPayload:
    """Return the image-prediction payload or raise a clear contract error."""
    image_predict = payload.get("image_predict")
    if not isinstance(image_predict, Mapping):
        raise ValueError("Image-based prediction payload is missing")
    return image_predict


def _draw_model_options(
    payload: CalibratePayload,
    image_predict: CalibrateImagePredictPayload,
) -> DrawModelOptions:
    return DrawModelOptions(
        skymodel=str(image_predict["skymodel"]),
        num_terms=int(image_predict["num_spectral_terms"]),
        name=str(image_predict["model_image_root"]),
        ra_dec=list(image_predict["model_image_ra_dec"]),
        frequency_bandwidth=list(image_predict["model_image_frequency_bandwidth"]),
        cellsize_deg=image_predict["model_image_cellsize"],
        imsize=list(image_predict["model_image_imsize"]),
        num_threads=int(payload["max_threads"]),
    )


def _wsclean_draw_model_options(
    payload: CalibratePayload,
    image_predict: CalibrateImagePredictPayload,
    *,
    model_root: str,
    frequency_bandwidth: list[object],
) -> DrawModelOptions:
    return DrawModelOptions(
        skymodel=str(image_predict["skymodel"]),
        num_terms=1,
        name=model_root,
        ra_dec=list(image_predict["model_image_ra_dec"]),
        frequency_bandwidth=frequency_bandwidth,
        cellsize_deg=image_predict["model_image_cellsize"],
        imsize=list(image_predict["model_image_imsize"]),
        num_threads=int(payload["max_threads"]),
    )


def _patch_names_from_region(region_file: str) -> list[str]:
    """Return facet names from a DS9 region file in WSClean column order."""
    from lsmtool.facet import read_ds9_region_file

    facet_regions = read_ds9_region_file(region_file)
    patch_names = [facet.name for facet in facet_regions if facet.name]
    if not patch_names:
        raise ValueError(f"No named facets were found in {region_file}")
    return patch_names


def _patch_list_token(patch_names: list[str]) -> str:
    """Return patch names in the DP3 model-column list syntax."""
    return "[" + ",".join(str(name) for name in patch_names) + "]"


def _copy_measurement_set_for_wsclean_predict(msin: str, pipeline_dir: str, index: int) -> str:
    """Copy a source Measurement Set before WSClean adds model columns."""
    source = Path(msin)
    if not source.is_dir():
        raise FileNotFoundError(f"WSClean predict input MS was not found: {source}")

    destination = Path(pipeline_dir) / f"{source.stem}_wsclean_predict_{index + 1}.ms"
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
    _make_tree_writable(destination)
    return str(destination)


def _measurement_set_channel_frequencies(msin: str) -> list[float]:
    """Read channel centre frequencies from a Measurement Set."""
    import casacore.tables as pt

    with pt.table(f"{msin}::SPECTRAL_WINDOW", ack=False) as table:
        frequencies = table.getcol("CHAN_FREQ")
    if len(frequencies) == 0:
        raise ValueError(f"SPECTRAL_WINDOW has no CHAN_FREQ rows: {msin}")
    return [float(value) for value in frequencies[0]]


def _frequency_chunks_for_ms(
    msin: str,
    fallback_frequency_bandwidth: list[object],
    *,
    max_bandwidth_hz: float = WSCLEAN_PREDICT_MAX_BANDWIDTH_HZ,
) -> list[dict[str, object]]:
    """Split MS channels into narrow WSClean model-image frequency chunks."""
    frequencies = _measurement_set_channel_frequencies(msin)
    if not frequencies:
        raise ValueError(f"No channel frequencies were found in {msin}")

    if len(frequencies) == 1:
        return [
            {
                "frequency_bandwidth": [frequencies[0], float(fallback_frequency_bandwidth[1])],
                "channel_range": (0, 0),
            }
        ]

    channel_widths = [
        abs(next_frequency - frequency)
        for frequency, next_frequency in zip(frequencies, frequencies[1:])
    ]
    channel_width = max(min(channel_widths), 1.0)
    total_bandwidth = abs(max(frequencies) - min(frequencies)) + channel_width
    chunk_count = max(1, math.ceil(total_bandwidth / max_bandwidth_hz))
    channels_per_chunk = math.ceil(len(frequencies) / chunk_count)

    chunks = []
    for start_channel in range(0, len(frequencies), channels_per_chunk):
        end_channel = min(start_channel + channels_per_chunk, len(frequencies)) - 1
        chunk_frequencies = frequencies[start_channel : end_channel + 1]
        bandwidth = abs(max(chunk_frequencies) - min(chunk_frequencies)) + channel_width
        chunks.append(
            {
                "frequency_bandwidth": [
                    sum(chunk_frequencies) / len(chunk_frequencies),
                    bandwidth,
                ],
                "channel_range": (start_channel, end_channel),
            }
        )
    return chunks


def _ensure_wsclean_model_fits(model_root: str) -> None:
    """Expose WSClean's drawn term-0 model using the filename predict expects."""
    root = Path(model_root)
    term_model = root.with_name(f"{root.name}-term-0.fits")
    predict_model = root.with_name(f"{root.name}-model.fits")
    require_file(str(term_model), "WSClean predict model image")

    if predict_model.exists() or predict_model.is_symlink():
        predict_model.unlink()
    try:
        predict_model.symlink_to(term_model.name)
    except OSError:
        shutil.copyfile(term_model, predict_model)
    require_file(str(predict_model), "WSClean predict model image link")


def _make_tree_writable(path: Path) -> None:
    """Ensure a copied Measurement Set can accept new model-data columns."""
    for item in [path, *path.rglob("*")]:
        try:
            item.chmod(item.stat().st_mode | stat.S_IWUSR)
        except OSError:
            pass

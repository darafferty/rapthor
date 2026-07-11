"""Prefect flows for imaging."""

from typing import Mapping, Optional

from prefect import flow, task
from prefect.exceptions import UnfinishedRun

import rapthor.execution.image.sector as image_sector
from rapthor.execution.config import ExecutionConfig
from rapthor.execution.image.contracts import ImageSectorPayload
from rapthor.execution.image.validation import validate_image_payload
from rapthor.execution.payloads import assert_serializable_payload
from rapthor.execution.prefect_logging import publish_python_logs_to_prefect
from rapthor.execution.run_names import operation_run_name, task_run_name, task_run_options
from rapthor.execution.task_metrics import record_task_runtime
from rapthor.execution.task_runner import run_flow_with_task_runner
from rapthor.lib.records import validate_output_record


@task(name="sector")
def image_sector_task(
    sector: ImageSectorPayload,
    pipeline_working_dir: str,
    execution_config: Optional[ExecutionConfig] = None,
    shell_operation_cls=None,
) -> dict:
    """Prefect task wrapper for one imaging sector."""
    with publish_python_logs_to_prefect(), record_task_runtime(pipeline_working_dir):
        return image_sector.run_image_sector(
            sector,
            pipeline_working_dir,
            execution_config=execution_config,
            shell_operation_cls=shell_operation_cls,
        )


@task(name="prepare_imaging_data")
def image_sector_prepare_visibility_task(
    sector: ImageSectorPayload,
    prepare_task: Mapping[str, object],
    pipeline_working_dir: str,
    execution_config: Optional[ExecutionConfig] = None,
    shell_operation_cls=None,
) -> dict:
    """Prefect task wrapper for one per-observation imaging-data preparation."""
    with publish_python_logs_to_prefect(), record_task_runtime(pipeline_working_dir):
        prepared_record = image_sector.prepare_image_sector_visibility(
            sector,
            prepare_task,
            pipeline_working_dir,
            execution_config=execution_config,
            shell_operation_cls=shell_operation_cls,
        )
    assert_serializable_payload(prepared_record)
    return prepared_record


@task(name="concatenate_visibilities")
def image_sector_concatenate_task(
    sector: ImageSectorPayload,
    prepared_records: list[Mapping[str, object]],
    pipeline_working_dir: str,
    execution_config: Optional[ExecutionConfig] = None,
    shell_operation_cls=None,
) -> dict:
    """Prefect task wrapper for concatenating prepared imaging data."""
    assert_serializable_payload(prepared_records)
    with publish_python_logs_to_prefect(), record_task_runtime(pipeline_working_dir):
        concat_record = image_sector.concatenate_image_sector_visibilities(
            sector,
            [dict(record) for record in prepared_records],
            pipeline_working_dir,
            execution_config=execution_config,
            shell_operation_cls=shell_operation_cls,
        )
    assert_serializable_payload(concat_record)
    return concat_record


@task(name="wsclean_image")
def image_sector_wsclean_task(
    sector: ImageSectorPayload,
    concat_record: Mapping[str, object],
    pipeline_working_dir: str,
    execution_config: Optional[ExecutionConfig] = None,
    shell_operation_cls=None,
) -> dict:
    """Prefect task wrapper for running or reusing WSClean images."""
    assert_serializable_payload(concat_record)
    with publish_python_logs_to_prefect(), record_task_runtime(pipeline_working_dir):
        image_products = image_sector.make_image_sector_wsclean_products(
            sector,
            concat_record,
            pipeline_working_dir,
            execution_config=execution_config,
            shell_operation_cls=shell_operation_cls,
        )
    assert_serializable_payload(image_products)
    return image_products


@task(name="finish_wsclean_images")
def image_sector_finish_wsclean_task(
    sector: ImageSectorPayload,
    image_products: Mapping[str, object],
    pipeline_working_dir: str,
    execution_config: Optional[ExecutionConfig] = None,
    shell_operation_cls=None,
) -> dict:
    """Prefect task wrapper for bright-source restoration and beam checks."""
    assert_serializable_payload(image_products)
    with publish_python_logs_to_prefect(), record_task_runtime(pipeline_working_dir):
        finished_products = image_sector.finish_image_sector_wsclean_products(
            sector,
            image_products,
            pipeline_working_dir,
            execution_config=execution_config,
            shell_operation_cls=shell_operation_cls,
        )
    assert_serializable_payload(finished_products)
    return finished_products


@task(name="make_residual_visibilities")
def image_sector_residual_visibilities_task(
    sector: ImageSectorPayload,
    concat_record: Mapping[str, object],
    image_products: Mapping[str, object],
    pipeline_working_dir: str,
    execution_config: Optional[ExecutionConfig] = None,
    shell_operation_cls=None,
) -> dict:
    """Prefect task wrapper for optional residual Measurement Set creation."""
    assert_serializable_payload(concat_record)
    assert_serializable_payload(image_products)
    with publish_python_logs_to_prefect(), record_task_runtime(pipeline_working_dir):
        residual_result = image_sector.make_image_sector_residual_visibility_product(
            sector,
            concat_record,
            image_products,
            pipeline_working_dir,
            execution_config=execution_config,
            shell_operation_cls=shell_operation_cls,
        )
    assert_serializable_payload(residual_result)
    return residual_result


@task(name="prepare_outputs")
def image_sector_prepare_outputs_task(
    prepared_records: list[Mapping[str, object]],
    concat_record: Mapping[str, object],
    image_products: Mapping[str, object],
    pipeline_working_dir: str,
    residual_result: Optional[Mapping[str, object]] = None,
) -> dict:
    """Prefect task wrapper for assembling the downstream prepared payload."""
    assert_serializable_payload(prepared_records)
    assert_serializable_payload(concat_record)
    assert_serializable_payload(image_products)
    assert_serializable_payload(residual_result)
    with publish_python_logs_to_prefect(), record_task_runtime(pipeline_working_dir):
        prepared = image_sector.assemble_image_sector_preparation(
            [dict(record) for record in prepared_records],
            concat_record,
            image_products,
            residual_result,
        )
    assert_serializable_payload(prepared)
    return prepared


@task(name="filter_skymodel")
def image_sector_filter_skymodel_task(
    sector: ImageSectorPayload,
    prepared: Mapping[str, object],
    pipeline_working_dir: str,
    execution_config: Optional[ExecutionConfig] = None,
    shell_operation_cls=None,
) -> dict:
    """Prefect task wrapper for one sector skymodel-filtering step."""
    assert_serializable_payload(prepared)
    with publish_python_logs_to_prefect(), record_task_runtime(pipeline_working_dir):
        filtered = image_sector.filter_image_sector_skymodel(
            sector,
            prepared,
            pipeline_working_dir,
            execution_config=execution_config,
            shell_operation_cls=shell_operation_cls,
        )
    assert_serializable_payload(filtered)
    return filtered


@task(name="calculate_image_diagnostics")
def image_sector_diagnostics_task(
    sector: ImageSectorPayload,
    prepared: Mapping[str, object],
    filtered: Mapping[str, object],
    pipeline_working_dir: str,
) -> dict:
    """Prefect task wrapper for one sector diagnostics calculation."""
    assert_serializable_payload(prepared)
    assert_serializable_payload(filtered)
    with publish_python_logs_to_prefect(), record_task_runtime(pipeline_working_dir):
        diagnostics = image_sector.calculate_image_sector_diagnostics(
            sector,
            prepared,
            filtered,
            pipeline_working_dir,
        )
    assert_serializable_payload(diagnostics)
    return diagnostics


@task(name="make_image_cube")
def image_sector_make_image_cube_task(
    sector: ImageSectorPayload,
    prepared: Mapping[str, object],
    pipeline_working_dir: str,
) -> dict:
    """Prefect task wrapper for one sector image-cube creation step."""
    assert_serializable_payload(prepared)
    with publish_python_logs_to_prefect(), record_task_runtime(pipeline_working_dir):
        image_cubes = image_sector.make_image_sector_cubes(
            sector,
            pipeline_working_dir,
        )
    assert_serializable_payload(image_cubes)
    return image_cubes


@task(name="make_catalog_from_image_cube")
def image_sector_make_catalog_from_image_cube_task(
    sector: ImageSectorPayload,
    image_cubes: Mapping[str, object],
    pipeline_working_dir: str,
    execution_config: Optional[ExecutionConfig] = None,
    shell_operation_cls=None,
) -> dict:
    """Prefect task wrapper for one sector image-cube catalog step."""
    assert_serializable_payload(image_cubes)
    with publish_python_logs_to_prefect(), record_task_runtime(pipeline_working_dir):
        catalog = image_sector.make_image_sector_cube_catalog(
            sector,
            image_cubes,
            execution_config=execution_config,
            shell_operation_cls=shell_operation_cls,
        )
    assert_serializable_payload(catalog)
    return catalog


@task(name="normalize_flux_scale")
def image_sector_normalize_flux_scale_task(
    sector: ImageSectorPayload,
    prepared: Mapping[str, object],
    catalog: Mapping[str, object],
    pipeline_working_dir: str,
) -> dict:
    """Prefect task wrapper for one sector flux-scale normalization step."""
    assert_serializable_payload(prepared)
    assert_serializable_payload(catalog)
    with publish_python_logs_to_prefect(), record_task_runtime(pipeline_working_dir):
        normalization = image_sector.normalize_image_sector_flux_scale(
            sector,
            prepared,
            catalog,
        )
    assert_serializable_payload(normalization)
    return normalization


@task(name="restore_skymodel")
def image_sector_restore_skymodel_task(
    sector: ImageSectorPayload,
    prepared: Mapping[str, object],
    filtered: Mapping[str, object],
    pipeline_working_dir: str,
) -> dict:
    """Prefect task wrapper for one sector filtered-skymodel restoration step."""
    assert_serializable_payload(prepared)
    assert_serializable_payload(filtered)
    with publish_python_logs_to_prefect(), record_task_runtime(pipeline_working_dir):
        restored_model = image_sector.restore_image_sector_skymodel(
            sector,
            prepared,
            filtered,
        )
    assert_serializable_payload(restored_model)
    return restored_model


@task(name="compress_images")
def image_sector_compress_images_task(
    sector: ImageSectorPayload,
    prepared: Mapping[str, object],
    diagnostics: Mapping[str, object],
    pipeline_working_dir: str,
    execution_config: Optional[ExecutionConfig] = None,
    shell_operation_cls=None,
) -> dict:
    """Prefect task wrapper for one sector image-compression step."""
    assert_serializable_payload(prepared)
    assert_serializable_payload(diagnostics)
    with publish_python_logs_to_prefect(), record_task_runtime(pipeline_working_dir):
        compression = image_sector.compress_image_sector_products(
            sector,
            prepared,
            diagnostics,
            pipeline_working_dir,
            execution_config=execution_config,
            shell_operation_cls=shell_operation_cls,
        )
    assert_serializable_payload(compression)
    return compression


@task(name="finalize")
def image_sector_finalize_task(
    sector: ImageSectorPayload,
    prepared: Mapping[str, object],
    filtered: Mapping[str, object],
    diagnostics: Mapping[str, object],
    pipeline_working_dir: str,
    image_cubes: Optional[Mapping[str, object]] = None,
    catalog: Optional[Mapping[str, object]] = None,
    normalization: Optional[Mapping[str, object]] = None,
    restored_model: Optional[Mapping[str, object]] = None,
    compression: Optional[Mapping[str, object]] = None,
    execution_config: Optional[ExecutionConfig] = None,
    shell_operation_cls=None,
) -> dict:
    """Prefect task wrapper for post-WSClean sector finalization."""
    assert_serializable_payload(prepared)
    assert_serializable_payload(filtered)
    assert_serializable_payload(diagnostics)
    assert_serializable_payload(image_cubes)
    assert_serializable_payload(catalog)
    assert_serializable_payload(normalization)
    assert_serializable_payload(restored_model)
    assert_serializable_payload(compression)
    with publish_python_logs_to_prefect(), record_task_runtime(pipeline_working_dir):
        return image_sector.finalize_image_sector(
            sector,
            prepared,
            filtered,
            diagnostics,
            pipeline_working_dir,
            image_cube_result=image_cubes,
            catalog_result=catalog,
            normalization_result=normalization,
            restored_model_result=restored_model,
            compression_result=compression,
            execution_config=execution_config,
            shell_operation_cls=shell_operation_cls,
        )


def _result_from_sector_records(sector_outputs: list[dict]) -> dict:
    result = {
        "filtered_skymodel_true_sky": [
            sector["filtered_skymodel_true_sky"] for sector in sector_outputs
        ],
        "filtered_skymodel_apparent_sky": [
            sector["filtered_skymodel_apparent_sky"] for sector in sector_outputs
        ],
        "pybdsf_catalog": [sector["pybdsf_catalog"] for sector in sector_outputs],
        "sector_diagnostics": [sector["sector_diagnostics"] for sector in sector_outputs],
        "sector_offsets": [sector["sector_offsets"] for sector in sector_outputs],
        "sector_diagnostic_plots": [sector["sector_diagnostic_plots"] for sector in sector_outputs],
        "visibilities": [sector["visibilities"] for sector in sector_outputs],
        "sector_I_images": [sector["sector_I_images"] for sector in sector_outputs],
        "sector_extra_images": [sector["sector_extra_images"] for sector in sector_outputs],
        "source_filtering_mask": [sector["source_filtering_mask"] for sector in sector_outputs],
    }
    if any(sector["sector_skymodels"] is not None for sector in sector_outputs):
        result["sector_skymodels"] = [sector["sector_skymodels"] for sector in sector_outputs]
    if any("sector_region_file" in sector for sector in sector_outputs):
        result["sector_region_file"] = [
            sector.get("sector_region_file") for sector in sector_outputs
        ]
    if any("sector_skymodel_image_fits" in sector for sector in sector_outputs):
        result["sector_skymodel_image_fits"] = [
            sector.get("sector_skymodel_image_fits") for sector in sector_outputs
        ]
    if any("sector_image_cubes" in sector for sector in sector_outputs):
        result["sector_image_cubes"] = [
            sector.get("sector_image_cubes") for sector in sector_outputs
        ]
        result["sector_image_cube_beams"] = [
            sector.get("sector_image_cube_beams") for sector in sector_outputs
        ]
        result["sector_image_cube_frequencies"] = [
            sector.get("sector_image_cube_frequencies") for sector in sector_outputs
        ]
    if any("sector_normalize_h5parm" in sector for sector in sector_outputs):
        result["sector_source_catalog"] = [
            sector.get("sector_source_catalog") for sector in sector_outputs
        ]
        result["sector_normalize_h5parm"] = [
            sector.get("sector_normalize_h5parm") for sector in sector_outputs
        ]
    if any("residual_visibilities" in sector for sector in sector_outputs):
        result["residual_visibilities"] = [
            sector.get("residual_visibilities") for sector in sector_outputs
        ]
    for value in result.values():
        validate_output_record(value, allow_none=True)
    return result


def _submit_split_image_sector_tasks(
    payload: Mapping[str, object],
    config: ExecutionConfig,
):
    sectors = list(payload["sectors"])

    def sector_step_name(index: int, step: str) -> str:
        if len(sectors) == 1:
            return task_run_name(step)
        sector = sectors[index]
        return task_run_name(sector.get("image_name") or f"sector_{index + 1}", step)

    prepared_record_futures = [
        [
            image_sector_prepare_visibility_task.with_options(
                **task_run_options(
                    sector_step_name(index, f"prepare_imaging_data_{task_index + 1}"),
                    tags=["dp3"],
                )
            ).submit(
                sector,
                prepare_task,
                payload["pipeline_working_dir"],
                execution_config=config,
            )
            for task_index, prepare_task in enumerate(sector["prepare_tasks"])
        ]
        for index, sector in enumerate(sectors)
    ]
    concat_sector_futures = [
        image_sector_concatenate_task.with_options(
            **task_run_options(
                sector_step_name(index, "concatenate_visibilities"),
                tags=["casacore"],
            )
        ).submit(
            sector,
            prepared_record_futures[index],
            payload["pipeline_working_dir"],
            execution_config=config,
        )
        for index, sector in enumerate(sectors)
    ]
    wsclean_sector_futures = [
        image_sector_wsclean_task.with_options(
            **task_run_options(sector_step_name(index, "wsclean_image"), tags=["wsclean"])
        ).submit(
            sector,
            concat_sector_futures[index],
            payload["pipeline_working_dir"],
            execution_config=config,
        )
        for index, sector in enumerate(sectors)
    ]
    finished_wsclean_sector_futures = [
        image_sector_finish_wsclean_task.with_options(
            **task_run_options(
                sector_step_name(index, "finish_wsclean_images"),
                tags=["wsclean"] if bool(sector.get("peel_bright_sources", False)) else ["python"],
            )
        ).submit(
            sector,
            wsclean_sector_futures[index],
            payload["pipeline_working_dir"],
            execution_config=config,
        )
        for index, sector in enumerate(sectors)
    ]
    residual_visibilities_sector_futures = [
        (
            image_sector_residual_visibilities_task.with_options(
                **task_run_options(
                    sector_step_name(index, "make_residual_visibilities"),
                    tags=["dp3"],
                )
            ).submit(
                sector,
                concat_sector_futures[index],
                finished_wsclean_sector_futures[index],
                payload["pipeline_working_dir"],
                execution_config=config,
            )
            if bool(sector.get("make_residual_visibilities", False))
            else None
        )
        for index, sector in enumerate(sectors)
    ]
    prepared_sector_futures = [
        image_sector_prepare_outputs_task.with_options(
            **task_run_options(sector_step_name(index, "prepare_outputs"), tags=["python"])
        ).submit(
            prepared_record_futures[index],
            concat_sector_futures[index],
            finished_wsclean_sector_futures[index],
            payload["pipeline_working_dir"],
            residual_visibilities_sector_futures[index],
        )
        for index, _sector in enumerate(sectors)
    ]
    filtered_sector_futures = [
        image_sector_filter_skymodel_task.with_options(
            **task_run_options(sector_step_name(index, "filter_skymodel"), tags=["python"])
        ).submit(
            sector,
            prepared_sector_futures[index],
            payload["pipeline_working_dir"],
            execution_config=config,
        )
        for index, sector in enumerate(sectors)
    ]
    diagnostics_sector_futures = [
        image_sector_diagnostics_task.with_options(
            **task_run_options(
                sector_step_name(index, "calculate_image_diagnostics"),
                tags=["python"],
            )
        ).submit(
            sector,
            prepared_sector_futures[index],
            filtered_sector_futures[index],
            payload["pipeline_working_dir"],
        )
        for index, sector in enumerate(sectors)
    ]
    image_cube_sector_futures = [
        (
            image_sector_make_image_cube_task.with_options(
                **task_run_options(sector_step_name(index, "make_image_cube"), tags=["python"])
            ).submit(
                sector,
                prepared_sector_futures[index],
                payload["pipeline_working_dir"],
            )
            if bool(sector.get("make_image_cube", False))
            else None
        )
        for index, sector in enumerate(sectors)
    ]
    catalog_sector_futures = [
        (
            image_sector_make_catalog_from_image_cube_task.with_options(
                **task_run_options(
                    sector_step_name(index, "make_catalog_from_image_cube"),
                    tags=["pybdsf"],
                )
            ).submit(
                sector,
                image_cube_sector_futures[index],
                payload["pipeline_working_dir"],
                execution_config=config,
            )
            if bool(sector.get("normalize_flux_scale", False))
            else None
        )
        for index, sector in enumerate(sectors)
    ]
    normalization_sector_futures = [
        (
            image_sector_normalize_flux_scale_task.with_options(
                **task_run_options(
                    sector_step_name(index, "normalize_flux_scale"),
                    tags=["python"],
                )
            ).submit(
                sector,
                prepared_sector_futures[index],
                catalog_sector_futures[index],
                payload["pipeline_working_dir"],
            )
            if bool(sector.get("normalize_flux_scale", False))
            else None
        )
        for index, sector in enumerate(sectors)
    ]
    restored_model_sector_futures = [
        (
            image_sector_restore_skymodel_task.with_options(
                **task_run_options(sector_step_name(index, "restore_skymodel"), tags=["python"])
            ).submit(
                sector,
                prepared_sector_futures[index],
                filtered_sector_futures[index],
                payload["pipeline_working_dir"],
            )
            if bool(sector.get("save_filtered_model_image", False))
            else None
        )
        for index, sector in enumerate(sectors)
    ]
    compression_sector_futures = [
        (
            image_sector_compress_images_task.with_options(
                **task_run_options(sector_step_name(index, "compress_images"), tags=["fpack"])
            ).submit(
                sector,
                prepared_sector_futures[index],
                diagnostics_sector_futures[index],
                payload["pipeline_working_dir"],
                execution_config=config,
            )
            if bool(sector.get("compress_images", False))
            else None
        )
        for index, sector in enumerate(sectors)
    ]
    finalized_sector_futures = [
        image_sector_finalize_task.with_options(
            **task_run_options(sector_step_name(index, "finalize"), tags=["python"])
        ).submit(
            sector,
            prepared_sector_futures[index],
            filtered_sector_futures[index],
            diagnostics_sector_futures[index],
            payload["pipeline_working_dir"],
            image_cube_sector_futures[index],
            catalog_sector_futures[index],
            normalization_sector_futures[index],
            restored_model_sector_futures[index],
            compression_sector_futures[index],
            execution_config=config,
        )
        for index, sector in enumerate(sectors)
    ]
    return (
        prepared_record_futures,
        concat_sector_futures,
        wsclean_sector_futures,
        finished_wsclean_sector_futures,
        residual_visibilities_sector_futures,
        prepared_sector_futures,
        filtered_sector_futures,
        diagnostics_sector_futures,
        image_cube_sector_futures,
        catalog_sector_futures,
        normalization_sector_futures,
        restored_model_sector_futures,
        compression_sector_futures,
        finalized_sector_futures,
    )


def _collect_image_sector_results(
    prepared_record_futures,
    concat_sector_futures,
    wsclean_sector_futures,
    finished_wsclean_sector_futures,
    residual_visibilities_sector_futures,
    prepared_sector_futures,
    filtered_sector_futures,
    diagnostics_sector_futures,
    image_cube_sector_futures,
    catalog_sector_futures,
    normalization_sector_futures,
    restored_model_sector_futures,
    compression_sector_futures,
    finalized_sector_futures,
) -> list[dict]:
    try:
        return [output.result() for output in finalized_sector_futures]
    except UnfinishedRun:
        for sector_prepared_records in prepared_record_futures:
            for prepared_record in sector_prepared_records:
                prepared_record.result()
        for concat_sector in concat_sector_futures:
            concat_sector.result()
        for wsclean_sector in wsclean_sector_futures:
            wsclean_sector.result()
        for finished_wsclean_sector in finished_wsclean_sector_futures:
            finished_wsclean_sector.result()
        for residual_visibilities_sector in residual_visibilities_sector_futures:
            if residual_visibilities_sector is not None:
                residual_visibilities_sector.result()
        for prepared_sector in prepared_sector_futures:
            prepared_sector.result()
        for filtered_sector in filtered_sector_futures:
            filtered_sector.result()
        for diagnostics_sector in diagnostics_sector_futures:
            diagnostics_sector.result()
        for image_cube_sector in image_cube_sector_futures:
            if image_cube_sector is not None:
                image_cube_sector.result()
        for catalog_sector in catalog_sector_futures:
            if catalog_sector is not None:
                catalog_sector.result()
        for normalization_sector in normalization_sector_futures:
            if normalization_sector is not None:
                normalization_sector.result()
        for restored_model_sector in restored_model_sector_futures:
            if restored_model_sector is not None:
                restored_model_sector.result()
        for compression_sector in compression_sector_futures:
            if compression_sector is not None:
                compression_sector.result()
        raise


def _run_image_prefect_tasks(
    payload: Mapping[str, object],
    execution_config: Optional[ExecutionConfig] = None,
) -> dict:
    assert_serializable_payload(payload)
    config = execution_config or ExecutionConfig(task_runner="sync")
    payload = validate_image_payload(payload)
    (
        prepared_record_futures,
        concat_sector_futures,
        wsclean_sector_futures,
        finished_wsclean_sector_futures,
        residual_visibilities_sector_futures,
        prepared_sector_futures,
        filtered_sector_futures,
        diagnostics_sector_futures,
        image_cube_sector_futures,
        catalog_sector_futures,
        normalization_sector_futures,
        restored_model_sector_futures,
        compression_sector_futures,
        finalized_sector_futures,
    ) = _submit_split_image_sector_tasks(payload, config)
    sector_outputs = _collect_image_sector_results(
        prepared_record_futures,
        concat_sector_futures,
        wsclean_sector_futures,
        finished_wsclean_sector_futures,
        residual_visibilities_sector_futures,
        prepared_sector_futures,
        filtered_sector_futures,
        diagnostics_sector_futures,
        image_cube_sector_futures,
        catalog_sector_futures,
        normalization_sector_futures,
        restored_model_sector_futures,
        compression_sector_futures,
        finalized_sector_futures,
    )
    return _result_from_sector_records(sector_outputs)


@flow(name="image")
def _image_flow(
    payload: Mapping[str, object],
    execution_config: Optional[ExecutionConfig] = None,
):
    """Prefect implementation for imaging."""
    with publish_python_logs_to_prefect():
        return _run_image_prefect_tasks(payload, execution_config=execution_config)


def image_flow(
    payload: Mapping[str, object],
    execution_config: Optional[ExecutionConfig] = None,
):
    """Prefect entry point for imaging."""
    return run_flow_with_task_runner(
        _image_flow,
        payload,
        flow_run_name=operation_run_name(payload, "image"),
        execution_config=execution_config,
    )

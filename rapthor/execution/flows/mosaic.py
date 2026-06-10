"""Prefect flow for the Mosaic operation."""

import os
from typing import Mapping, Optional

from prefect import flow, task

from rapthor.execution.commands import normalize_command
from rapthor.execution.artifacts import publish_fits_image_artifacts
from rapthor.execution.config import ExecutionConfig
from rapthor.execution.flows.runtime import run_flow_with_task_runner
from rapthor.execution.outputs import file_record, validate_output_record
from rapthor.execution.payloads import assert_serializable_payload
from rapthor.execution.shell import ShellCommand, run_shell_command


def _bool_token(value: bool) -> str:
    return "True" if value else "False"


def _file_record_path(record: object) -> str:
    if isinstance(record, Mapping) and record.get("class") == "File":
        path = record.get("path")
        if isinstance(path, str) and path:
            return path
    raise ValueError(f"Expected a File output record, got {record!r}")


def _validate_basename(filename: object, name: str) -> str:
    if not isinstance(filename, str) or not filename:
        raise ValueError(f"{name} must be a non-empty string")
    if os.path.isabs(filename) or os.path.basename(filename) != filename:
        raise ValueError(f"{name} must be a basename")
    return filename


def _validate_image_type_payloads(payload: Mapping[str, object]) -> tuple[str, bool, list[Mapping]]:
    pipeline_working_dir = str(payload["pipeline_working_dir"])
    compress_images = bool(payload.get("compress_images", False))
    image_types = payload.get("image_types", [])
    if not isinstance(image_types, list):
        raise ValueError("image_types must be a list")
    for index, image_type in enumerate(image_types):
        if not isinstance(image_type, Mapping):
            raise ValueError(f"image_types[{index}] must be a mapping")
        template_filename = _validate_basename(
            image_type.get("template_image_filename"), "template_image_filename"
        )
        mosaic_filename = _validate_basename(image_type.get("mosaic_filename"), "mosaic_filename")
        expected_template_path = os.path.join(pipeline_working_dir, template_filename)
        expected_mosaic_path = os.path.join(pipeline_working_dir, mosaic_filename)
        if str(image_type.get("template_image_path")) != expected_template_path:
            raise ValueError(
                f"image_types[{index}].template_image_path must be {expected_template_path}"
            )
        if str(image_type.get("mosaic_path")) != expected_mosaic_path:
            raise ValueError(f"image_types[{index}].mosaic_path must be {expected_mosaic_path}")
        sector_images = image_type.get("sector_image_filenames")
        sector_vertices = image_type.get("sector_vertices_filenames")
        regridded_images = image_type.get("regridded_image_filenames")
        if not isinstance(sector_images, list):
            raise ValueError(f"image_types[{index}].sector_image_filenames must be a list")
        if not isinstance(sector_vertices, list):
            raise ValueError(f"image_types[{index}].sector_vertices_filenames must be a list")
        if not isinstance(regridded_images, list):
            raise ValueError(f"image_types[{index}].regridded_image_filenames must be a list")
        if len(sector_images) != len(sector_vertices) or len(sector_images) != len(
            regridded_images
        ):
            raise ValueError(f"image_types[{index}] input and regridded lists must match")
        for regridded_index, regridded_image in enumerate(regridded_images):
            _validate_basename(
                regridded_image,
                f"image_types[{index}].regridded_image_filenames[{regridded_index}]",
            )
    return pipeline_working_dir, compress_images, image_types


def _validate_unique_mosaic_paths(image_types: list[Mapping]) -> None:
    seen_mosaic_paths = set()
    for index, image_type in enumerate(image_types):
        mosaic_path = str(image_type["mosaic_path"])
        if mosaic_path in seen_mosaic_paths:
            raise ValueError(f"image_types[{index}].mosaic_path must be unique")
        seen_mosaic_paths.add(mosaic_path)


def _join_path_list(paths: list[str]) -> str:
    return ",".join(paths)


def build_make_mosaic_template_command(
    input_image_filenames: list[str],
    sector_vertices_filenames: list[str],
    template_image_filename: str,
    skip_processing: bool = False,
) -> list[str]:
    """Build the `make_mosaic_template.py` command for one image type."""
    return [
        "make_mosaic_template.py",
        _join_path_list(input_image_filenames),
        _join_path_list(sector_vertices_filenames),
        template_image_filename,
        f"--skip={_bool_token(skip_processing)}",
    ]


def build_regrid_image_command(
    input_image_filename: str,
    template_image_filename: str,
    sector_vertices_filename: str,
    regridded_image_filename: str,
    skip_processing: bool = False,
) -> list[str]:
    """Build the `regrid_image.py` command for one sector image."""
    return [
        "regrid_image.py",
        input_image_filename,
        template_image_filename,
        sector_vertices_filename,
        regridded_image_filename,
        f"--skip={_bool_token(skip_processing)}",
    ]


def build_make_mosaic_command(
    regridded_image_filenames: list[str],
    template_image_filename: str,
    mosaic_filename: str,
    skip_processing: bool = False,
) -> list[str]:
    """Build the `make_mosaic.py` command for one image type."""
    return [
        "make_mosaic.py",
        _join_path_list(regridded_image_filenames),
        template_image_filename,
        mosaic_filename,
        f"--skip={_bool_token(skip_processing)}",
    ]


def build_compress_mosaic_command(mosaic_filename: str) -> list[str]:
    """Build the `fpack` command for one mosaic image."""
    return ["fpack", mosaic_filename]


def mosaic_payload_from_inputs(
    input_parms: Mapping[str, object],
    pipeline_working_dir: object,
    compress_images: bool = False,
) -> dict:
    """Create a serializable Mosaic flow payload from operation inputs."""
    pipeline_dir = str(pipeline_working_dir)
    skip_processing = bool(input_parms.get("skip_processing", False))
    if skip_processing:
        return assert_serializable_payload(
            {
                "pipeline_working_dir": pipeline_dir,
                "compress_images": bool(compress_images),
                "skip_processing": True,
                "image_types": [],
            }
        )

    sector_image_filenames = input_parms.get("sector_image_filename", [])
    sector_vertices_filenames = input_parms.get("sector_vertices_filename", [])
    template_image_filenames = input_parms.get("template_image_filename", [])
    regridded_image_filenames = input_parms.get("regridded_image_filename", [])
    mosaic_filenames = input_parms.get("mosaic_filename", [])
    image_type_inputs = [
        sector_image_filenames,
        sector_vertices_filenames,
        template_image_filenames,
        regridded_image_filenames,
        mosaic_filenames,
    ]
    if not all(isinstance(value, list) for value in image_type_inputs):
        raise ValueError("Mosaic inputs must be lists")
    image_type_count = len(mosaic_filenames)
    if any(len(value) != image_type_count for value in image_type_inputs):
        raise ValueError("Mosaic input lists must have the same length")

    image_types = []
    for index in range(image_type_count):
        sector_images = sector_image_filenames[index]
        sector_vertices = sector_vertices_filenames[index]
        regridded_images = regridded_image_filenames[index]
        if not isinstance(sector_images, list):
            raise ValueError(f"sector_image_filename[{index}] must be a list")
        if not isinstance(sector_vertices, list):
            raise ValueError(f"sector_vertices_filename[{index}] must be a list")
        if not isinstance(regridded_images, list):
            raise ValueError(f"regridded_image_filename[{index}] must be a list")
        if len(sector_images) != len(sector_vertices) or len(sector_images) != len(
            regridded_images
        ):
            raise ValueError(f"Mosaic scatter lists at index {index} must have the same length")

        template_filename = _validate_basename(
            template_image_filenames[index], f"template_image_filename[{index}]"
        )
        mosaic_filename = _validate_basename(mosaic_filenames[index], f"mosaic_filename[{index}]")
        image_types.append(
            {
                "sector_image_filenames": [_file_record_path(record) for record in sector_images],
                "sector_vertices_filenames": [
                    _file_record_path(record) for record in sector_vertices
                ],
                "template_image_filename": template_filename,
                "template_image_path": os.path.join(pipeline_dir, template_filename),
                "regridded_image_filenames": [
                    _validate_basename(filename, f"regridded_image_filename[{index}]")
                    for filename in regridded_images
                ],
                "mosaic_filename": mosaic_filename,
                "mosaic_path": os.path.join(pipeline_dir, mosaic_filename),
            }
        )

    payload = {
        "pipeline_working_dir": pipeline_dir,
        "compress_images": bool(compress_images),
        "skip_processing": False,
        "image_types": image_types,
    }
    _validate_unique_mosaic_paths(image_types)
    return assert_serializable_payload(payload)


def _run_shell_and_validate(
    command: list[str],
    output_path: str,
    pipeline_working_dir: str,
    execution_config: ExecutionConfig,
    shell_operation_cls=None,
) -> None:
    run_shell_command(
        ShellCommand(command=command, working_directory=pipeline_working_dir),
        execution_config,
        shell_operation_cls=shell_operation_cls,
    )
    if not os.path.isfile(output_path):
        raise FileNotFoundError(f"Mosaic output was not created: {output_path}")


def run_mosaic_image_type(
    image_type: Mapping[str, object],
    compress_images: bool,
    pipeline_working_dir: str,
    execution_config: Optional[ExecutionConfig] = None,
    shell_operation_cls=None,
) -> dict:
    """Run mosaicking for one image type and return a file output record."""
    config = execution_config or ExecutionConfig(task_runner="sync")
    sector_images = list(image_type["sector_image_filenames"])
    sector_vertices = list(image_type["sector_vertices_filenames"])
    regridded_images = list(image_type["regridded_image_filenames"])
    template_filename = str(image_type["template_image_filename"])
    template_path = str(image_type["template_image_path"])
    mosaic_filename = str(image_type["mosaic_filename"])
    mosaic_path = str(image_type["mosaic_path"])

    _run_shell_and_validate(
        build_make_mosaic_template_command(sector_images, sector_vertices, template_filename),
        template_path,
        pipeline_working_dir,
        config,
        shell_operation_cls=shell_operation_cls,
    )
    for sector_image, vertices_file, regridded_image in zip(
        sector_images, sector_vertices, regridded_images
    ):
        _run_shell_and_validate(
            build_regrid_image_command(
                sector_image,
                template_filename,
                vertices_file,
                regridded_image,
            ),
            os.path.join(pipeline_working_dir, regridded_image),
            pipeline_working_dir,
            config,
            shell_operation_cls=shell_operation_cls,
        )
    _run_shell_and_validate(
        build_make_mosaic_command(regridded_images, template_filename, mosaic_filename),
        mosaic_path,
        pipeline_working_dir,
        config,
        shell_operation_cls=shell_operation_cls,
    )
    if compress_images:
        compressed_path = f"{mosaic_path}.fz"
        _run_shell_and_validate(
            build_compress_mosaic_command(mosaic_filename),
            compressed_path,
            pipeline_working_dir,
            config,
            shell_operation_cls=shell_operation_cls,
        )
        output_record = file_record(compressed_path)
        publish_fits_image_artifacts([output_record], pipeline_working_dir)
        return output_record
    output_record = file_record(mosaic_path)
    publish_fits_image_artifacts([output_record], pipeline_working_dir)
    return output_record


@task(name="mosaic_image_type")
def mosaic_image_type_task(
    image_type: Mapping[str, object],
    compress_images: bool,
    pipeline_working_dir: str,
    execution_config: Optional[ExecutionConfig] = None,
) -> dict:
    """Prefect task wrapper for one image type."""
    return run_mosaic_image_type(
        image_type,
        compress_images,
        pipeline_working_dir,
        execution_config=execution_config,
    )


def _result_from_mosaic_records(outputs: list[dict]) -> dict:
    result = {"mosaic_image": outputs}
    validate_output_record(result["mosaic_image"])
    return result


def run_mosaic_flow(
    payload: Mapping[str, object],
    execution_config: Optional[ExecutionConfig] = None,
    shell_operation_cls=None,
) -> dict:
    """Run mosaic commands and return finalizer-compatible outputs."""
    assert_serializable_payload(payload)
    if payload.get("skip_processing", False):
        return {}

    config = execution_config or ExecutionConfig(task_runner="sync")
    pipeline_working_dir, compress_images, image_types = _validate_image_type_payloads(payload)
    _validate_unique_mosaic_paths(image_types)
    outputs = [
        run_mosaic_image_type(
            image_type,
            compress_images,
            pipeline_working_dir,
            execution_config=config,
            shell_operation_cls=shell_operation_cls,
        )
        for image_type in image_types
    ]
    return _result_from_mosaic_records(outputs)


def _run_mosaic_prefect_tasks(
    payload: Mapping[str, object],
    execution_config: Optional[ExecutionConfig] = None,
) -> dict:
    if payload.get("skip_processing", False):
        return {}

    config = execution_config or ExecutionConfig(task_runner="sync")
    pipeline_working_dir, compress_images, image_types = _validate_image_type_payloads(payload)
    _validate_unique_mosaic_paths(image_types)
    outputs = [
        mosaic_image_type_task.submit(
            image_type,
            compress_images,
            pipeline_working_dir,
            execution_config=config,
        )
        for image_type in image_types
    ]
    outputs = [output.result() for output in outputs]
    return _result_from_mosaic_records(outputs)


@flow(name="mosaic")
def _mosaic_flow(
    payload: Mapping[str, object],
    execution_config: Optional[ExecutionConfig] = None,
):
    """Prefect implementation for Mosaic."""
    return _run_mosaic_prefect_tasks(payload, execution_config=execution_config)


def mosaic_flow(
    payload: Mapping[str, object],
    execution_config: Optional[ExecutionConfig] = None,
):
    """Prefect entry point for Mosaic."""
    return run_flow_with_task_runner(
        _mosaic_flow,
        payload,
        execution_config=execution_config,
    )


def normalized_make_mosaic_template_command(
    input_image_filenames: list[str],
    sector_vertices_filenames: list[str],
    template_image_filename: str,
    skip_processing: bool = False,
) -> list[str]:
    """Return normalized template command tokens for fixture comparisons."""
    return normalize_command(
        build_make_mosaic_template_command(
            input_image_filenames,
            sector_vertices_filenames,
            template_image_filename,
            skip_processing=skip_processing,
        )
    )


def normalized_regrid_image_command(
    input_image_filename: str,
    template_image_filename: str,
    sector_vertices_filename: str,
    regridded_image_filename: str,
    skip_processing: bool = False,
) -> list[str]:
    """Return normalized regrid command tokens for fixture comparisons."""
    return normalize_command(
        build_regrid_image_command(
            input_image_filename,
            template_image_filename,
            sector_vertices_filename,
            regridded_image_filename,
            skip_processing=skip_processing,
        )
    )


def normalized_make_mosaic_command(
    regridded_image_filenames: list[str],
    template_image_filename: str,
    mosaic_filename: str,
    skip_processing: bool = False,
) -> list[str]:
    """Return normalized mosaic command tokens for fixture comparisons."""
    return normalize_command(
        build_make_mosaic_command(
            regridded_image_filenames,
            template_image_filename,
            mosaic_filename,
            skip_processing=skip_processing,
        )
    )

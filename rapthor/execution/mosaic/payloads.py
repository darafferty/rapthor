"""Mosaic execution payload builders and validators."""

import os
from typing import Mapping, TypedDict

from rapthor.execution.payloads import (
    assert_serializable_payload,
    validate_basename,
    validate_string_list,
)
from rapthor.lib.records import file_record_path


class MosaicImageTypePayload(TypedDict):
    """Serializable inputs and expected outputs for one mosaic image type."""

    sector_image_filenames: list[str]
    sector_vertices_filenames: list[str]
    template_image_filename: str
    template_image_path: str
    regridded_image_filenames: list[str]
    mosaic_filename: str
    mosaic_path: str


class MosaicPayload(TypedDict):
    """Serializable payload submitted to the mosaic flow."""

    pipeline_working_dir: str
    compress_images: bool
    skip_processing: bool
    image_types: list[MosaicImageTypePayload]


def _validate_unique_mosaic_paths(image_types: list[MosaicImageTypePayload]) -> None:
    """Reject payloads that would write two image types to the same mosaic."""
    mosaic_paths = [image_type["mosaic_path"] for image_type in image_types]
    if len(mosaic_paths) != len(set(mosaic_paths)):
        raise ValueError("mosaic paths must be unique")


def validate_mosaic_payload(payload: Mapping[str, object]) -> MosaicPayload:
    """Validate a Mosaic payload received by a flow or worker."""
    pipeline_working_dir = str(payload["pipeline_working_dir"])
    compress_images = bool(payload.get("compress_images", False))
    skip_processing = bool(payload.get("skip_processing", False))
    raw_image_types = payload.get("image_types", [])
    if not isinstance(raw_image_types, list):
        raise ValueError("image_types must be a list")
    image_types: list[MosaicImageTypePayload] = []
    for index, image_type in enumerate(raw_image_types):
        if not isinstance(image_type, Mapping):
            raise ValueError(f"image_types[{index}] must be a mapping")
        template_filename = validate_basename(
            image_type.get("template_image_filename"), "template_image_filename"
        )
        mosaic_filename = validate_basename(image_type.get("mosaic_filename"), "mosaic_filename")
        expected_template_path = os.path.join(pipeline_working_dir, template_filename)
        expected_mosaic_path = os.path.join(pipeline_working_dir, mosaic_filename)
        if str(image_type.get("template_image_path")) != expected_template_path:
            raise ValueError(
                f"image_types[{index}].template_image_path must be {expected_template_path}"
            )
        if str(image_type.get("mosaic_path")) != expected_mosaic_path:
            raise ValueError(f"image_types[{index}].mosaic_path must be {expected_mosaic_path}")
        sector_images = validate_string_list(
            image_type.get("sector_image_filenames"),
            f"image_types[{index}].sector_image_filenames",
        )
        sector_vertices = validate_string_list(
            image_type.get("sector_vertices_filenames"),
            f"image_types[{index}].sector_vertices_filenames",
        )
        regridded_images = validate_string_list(
            image_type.get("regridded_image_filenames"),
            f"image_types[{index}].regridded_image_filenames",
        )
        if len(sector_images) != len(sector_vertices) or len(sector_images) != len(
            regridded_images
        ):
            raise ValueError(f"image_types[{index}] input and regridded lists must match")
        image_types.append(
            {
                "sector_image_filenames": sector_images,
                "sector_vertices_filenames": sector_vertices,
                "template_image_filename": template_filename,
                "template_image_path": expected_template_path,
                "regridded_image_filenames": [
                    validate_basename(
                        regridded_image,
                        f"image_types[{index}].regridded_image_filenames[{regridded_index}]",
                    )
                    for regridded_index, regridded_image in enumerate(regridded_images)
                ],
                "mosaic_filename": mosaic_filename,
                "mosaic_path": expected_mosaic_path,
            }
        )
    _validate_unique_mosaic_paths(image_types)
    return {
        "pipeline_working_dir": pipeline_working_dir,
        "compress_images": compress_images,
        "skip_processing": skip_processing,
        "image_types": image_types,
    }


def mosaic_payload_from_inputs(
    input_parms: Mapping[str, object],
    pipeline_working_dir: object,
    compress_images: bool = False,
) -> MosaicPayload:
    """Create a serializable Mosaic flow payload from operation inputs."""
    pipeline_dir = str(pipeline_working_dir)
    skip_processing = bool(input_parms.get("skip_processing", False))
    if skip_processing:
        payload: MosaicPayload = {
            "pipeline_working_dir": pipeline_dir,
            "compress_images": bool(compress_images),
            "skip_processing": True,
            "image_types": [],
        }
        assert_serializable_payload(payload)
        return payload

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

    image_types: list[MosaicImageTypePayload] = []
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

        template_filename = validate_basename(
            template_image_filenames[index], f"template_image_filename[{index}]"
        )
        mosaic_filename = validate_basename(mosaic_filenames[index], f"mosaic_filename[{index}]")
        image_types.append(
            {
                "sector_image_filenames": [file_record_path(record) for record in sector_images],
                "sector_vertices_filenames": [
                    file_record_path(record) for record in sector_vertices
                ],
                "template_image_filename": template_filename,
                "template_image_path": os.path.join(pipeline_dir, template_filename),
                "regridded_image_filenames": [
                    validate_basename(filename, f"regridded_image_filename[{index}]")
                    for filename in regridded_images
                ],
                "mosaic_filename": mosaic_filename,
                "mosaic_path": os.path.join(pipeline_dir, mosaic_filename),
            }
        )

    payload: MosaicPayload = {
        "pipeline_working_dir": pipeline_dir,
        "compress_images": bool(compress_images),
        "skip_processing": False,
        "image_types": image_types,
    }
    _validate_unique_mosaic_paths(image_types)
    assert_serializable_payload(payload)
    return payload

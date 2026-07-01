"""Payload validators for image execution."""

from typing import Mapping

from rapthor.execution.image.contracts import (
    ImageCubeSpecPayload,
    ImagePayload,
    ImagePrepareTaskPayload,
    ImageSectorPayload,
)
from rapthor.execution.payloads import validate_basename as _validate_basename
from rapthor.execution.payloads import validate_int_list as _validate_int_list


def _validate_str_list(values: object, name: str) -> list[str]:
    if not isinstance(values, list) or not all(
        isinstance(value, str) and value for value in values
    ):
        raise ValueError(f"{name} must be a list of strings")
    return list(values)


def _validate_prepare_task(
    prepare_task: Mapping[str, object],
    sector_index: int,
    task_index: int,
) -> ImagePrepareTaskPayload:
    maxinterval = prepare_task.get("maxinterval")
    return {
        "msin": str(prepare_task["msin"]),
        "msout": _validate_basename(
            prepare_task["msout"],
            f"sectors[{sector_index}].prepare_tasks[{task_index}].msout",
        ),
        "msout_path": str(prepare_task["msout_path"]),
        "starttime": str(prepare_task["starttime"]),
        "ntimes": int(prepare_task["ntimes"]),
        "freqstep": int(prepare_task["freqstep"]),
        "timestep": int(prepare_task["timestep"]),
        "maxinterval": None if maxinterval is None else int(maxinterval),
    }


def _validate_image_cube_spec(
    spec: Mapping[str, object],
    sector_index: int,
    spec_index: int,
) -> ImageCubeSpecPayload:
    return {
        "pol": str(spec["pol"]),
        "filename": _validate_basename(
            spec["filename"],
            f"sectors[{sector_index}].image_cube_specs[{spec_index}].filename",
        ),
        "path": str(spec["path"]),
    }


def _validate_image_sector(sector: Mapping[str, object], index: int) -> ImageSectorPayload:
    raw_prepare_tasks = sector.get("prepare_tasks", [])
    if not isinstance(raw_prepare_tasks, list):
        raise ValueError(f"sectors[{index}].prepare_tasks must be a list")
    prepare_tasks = []
    for task_index, prepare_task in enumerate(raw_prepare_tasks):
        if not isinstance(prepare_task, Mapping):
            raise ValueError(f"sectors[{index}].prepare_tasks[{task_index}] must be a mapping")
        prepare_tasks.append(_validate_prepare_task(prepare_task, index, task_index))

    raw_image_cube_specs = sector.get("image_cube_specs", [])
    if not isinstance(raw_image_cube_specs, list):
        raise ValueError(f"sectors[{index}].image_cube_specs must be a list")
    image_cube_specs = []
    for spec_index, spec in enumerate(raw_image_cube_specs):
        if not isinstance(spec, Mapping):
            raise ValueError(f"sectors[{index}].image_cube_specs[{spec_index}] must be a mapping")
        image_cube_specs.append(_validate_image_cube_spec(spec, index, spec_index))

    validated_sector = dict(sector)
    validated_sector["prepare_tasks"] = prepare_tasks
    validated_sector["image_cube_specs"] = image_cube_specs
    validated_sector["wsclean_imsize"] = _validate_int_list(
        sector.get("wsclean_imsize"),
        f"sectors[{index}].wsclean_imsize",
    )
    validated_sector["dd_psf_grid"] = _validate_int_list(
        sector.get("dd_psf_grid"),
        f"sectors[{index}].dd_psf_grid",
    )
    validated_sector["obs_original_paths"] = _validate_str_list(
        sector.get("obs_original_paths"),
        f"sectors[{index}].obs_original_paths",
    )
    validated_sector["obs_starttime"] = _validate_str_list(
        sector.get("obs_starttime"),
        f"sectors[{index}].obs_starttime",
    )
    validated_sector["obs_ntimes"] = _validate_int_list(
        sector.get("obs_ntimes"),
        f"sectors[{index}].obs_ntimes",
    )
    return validated_sector


def validate_image_payload(payload: Mapping[str, object]) -> ImagePayload:
    supported_modes = {
        "facet_full_stokes",
        "facet_stokes_i",
        "no_dde_full_stokes",
        "no_dde_stokes_i",
        "screens_full_stokes",
        "screens_stokes_i",
    }
    mode = str(payload["mode"])
    if mode not in supported_modes:
        raise ValueError("Only no-DDE, facet, and screen image payloads are supported")
    pipeline_working_dir = str(payload["pipeline_working_dir"])
    raw_sectors = payload.get("sectors", [])
    if not isinstance(raw_sectors, list):
        raise ValueError("sectors must be a list")
    sectors = []
    for index, sector in enumerate(raw_sectors):
        if not isinstance(sector, Mapping):
            raise ValueError(f"sectors[{index}] must be a mapping")
        sectors.append(_validate_image_sector(sector, index))
    return {
        "mode": mode,
        "use_mpi": bool(payload.get("use_mpi", False)),
        "pipeline_working_dir": pipeline_working_dir,
        "sectors": sectors,
    }

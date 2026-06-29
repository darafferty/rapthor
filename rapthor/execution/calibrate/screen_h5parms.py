"""Helpers for collecting screen-solution h5parm files."""

import os
from typing import Sequence

import h5py


def collect_screen_h5parms(
    h5parm_files: Sequence[str],
    output_h5parm: str,
    overwrite: bool = False,
) -> None:
    """
    Collect screen-solution h5parms into one h5parm by concatenating in time.

    Input files must have the same group and dataset structure. Datasets named
    ``time``, ``val``, and ``weight`` are extended along their time axes as
    additional files are read.
    """
    if os.path.exists(output_h5parm):
        if overwrite:
            os.remove(output_h5parm)
        else:
            raise FileExistsError("The output H5parm file exists and overwrite=False")

    with h5py.File(output_h5parm, "w") as output_file:
        for h5parm_file in h5parm_files:
            with h5py.File(h5parm_file, "r") as input_file:
                input_file.visititems(_copy_or_append_item(output_file))


def parse_h5parm_file_list(h5parm_files: Sequence[str]) -> list[str]:
    """Parse positional h5parm arguments, including a single comma-separated value."""
    h5parm_files = list(h5parm_files)
    if len(h5parm_files) == 1 and (
        "," in h5parm_files[0] or ("[" in h5parm_files[0] and "]" in h5parm_files[0])
    ):
        return [filename.strip() for filename in h5parm_files[0].strip("[]").split(",")]
    return h5parm_files


def _copy_or_append_item(output_file):
    """Return an h5py visitor that copies new items and appends repeated datasets."""

    def visitor(item_name, item):
        item_basename = item_name.split("/")[-1]
        if item_name not in output_file:
            _copy_new_item(output_file, item_name, item, item_basename)
            return
        _append_existing_item(output_file, item_name, item, item_basename)

    return visitor


def _copy_new_item(output_file, item_name: str, item, item_basename: str) -> None:
    """Copy one h5py group or dataset into the output file."""
    if isinstance(item, h5py.Dataset):
        maxshape = item.maxshape
        if item_basename == "time":
            maxshape = (None,)
        elif item_basename in ["val", "weight"]:
            maxshape = (maxshape[0], maxshape[1], None, maxshape[3])
        dataset = output_file.create_dataset_like(
            item_name,
            item,
            chunks=item.shape if all(item.shape) else None,
            maxshape=maxshape,
        )
        if "AXES" in item.attrs:
            dataset.attrs["AXES"] = item.attrs["AXES"]
        dataset.resize(item.shape)
        dataset[:] = item[:]
        return

    if isinstance(item, h5py.Group):
        group = output_file.create_group(item_name)
        if "TITLE" in item.attrs:
            group.attrs["TITLE"] = item.attrs["TITLE"]
        group.attrs["h5parm_version"] = item.attrs.get("h5parm_version", 1.0)


def _append_existing_item(output_file, item_name: str, item, item_basename: str) -> None:
    """Append repeated time, value, and weight datasets along their time axes."""
    if item_basename == "time":
        new_shape = (output_file[item_name].shape[0] + item.shape[0],)
        output_file[item_name].resize(new_shape)
    elif item_basename in ["val", "weight"]:
        existing_shape = output_file[item_name].shape
        new_shape = (
            existing_shape[0],
            existing_shape[1],
            existing_shape[2] + item.shape[2],
            existing_shape[3],
        )
        output_file[item_name].resize(new_shape)

    if item_basename in ["time", "val", "weight"]:
        slicer = tuple(slice(-size, None) for size in item.shape)
        output_file[item_name][slicer] = item[:]

"""Helpers for aligning h5parm source tables with sky-model patches."""

import numpy as np
import lsmtool
from losoto.h5parm import h5parm
from lsmtool.operations_lib import normalize_ra_dec


def adjust_h5parm_source_coordinates(
    skymodel: str,
    h5parm_file: str,
    solset_name: str = "sol000",
) -> None:
    """Adjust h5parm source coordinates to match patch positions in a sky model."""
    source_positions = lsmtool.load(skymodel).getPatchPositions()
    with h5parm(h5parm_file, readonly=False) as parms:
        solset = parms.getSolset(solset_name)
        soltab = solset.getSoltabs()[0]
        direction_independent = not hasattr(soltab, "dir")

        if direction_independent:
            print(
                f"The solutions in solution set {solset_name} of the input h5parm file are "
                "direction-independent. The solutions will be duplicated for all directions "
            )
        elif len(source_positions) != len(soltab.dir):
            raise ValueError(
                "The patches in the sky model and the directions in the h5parm "
                "must have the same length"
            )

        source_names, source_dirs_rad = _source_table_rows(
            source_positions,
            soltab,
            direction_independent=direction_independent,
        )
        _replace_source_table(parms, solset, solset_name, source_names, source_dirs_rad)

        if direction_independent:
            _duplicate_solution_values_for_directions(solset, source_names)


def _source_table_rows(source_positions, soltab, *, direction_independent: bool):
    """Return h5parm source-table names and radian coordinates."""
    source_names = []
    source_dirs_deg = []
    if direction_independent:
        iterator = (
            (f"[{source_name}]", position) for source_name, position in source_positions.items()
        )
    else:
        iterator = (
            (source, _position_for_direction(source_positions, source)) for source in soltab.dir
        )

    for source_name, position in iterator:
        ra_deg, dec_deg = normalize_ra_dec(position[0].value, position[1].value)
        source_names.append(source_name)
        source_dirs_deg.append([ra_deg, dec_deg])

    source_dirs_deg = np.array(source_dirs_deg)
    ra_deg = source_dirs_deg.T[0]
    dec_deg = source_dirs_deg.T[1]
    source_dirs_rad = [
        [ra * np.pi / 180.0, dec * np.pi / 180.0] for ra, dec in zip(ra_deg, dec_deg)
    ]
    return source_names, source_dirs_rad


def _position_for_direction(source_positions, source: str):
    """Look up one h5parm direction in the sky-model patch positions."""
    try:
        return source_positions[source.strip("[]")]
    except KeyError as err:
        raise ValueError(
            "A direction is present in the h5parm that is not in the sky model"
        ) from err


def _replace_source_table(parms, solset, solset_name: str, source_names, source_dirs_rad) -> None:
    """Replace the h5parm source table with the adjusted source rows."""
    source_table = solset.obj._f_get_child("source")
    source_table._f_remove(recursive=True)
    descriptor = np.dtype([("name", np.str_, 128), ("dir", np.float32, 2)])
    solset_node = parms.H.get_node("/", solset_name)
    parms.H.create_table(
        solset_node,
        "source",
        descriptor,
        title="Source names and directions",
        expectedrows=25,
    )

    source_table = solset.obj._f_get_child("source")
    source_table.append(list(zip(*(source_names, source_dirs_rad))))


def _duplicate_solution_values_for_directions(solset, source_names) -> None:
    """Duplicate direction-independent soltab values across all source directions."""
    for soltab in solset.getSoltabs():
        values, _ = soltab.getValues()
        weights, _ = soltab.getValues(weight=True)
        new_shape = [*values.shape, len(source_names)]
        new_values = np.zeros(new_shape)
        new_weights = np.zeros(new_shape)
        for direction_index in range(len(source_names)):
            new_values[:, direction_index] = values
            new_weights[:, direction_index] = weights

        soltab_name = soltab.name
        soltab_type = soltab.getType()
        axes_names = soltab.getAxesNames()
        axes_names.append("dir")
        axes_values = [soltab.getAxisValues(axis_name) for axis_name in soltab.getAxesNames()]
        axes_values.append(source_names)
        soltab.delete()
        solset.makeSoltab(
            soltab_type,
            soltab_name,
            axesNames=axes_names,
            axesVals=axes_values,
            vals=new_values,
            weights=new_weights,
        )

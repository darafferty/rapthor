"""WSClean model rendering helpers for mosaic products."""

import os
import re
import shutil
from pathlib import Path
from typing import Sequence

import lsmtool
import numpy as np
from astropy.io import fits as pyfits
from astropy.table import vstack
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from lsmtool.utils import format_coordinates

from rapthor.execution.config import ExecutionConfig
from rapthor.execution.mosaic.commands import (
    DrawModelMosaicOptions,
    build_draw_model_mosaic_command,
)
from rapthor.execution.outputs import require_file
from rapthor.execution.shell import run_external_command

_FITS_SUFFIX = re.compile(r"\.fits$")
_DEFAULT_MODEL_BANDWIDTH_HZ = 1.0e6


def render_model_mosaic_with_wsclean(
    sector_skymodels: Sequence[str],
    template_image: str,
    output_image: str,
    pipeline_working_dir: str,
    execution_config: ExecutionConfig,
    shell_operation_cls=None,
) -> dict:
    """Render one model mosaic from sector sky-model component lists."""
    combined_skymodel = _combined_skymodel_path(output_image)
    combine_sector_skymodels(sector_skymodels, combined_skymodel)
    output_root = _draw_model_output_root(output_image)
    options = _draw_model_options(
        combined_skymodel,
        output_root,
        template_image,
        num_threads=execution_config.local_dask_threads_per_worker,
    )
    run_external_command(
        build_draw_model_mosaic_command(options),
        pipeline_working_dir,
        execution_config,
        shell_operation_cls=shell_operation_cls,
    )
    _expose_wsclean_term_model(output_root, output_image)
    return require_file(output_image, "Mosaic output")


def combine_sector_skymodels(sector_skymodels: Sequence[str], output_skymodel: str) -> dict:
    """Combine sector sky models, suffixing names to keep components unique."""
    combined = None
    for index, skymodel_path in enumerate(sector_skymodels, start=1):
        skymodel = lsmtool.load(str(skymodel_path))
        if len(skymodel) == 0:
            continue
        _suffix_skymodel_names(skymodel, f"sector_{index}")
        if combined is None:
            combined = skymodel
        else:
            combined.table = vstack(
                [combined.table.filled(), skymodel.table.filled()],
                metadata_conflicts="silent",
            )

    if combined is None:
        raise ValueError("Cannot render a model mosaic from empty sector sky models")
    if combined.hasPatches:
        combined._updateGroups()
        combined.setPatchPositions(method="wmean")
    combined.write(output_skymodel, clobber=True)
    return require_file(output_skymodel, "Mosaic model sky model")


def _draw_model_options(
    skymodel: str,
    output_root: str,
    template_image: str,
    *,
    num_threads: int,
) -> DrawModelMosaicOptions:
    with pyfits.open(template_image) as hdul:
        header = hdul[0].header.copy()
        shape = hdul[0].data.shape

    wcs = WCS(header).celestial
    ysize, xsize = shape[-2:]
    ra_deg, dec_deg = wcs.pixel_to_world_values((xsize - 1) / 2.0, (ysize - 1) / 2.0)
    pixel_scales = proj_plane_pixel_scales(wcs)
    frequency_hz = _template_frequency_hz(header, skymodel)
    return DrawModelMosaicOptions(
        skymodel=skymodel,
        output_root=output_root,
        ra_dec=list(format_coordinates(float(ra_deg), float(dec_deg))),
        frequency_bandwidth=[frequency_hz, _DEFAULT_MODEL_BANDWIDTH_HZ],
        cellsize_deg=float(abs(pixel_scales[0])),
        imsize=[int(xsize), int(ysize)],
        num_threads=max(1, int(num_threads)),
    )


def _suffix_skymodel_names(skymodel, suffix: str) -> None:
    """Suffix source and patch names from one sector before concatenation."""
    column_names = skymodel.getColNames()
    if skymodel.hasPatches and "Patch" in column_names:
        patch_names = skymodel.getColValues("Patch")
        skymodel.setColValues("Patch", np.array([f"{name}_{suffix}" for name in patch_names]))
    if "Name" in column_names:
        source_names = skymodel.getColValues("Name")
        skymodel.setColValues("Name", np.array([f"{name}_{suffix}" for name in source_names]))


def _template_frequency_hz(header: pyfits.Header, skymodel: str) -> float:
    for key in ("FREQ", "RESTFREQ", "CRVAL3"):
        if key in header:
            return float(header[key])
    return _skymodel_reference_frequency_hz(skymodel)


def _skymodel_reference_frequency_hz(skymodel: str) -> float:
    model = lsmtool.load(str(skymodel))
    if "ReferenceFrequency" in model.getColNames():
        frequencies = np.asarray(model.getColValues("ReferenceFrequency"), dtype=float)
        frequencies = frequencies[np.isfinite(frequencies)]
        if frequencies.size:
            return float(np.median(frequencies))
    if "ReferenceFrequency" in model.table.meta:
        return float(model.table.meta["ReferenceFrequency"])
    raise ValueError(f"Could not determine reference frequency for {skymodel}")


def _combined_skymodel_path(output_image: str) -> str:
    return f"{_draw_model_output_root(output_image)}.skymodel"


def _draw_model_output_root(output_image: str) -> str:
    return _FITS_SUFFIX.sub("", output_image)


def _expose_wsclean_term_model(output_root: str, output_image: str) -> None:
    root = Path(output_root)
    term_model = root.with_name(f"{root.name}-term-0.fits")
    require_file(str(term_model), "WSClean mosaic model image")

    output_path = Path(output_image)
    if output_path.exists() or output_path.is_symlink():
        output_path.unlink()
    try:
        output_path.symlink_to(os.path.relpath(term_model, output_path.parent))
    except OSError:
        shutil.copyfile(term_model, output_path)
    require_file(output_image, "Mosaic output")

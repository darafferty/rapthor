"""Download the sky models that a run without internet access needs."""

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
from lsmtool.download_skymodel import download_skymodel
from lsmtool.operations_lib import normalize_ra_dec

from rapthor.execution.image.flux_normalization import SURVEY_METADATA, get_field_phase_center

log = logging.getLogger("rapthor:catalog_preparation")

DEFAULT_RADIUS_DEG = 5.0
PHOTOMETRY_SURVEY = "TGSS"
ASTROMETRY_SURVEY = "PANSTARRS"
# The frequency of each survey in SURVEY_METADATA under its LSMTool name, ordered by frequency
# as the normalization uses them
NORMALIZATION_SURVEY_FREQUENCIES = {
    survey.upper(): float(SURVEY_METADATA[survey]["frequency"])
    for survey in sorted(SURVEY_METADATA, key=lambda name: SURVEY_METADATA[name]["frequency"])
}


@dataclass(frozen=True)
class PreparedCatalogs:
    """Paths of the downloaded sky models, and the surveys that could not be downloaded."""

    paths: dict[str, str]
    failures: dict[str, str]


def prepare_catalogs(
    ms_file: Union[str, Path],
    output_dir: Union[str, Path],
    radius_deg: float = DEFAULT_RADIUS_DEG,
    overwrite: bool = False,
) -> PreparedCatalogs:
    """
    Download the sky models around the pointing of a Measurement Set.

    A sky model that is already present is kept unless ``overwrite`` is set. A failure does
    not stop the other downloads, so that a rerun only has to get what is missing.

    Parameters
    ----------
    ms_file : str or Path
        Measurement Set whose pointing sets the center of the downloads
    output_dir : str or Path
        Directory for the sky models, which is created if needed
    radius_deg : float, optional
        Radius of the downloads in degrees
    overwrite : bool, optional
        Replace sky models that are already present

    Returns
    -------
    catalogs : PreparedCatalogs
        The sky model path of each survey that is available and the error of each that is not
    """
    if radius_deg <= 0:
        raise ValueError(f"The download radius must be positive, not {radius_deg}.")
    if not Path(ms_file).is_dir():
        raise FileNotFoundError(f'Measurement Set not found at "{ms_file}"')
    ra_rad, dec_rad = get_field_phase_center(str(ms_file))
    ra_deg, dec_deg = normalize_ra_dec(np.degrees(ra_rad), np.degrees(dec_rad))
    cone_params = {"ra": ra_deg, "dec": dec_deg, "radius": radius_deg}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    failures = {}
    for survey in (PHOTOMETRY_SURVEY, ASTROMETRY_SURVEY, *NORMALIZATION_SURVEY_FREQUENCIES):
        path = output_dir / f"{survey}_model.txt"
        if path.exists() and not overwrite:
            log.info("Keeping the %s sky model that is already at %s", survey, path)
            paths[survey] = str(path)
            continue
        try:
            _download_catalog(cone_params, survey, path)
        # What a failed download raises depends on the server of the survey
        except Exception as error:
            log.error("The %s sky model could not be downloaded: %s", survey, error)
            failures[survey] = str(error)
        else:
            paths[survey] = str(path)
    return PreparedCatalogs(paths=paths, failures=failures)


def _download_catalog(cone_params: dict, survey: str, path: Path) -> None:
    """Download a sky model so that its final path only ever holds a complete file."""
    log.info("Downloading the %s sky model to %s", survey, path)
    # A directory of its own keeps this apart from any other run on the same output directory
    with tempfile.TemporaryDirectory(dir=path.parent, prefix=".download_") as temp_dir:
        partial_path = Path(temp_dir) / path.name
        download_skymodel(
            cone_params,
            skymodel_path=str(partial_path),
            overwrite=True,
            survey=survey,
        )
        partial_path.replace(path)


def parset_options(catalogs: PreparedCatalogs) -> str:
    """Return the parset options that use a complete set of downloaded sky models."""
    normalization_paths = [catalogs.paths[survey] for survey in NORMALIZATION_SURVEY_FREQUENCIES]
    return "\n".join(
        [
            "[imaging]",
            f"photometry_skymodel = {catalogs.paths[PHOTOMETRY_SURVEY]}",
            f"astrometry_skymodel = {catalogs.paths[ASTROMETRY_SURVEY]}",
            f"normalization_skymodels = [{', '.join(normalization_paths)}]",
            f"normalization_reference_frequencies = {list(NORMALIZATION_SURVEY_FREQUENCIES.values())}",
            "",
            "[cluster]",
            "allow_internet_access = False",
        ]
    )

"""Download the sky models that a run without internet access needs."""

import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Optional

import numpy as np
from lsmtool.download_skymodel import download_skymodel
from lsmtool.operations_lib import normalize_ra_dec

from rapthor.execution.image.flux_normalization import SURVEY_METADATA, get_field_phase_center

log = logging.getLogger("rapthor:catalog_preparation")

DEFAULT_RADIUS_DEG = 5.0
PHOTOMETRY_SURVEY = "TGSS"
ASTROMETRY_SURVEY = "PANSTARRS"
# The keys of SURVEY_METADATA, ordered by frequency as the normalization uses them
NORMALIZATION_SURVEYS = tuple(
    sorted(SURVEY_METADATA, key=lambda survey: SURVEY_METADATA[survey]["frequency"])
)


@dataclass(frozen=True)
class PreparedCatalogs:
    """Paths of the downloaded sky models, and the surveys that could not be downloaded."""

    paths: dict[str, str]
    failures: dict[str, str]


def catalog_path(output_dir: str, survey: str) -> str:
    """Return the path of the sky model of a survey."""
    return os.path.join(output_dir, f"{survey}_model.txt")


def prepare_catalogs(
    ms_file: str,
    output_dir: str,
    radius_deg: float = DEFAULT_RADIUS_DEG,
    overwrite: bool = False,
) -> PreparedCatalogs:
    """
    Download the sky models around the pointing of a Measurement Set.

    A sky model that is already present is kept unless ``overwrite`` is set. A failure does
    not stop the other downloads, so that a rerun only has to get what is missing.

    Parameters
    ----------
    ms_file : str
        Measurement Set whose pointing sets the center of the downloads
    output_dir : str
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
    if not os.path.isdir(ms_file):
        raise FileNotFoundError(f'Measurement Set not found at "{ms_file}"')
    ra_rad, dec_rad = get_field_phase_center(ms_file)
    ra_deg, dec_deg = normalize_ra_dec(np.degrees(ra_rad), np.degrees(dec_rad))
    cone_params = {"ra": ra_deg, "dec": dec_deg, "radius": radius_deg}

    os.makedirs(output_dir, exist_ok=True)
    paths = {}
    failures = {}
    surveys = (PHOTOMETRY_SURVEY, ASTROMETRY_SURVEY, *map(str.upper, NORMALIZATION_SURVEYS))
    for survey in surveys:
        path = catalog_path(output_dir, survey)
        if os.path.exists(path) and not overwrite:
            log.info("Keeping the %s sky model that is already at %s", survey, path)
            paths[survey] = path
            continue
        error = _download_catalog(cone_params, survey, path)
        if error is None:
            paths[survey] = path
        else:
            failures[survey] = error
    return PreparedCatalogs(paths=paths, failures=failures)


def _download_catalog(cone_params: dict, survey: str, path: str) -> Optional[str]:
    """Download a sky model so that its final path only ever holds a complete file."""
    log.info("Downloading the %s sky model to %s", survey, path)
    # A directory of its own keeps this apart from any other run on the same output directory
    with tempfile.TemporaryDirectory(dir=os.path.dirname(path), prefix=".download_") as temp_dir:
        partial_path = os.path.join(temp_dir, os.path.basename(path))
        try:
            download_skymodel(
                cone_params,
                skymodel_path=partial_path,
                overwrite=True,
                survey=survey,
            )
        # What a failed download raises depends on the server of the survey
        except Exception as error:
            log.error("The %s sky model could not be downloaded: %s", survey, error)
            return str(error)
        os.replace(partial_path, path)
    return None


def parset_options(catalogs: PreparedCatalogs) -> str:
    """Return the parset options that use a complete set of downloaded sky models."""
    normalization_paths = [catalogs.paths[survey.upper()] for survey in NORMALIZATION_SURVEYS]
    normalization_frequencies = [
        float(SURVEY_METADATA[survey]["frequency"]) for survey in NORMALIZATION_SURVEYS
    ]
    return "\n".join(
        [
            "[imaging]",
            f"photometry_skymodel = {catalogs.paths[PHOTOMETRY_SURVEY]}",
            f"astrometry_skymodel = {catalogs.paths[ASTROMETRY_SURVEY]}",
            f"normalization_skymodels = [{', '.join(normalization_paths)}]",
            f"normalization_reference_frequencies = {normalization_frequencies}",
            "",
            "[cluster]",
            "allow_internet_access = False",
        ]
    )

#!/usr/bin/env python3
"""
Script to download a sky model using LSMTool
"""
from argparse import ArgumentParser, RawTextHelpFormatter

import casacore.tables as ct
from lsmtool.download_skymodel import download_skymodel
from lsmtool.operations_lib import normalize_ra_dec
import numpy as np


def download_model(msin, survey, radius, skymodel):
    """
    Download a sky model using LSMTool

    Parameters
    ----------
    msin : str
        Filename of input MS file (used to determine pointing info)
    survey : str
        Name of the survey to query (must be one supported by LSMTool's
        download_skymodel() function)
    radius : float
        Query radius in degrees
    skymodel : str
        Filename of the output sky model
    """
    with ct.table(msin + "::FIELD", ack=False) as obs:
        ra, dec = normalize_ra_dec(
            np.degrees(float(obs.col("REFERENCE_DIR")[0][0][0])),
            np.degrees(float(obs.col("REFERENCE_DIR")[0][0][1])),
        )

    download_skymodel(
        {"ra": ra, "dec": dec, "radius": radius},
        survey=survey,
        skymodel_path=skymodel,
        overwrite=True,
    )


if __name__ == "__main__":
    """
    Download a sky model using LSMTool.

    """
    descriptiontext = "Download a sky model.\n"
    parser = ArgumentParser(description=descriptiontext, formatter_class=RawTextHelpFormatter)
    parser.add_argument("msin", help="Input Measurement Set", type=str)
    parser.add_argument("survey", help="Survey to query", type=str)
    parser.add_argument("radius", help="Query radius in degrees", type=float)
    parser.add_argument("skymodel", help="Output sky model filename", type=str)
    args = parser.parse_args()
    download_model(args.msin, args.survey, args.radius, args.skymodel)

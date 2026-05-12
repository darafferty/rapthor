#!/usr/bin/env python3
"""
Script to predict using wsclean
"""
from argparse import ArgumentParser, RawTextHelpFormatter
import os
import shutil
import subprocess
import sys
from pathlib import Path
import logging
import shutil

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


import casacore.tables as pt
import numpy as np


def predict(msfile, ds9_region_file, model_image):
    """

    Parameters
    ----------
    msfile : MS name, output will be written to separate columns
    ds9_region_file: DS9 region file, specifying facet regions and names
    model_image: FITS image to use as model

    Returns
    -------
    int : 0 if successfull; non-zero otherwise

    Raises
    ------
    ValueError
        If no valid Measurement Set, DS9 region file or model
        image exists
    """
    # Check pre-conditions
    if not os.path.exists(msfile):
        raise ValueError(f"Input measurement set {msfile!r} does not exist")
    if not os.path.exists(ds9_region_file):
        raise ValueError(f"DS9 region file {ds9_region_file!r} does not exist")
    if not os.path.exists(model_image):
        raise ValueError(f"Model image {model_image!r} does not exist")

    # Run the command
    try:
        return subprocess.run(cmd, check=True).returncode
    except subprocess.CalledProcessError as err:
        print(err, file=sys.stderr)
        return err.returncode


def main():
    """ """
    descriptiontext = "Predict model data using WSClean.\n"
    parser = ArgumentParser(
        description=descriptiontext, formatter_class=RawTextHelpFormatter
    )
    parser.add_argument(
        "--msin", help="Input/Output measurement set", type=str, default=""
    )
    parser.add_argument("--region", help="DS9 region file", type=str, default="")
    parser.add_argument("--model", help="Model FITS image", type=str, default="")
    args = parser.parse_args()
    return predict(args.msin, args.region, args.model)


if __name__ == "__main__":
    sys.exit(main())

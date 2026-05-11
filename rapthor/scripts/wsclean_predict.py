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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


import casacore.tables as pt
import numpy as np


def predict(msfile, ds9_region_file, model_image):
    """

    Parameters
    ----------
    msfile : MS name
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
        raise ValueError(f"Input Measurement Set {msfile!r} does not exist")

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
        "--msin", help="Input/Output Measurement Set", type=str, default=""
    )
    args = parser.parse_args()


if __name__ == "__main__":
    sys.exit(main())

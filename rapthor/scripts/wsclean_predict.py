#!/usr/bin/env python3
"""
Script to predict using wsclean
"""
from argparse import ArgumentParser, RawTextHelpFormatter
import os
import subprocess
import sys
from pathlib import Path
import logging
from lsmtool.facet import read_ds9_region_file

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


import casacore.tables as ct
import numpy as np


def add_columns_to_ms(msname, colnames):
    """
    Add additional columns to table
    Do this in one call, rather than using taql and doing it in serial
    """
    tt = ct.table(msname, readonly=False)
    cl = tt.getcol("DATA")
    (nrows, nchans, npols) = cl.shape
    vl = np.zeros(shape=cl.shape, dtype=cl.dtype)
    dmi = tt.getdminfo("DATA")
    for colname in colnames:
        dmi["NAME"] = colname
        mkd = ct.maketabdesc(
            ct.makearrcoldesc(
                colname,
                shape=np.array(np.zeros([nchans, npols])).shape,
                valuetype="complex",
                value=0.0,
            )
        )
        tt.addcols(mkd, dmi)
        tt.putcol(colname, vl)
    tt.close()


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

    # remove '-model.fits' from image name
    model = model_image.replace("-model.fits", "")
    print(model)

    # extract region names
    facets = read_ds9_region_file(ds9_region_file)
    facet_names = list()
    for facet in facets:
        facet_names.append(facet.name)

    add_columns_to_ms(msfile, facet_names)

    err_code = 0
    # Run the command
    for facet in facet_names:
        cmd = [
            "wsclean",
            "-predict",
            "-facet-regions",
            str(ds9_region_file),
            "-model-column",
            str(facet),
            "-select-facets",
            "{" + str(facet) + "}",
            "-name",
            str(model),
            str(msfile),
        ]
        try:
            subprocess.run(cmd, check=True).returncode
        except subprocess.CalledProcessError as err:
            print(err, file=sys.stderr)
            err_code = err.returncode
            break

    return err_code


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

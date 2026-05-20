#!/usr/bin/env python3
"""
Script to predict using wsclean
"""

import argparse
from argparse import ArgumentParser, RawTextHelpFormatter
import os
import json
import subprocess
import sys
import logging
import casacore.tables as ct
from lsmtool.facet import read_ds9_region_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def remove_columns_from_ms(msfile, ds9_region_file):
    """
    Remove extra columns added from prediction

    Parameters
    ----------
    msfile : MS name, output will be written to separate columns
    ds9_region_file: DS9 region file, specifying facet regions and names
    """
    tt = ct.table(msfile, readonly=False)
    colnames = tt.colnames()

    facets = read_ds9_region_file(ds9_region_file)
    for facet in facets:
        colname = facet.name
        if colname in colnames:
            tt.removecols(colname)
    tt.close()


def predict(msfile, ds9_region_file, model_image, storage_manager):
    """
    Predict model image to msfile

    Parameters
    ----------
    msfile : MS name, output will be written to separate columns
    ds9_region_file: DS9 region file, specifying facet regions and names
    Note: names in region file should be with {}, like {Patch_1}, After
    parsing the {} will be dropped
    model_image: FITS image to use as model
    """

    # remove '-model.fits' from image name
    model = model_image.replace("-model.fits", "")

    # extract region names
    facets = read_ds9_region_file(ds9_region_file)
    facet_names = list()
    for facet in facets:
        facet_names.append(facet.name)

    # TODO: If the msfile is read-only, copy the file to a temporary ms
    # for example : copy xxx.ms to xxx-predict-tmpfile.ms
    # and predict the data

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
            "-model-storage-manager",
            str(storage_manager),
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
    """
    Script can be used in two ways:
    1) To add columns and predict model
      wsclean_predict.py --region sector_1_facets_ds9.reg --msin small.ms --model images/field-MFS-model.fits
    2) To remove extra columns created from step 1
      wsclean_predict.py --region sector_1_facets_ds9.reg --msin small.ms --cleanup

    Returns
    -------
    int : 0 if successfull; non-zero otherwise

    Raises
    ------
    ValueError
        If no valid Measurement Set, DS9 region file or model
        image exists
    """
    descriptiontext = "Predict model data using WSClean.\n"
    parser = ArgumentParser(description=descriptiontext, formatter_class=RawTextHelpFormatter)
    parser.add_argument("--msin", help="Input/Output measurement set", type=str, default="")
    parser.add_argument("--region", help="DS9 region file", type=str, default="")
    parser.add_argument("--model", help="Model FITS image", type=str, default="")
    parser.add_argument("--storage_manager", help="Storage manager", type=str, default="default")
    parser.add_argument(
        "--cleanup", action=argparse.BooleanOptionalAction, help="Remove exra columns"
    )
    args = parser.parse_args()

    # Check pre-conditions
    if not os.path.exists(args.msin):
        raise ValueError(f"Input measurement set {args.msin!r} does not exist")
    if not os.path.exists(args.region):
        raise ValueError(f"DS9 region file {args.region!r} does not exist")
    if not args.cleanup:
        if not os.path.exists(args.model):
            raise ValueError(f"Model image {args.model!r} does not exist")

    # if msin is read only, create a copy of msin to work with,
    # return this as output
    out_dict = {"msout": args.msin}
    out_file = f"{args.msin}.wsclean_predict.json"
    with open(out_file, "w") as fp:
        json.dump(out_dict, fp)

    if args.cleanup:
        return remove_columns_from_ms(args.msin, args.region)
    else:
        return predict(args.msin, args.region, args.model, args.storage_manager)


if __name__ == "__main__":
    sys.exit(main())

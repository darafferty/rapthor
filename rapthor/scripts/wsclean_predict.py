#!/usr/bin/env python3
"""
Script to predict using wsclean
"""

import argparse
from argparse import ArgumentParser, RawTextHelpFormatter
import os
import stat
import shutil
import uuid
import json
import subprocess
import sys
import logging
import casacore.tables as ct
from lsmtool.facet import read_ds9_region_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def make_writable(msfile):
    """
    Check if msfile is writable, if not, create a writable copy
    and return it as output
    """

    if os.access(msfile, os.W_OK):
        return msfile
    # get base dir to create files
    tmpdir = "$(runtime.tmpdir)"
    newms = os.path.join(tmpdir, os.path.basename(msfile) + "_" + str(uuid.uuid4()))
    # copy msfile to newms
    shutil.copytree(msfile, newms, dirs_exist_ok=True)
    # change root dir +rwx
    current_mode = os.stat(newms).st_mode
    os.chmod(
        newms,
        current_mode | stat.S_IWUSR | stat.S_IWGRP | stat.S_IXUSR | stat.S_IWOTH | stat.S_IXOTH,
    )

    # Walk through the directory tree
    for root, dirs, files in os.walk(newms):
        # Update permissions for subdirectories
        for d in dirs:
            dir_p = os.path.join(root, d)
            current_mode = os.stat(dir_p).st_mode
            # +rwx
            os.chmod(dir_p, current_mode | stat.S_IWUSR | stat.S_IWGRP | stat.S_IXUSR)
        # Update permissions for files
        for f in files:
            file_p = os.path.join(root, f)
            current_mode = os.stat(file_p).st_mode
            # +rw
            os.chmod(file_p, current_mode | stat.S_IWUSR | stat.S_IWGRP)

    return newms


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


def predict(msfiles, ds9_region_file, model_images, storage_manager):
    """
    Predict model image to msfile

    Parameters
    ----------
    msfiles : List: MS names, output will be written to separate columns
    ds9_region_file: DS9 region file, specifying facet regions and names
    Note: names in region file should be with {}, like {Patch_1}, After
    parsing the {} will be dropped
    model_images: List: FITS images to use as model
    """
    # work with only first model image (for now)
    # model images can have arbitrary names,
    # make a symlink in same dir with workable name
    model_image = model_images[0]
    tmpdir = os.path.dirname(model_image)
    model_image = tmpdir + "/predict-model.fits"
    os.symlink(model_images[0], model_image)
    # remove '-model.fits' from image name
    model = model_image.replace("-model.fits", "")

    # extract region names
    facets = read_ds9_region_file(ds9_region_file)
    facet_names = list()
    for facet in facets:
        facet_names.append(facet.name)

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
            str(facet),
            "-name",
            str(model),
            "-model-storage-manager",
            str(storage_manager),
            *[str(msfilename) for msfilename in msfiles],
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
    parser.add_argument(
        "--msin", help="Input/Output measurement set", nargs="+", type=str, default=[]
    )
    parser.add_argument("--region", help="DS9 region file", type=str, default="")
    parser.add_argument("--model", help="Model FITS image", nargs="+", type=str, default=[])
    parser.add_argument("--storage_manager", help="Storage manager", type=str, default="default")
    parser.add_argument(
        "--cleanup", action=argparse.BooleanOptionalAction, help="Remove exra columns"
    )
    args = parser.parse_args()
    # Note: the output file name should match file read in CWL step
    output_info = "msout_names.json"
    # Check pre-conditions
    if not (len(args.msin) > 0 and os.path.exists(args.msin[0])):
        raise ValueError(f"Input measurement set {args.msin!r} does not exist")
    if not os.path.exists(args.region):
        raise ValueError(f"DS9 region file {args.region!r} does not exist")
    if not args.cleanup:
        if not (len(args.model) > 0 and os.path.exists(args.model[0])):
            raise ValueError(f"Model image {args.model!r} does not exist")

    # if msin is read only, create a copy of msin to work with,
    # return this as output
    msnames = list()
    for msname in args.msin:
        msnames.append(make_writable(msname))

    facets = read_ds9_region_file(args.region)
    facet_names = "["
    start_facet = True
    for facet in facets:
        if start_facet:
            facet_names += facet.name
            start_facet = False
        else:
            facet_names += "," + facet.name
    facet_names += "]"

    out_dict = {"msout": msnames, "patches": facet_names}

    with open(output_info, "w") as f:
        json.dump(out_dict, f)

    if args.cleanup:
        return remove_columns_from_ms(msnames, args.region)
    else:
        return predict(msnames, args.region, args.model, args.storage_manager)


if __name__ == "__main__":
    sys.exit(main())

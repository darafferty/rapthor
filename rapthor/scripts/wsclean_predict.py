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
import numpy as np
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


def predict(
    msfiles,
    ds9_region_file,
    sky_model,
    ra_dec,
    frequency_bandwidth,
    imsize,
    cellsize_deg,
    storage_manager,
):
    """
    Predict model image to msfile

    Parameters
    ----------
    msfiles : List: MS names, output will be written to separate columns
    ds9_region_file: DS9 region file, specifying facet regions and names
    Note: names in region file should be with {}, like {Patch_1}, After
    parsing the {} will be dropped
    sky_model: input sky model
    ra_dec, frequency_bandwidth, imsize, cellsize_deg:
    parameters used for drawing the model image
    """
    # get channel frequencies
    freq_list = list()
    for ms in msfiles:
        chan_freqs = ct.table(ms + "::SPECTRAL_WINDOW").getcol("CHAN_FREQ")
        freq_list.append(chan_freqs[0])
    # get sorted, unique frequencies
    freq_list = np.sort(np.unique(np.concat(freq_list)))
    n_chan = freq_list.size
    if n_chan > 1:
        bandwidth = freq_list[-1] - freq_list[0] + (freq_list[1] - freq_list[0])
    else:
        bandwidth = frequency_bandwidth[1]
    # split channels into chunks of 2 MHz bandwidth
    predict_bandwidth = 2.0e6
    n_chunks = int(bandwidth / predict_bandwidth)
    chan_list = np.arange(n_chan)
    freq_chunks = np.array_split(freq_list, n_chunks)
    chan_chunks = np.array_split(chan_list, n_chunks)

    # extract region names
    facets = read_ds9_region_file(ds9_region_file)
    facet_names = list()
    for facet in facets:
        facet_names.append(facet.name)

    # model images can have arbitrary names,
    tmpdir = os.path.dirname(sky_model)
    model_name = os.path.join(tmpdir, "predict")
    model = model_name + "-model.fits"
    # make a symlink in same dir with workable name
    os.symlink(model_name + "-term-0.fits", model)

    for freqs, chans in zip(freq_chunks, chan_chunks):
        chunk_bandwidth = freqs[-1] - freqs[0] + (freq_list[1] - freq_list[0])
        chunk_freq = np.mean(freqs)
        err_code = 0
        # only one spectral term is created
        # output will be $(model_name)-term-0.fits
        cmd = [
            "wsclean",
            "-draw-model",
            str(sky_model),
            "-draw-spectral-terms",
            str(1),
            "-name",
            str(model_name),
            "-draw-centre",
            str(ra_dec[0]),
            str(ra_dec[1]),
            "-draw-frequencies",
            str(chunk_freq),
            str(chunk_bandwidth),
            "-size",
            str(imsize[0]),
            str(imsize[1]),
            "-scale",
            str(cellsize_deg),
        ]
        try:
            subprocess.run(cmd, check=True).returncode
        except subprocess.CalledProcessError as err:
            print(err, file=sys.stderr)
            err_code = err.returncode

        err_code = 0
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
                str(model_name),
                "-channel-range",
                str(chans[0]),
                str(chans[-1]),
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
      wsclean_predict.py --region sector_1_facets_ds9.reg --msin small.ms

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
    parser.add_argument("--skymodel", help="Sky model", type=str, default="")
    parser.add_argument(
        "--ra_dec",
        help="RA and Dec coordinates of model image center",
        type=str,
        nargs=2,
        default=[],
    )
    parser.add_argument(
        "--frequency_bandwidth",
        help="Center frequency and full bandwdith",
        type=str,
        nargs=2,
        default=[],
    )
    parser.add_argument("--cellsize", help="Model image cell size (deg)", type=float, default=1)
    parser.add_argument(
        "--imsize", help="Model image size n_x x n_y (pixels)", type=int, nargs=2, default=[]
    )
    parser.add_argument("--threads", help="Max threads to use", type=int, default=1)
    parser.add_argument(
        "--time_freq_smearing",
        help="Enable time frequency smearing",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument("--storage_manager", help="Storage manager", type=str, default="default")
    args = parser.parse_args()
    # Note: the output file name should match file read in CWL step
    output_info = "msout_names.json"
    # Check pre-conditions
    if not (len(args.msin) > 0 and os.path.exists(args.msin[0])):
        raise ValueError(f"Input measurement set {args.msin!r} does not exist")
    if not os.path.exists(args.region):
        raise ValueError(f"DS9 region file {args.region!r} does not exist")
    if not os.path.exists(args.skymodel):
        raise ValueError(f"Sky model file {args.skymodel!r} does not exist")
    if len(args.ra_dec) != 2:
        raise ValueError(f"Invalid RA Dec {args.ra_dec!r}")
    if len(args.frequency_bandwidth) != 2:
        raise ValueError(f"Invalid frequency and bandwidth {args.frequency_bandwidth!r}")
    if len(args.imsize) != 2:
        raise ValueError(f"Invalid image size {args.imsize!r}")

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

    # draw model and predict
    return predict(
        msnames,
        args.region,
        args.skymodel,
        args.ra_dec,
        args.frequency_bandwidth,
        args.imsize,
        args.cellsize,
        args.storage_manager,
    )


if __name__ == "__main__":
    sys.exit(main())

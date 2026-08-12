#!/usr/bin/env python3
"""
Script to predict using wsclean
"""

import argparse
import json
import logging
import os
import shutil
import stat
import subprocess
import sys
import uuid
from argparse import ArgumentParser, RawTextHelpFormatter

import casacore.tables as ct
import numpy as np
import math
import lsmtool
from lsmtool.facet import read_ds9_region_file
from scipy import constants

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def make_writable(msfile):
    """
    Always create a writable copy
    and return it as output
    """

    # get base dir to create files
    tmpdir = "$(runtime.tmpdir)"
    newms = os.path.join(tmpdir, os.path.basename(msfile) + "_" + str(uuid.uuid4()))
    # copy msfile to newms (TBD: use DP3 to average data if needed)
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


def optimal_imaging_parameters(msname, skymodel, ra, dec):
    """
    Return optimal image parameters (n_pixels, cell_size)
    for sky model rendering
    see sec. sky model resampling on wsclean manual
    """

    # window size used in -sinc-window-size
    window_size = 127
    # beta
    beta = 100

    # determine max value of (l,m) in the sky model (radians)
    sky = lsmtool.load(skymodel)
    distances = sky.getDistance(ra, dec)
    max_dist = np.argmax(distances)
    print(distances)
    print(distances.shape)
    print(f"max distance {max_dist} {distances[max_dist]}")
    # Sin projection to get l,m distance
    max_lm = math.sin(math.radians(distances[max_dist]))
    print(f"lmax {max_lm} rad")
    d0 = max_lm

    # determine max values of u,v,w in the data (m)
    uvw = ct.table(msname).getcol("UVW")
    print(uvw.shape)
    max_u = np.argmax(uvw[:, 0])
    max_v = np.argmax(uvw[:, 1])
    max_w = np.argmax(uvw[:, 2])
    print(f"uvw max {uvw[max_u, 0]} {uvw[max_v, 1]} {uvw[max_w, 2]}")
    max_uv = max(uvw[max_u, 0], uvw[max_v, 1])
    max_w = uvw[max_w, 2]

    # determine the max frequency Hz (done also in predict, could be combined)
    freq = ct.table(msname + "::SPECTRAL_WINDOW").getcol("CHAN_FREQ")
    print(freq.shape)
    max_f = np.argmax(freq[0, :])
    print(f"freq max {freq[0, max_f]}")
    max_freq = freq[0, max_f]

    # convert uvw to wavelengths
    max_uv *= max_freq / constants.c
    max_w *= max_freq / constants.c
    print(f"max uv {max_uv} w {max_w}")

    # base cell size
    s0 = 1 / (2 * max_uv)
    # lobe size of conv. kernel
    f0 = math.sqrt(1 + (beta / math.pi) ** 2) / window_size

    d = d0 + window_size * s0
    s = s0  # /(1+2*f0+2*d*max_w)

    n_pix = int(d / s)
    print(f"d={d} s={s} n_pix={n_pix}")

    return 1, 1


def predict(
    msfile,
    ds9_region_file,
    sky_model,
    ra_dec,
    frequency_bandwidth,
    imsize,
    cellsize_deg,
    time_freq_smearing,
    storage_manager,
    predict_bandwidth,
    n_threads,
):
    """
    Predict model image to msfile

    Parameters
    ----------
    msfile : MS name, output will be written to separate columns of this MS
    ds9_region_file: DS9 region file, specifying facet regions and names
    Note: names in region file should be with {}, like {Patch_1}, After
    parsing the {} will be dropped
    sky_model: input sky model

    ra_dec, frequency_bandwidth, imsize, cellsize_deg:
    parameters used for drawing the model image

    time_freq_smearing: if true, enable smearing in predict
    storage_manager: storage manager to use 'default'
    predict_bandwidth: bandwidth of prediction, channels will be split into groups
    n_threads: max threads to use

    """
    # get channel frequencies
    freq_list = ct.table(msfile + "::SPECTRAL_WINDOW").getcol("CHAN_FREQ")
    freq_list = freq_list[0]
    n_chan = freq_list.size
    if n_chan > 1:
        bandwidth = freq_list[-1] - freq_list[0] + (freq_list[1] - freq_list[0])
    else:
        bandwidth = float(frequency_bandwidth[1])
    # split channels into chunks of predict_bandwidth
    if bandwidth > predict_bandwidth:
        n_chunks = int(bandwidth / predict_bandwidth)
    else:
        n_chunks = 1
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
    model_images = list()
    model_counter = 0
    for freqs, chans in zip(freq_chunks, chan_chunks):
        if len(freqs) > 1:
            chunk_bandwidth = freqs[-1] - freqs[0] + (freq_list[1] - freq_list[0])
        else:
            chunk_bandwidth = bandwidth
        chunk_freq = np.mean(freqs)
        err_code = 0
        # only one spectral term is created
        # output will be $(model_name)-term-0.fits
        model_name = os.path.join(tmpdir, f"predict-{model_counter:04d}")
        model_images.append(model_name)
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
        model_counter += 1

    n_models = len(model_images)
    if n_models > 1:
        # make a symlink in same dir with workable name
        for model in model_images:
            # link predict-xxxx-term-0.fits as predict-xxxx-model.fits
            try:
                os.symlink(model + "-term-0.fits", model + "-model.fits")
            except FileExistsError:
                raise
    else:
        model = model_images[0]
        # single model image, drop -xxxx-
        try:
            os.symlink(
                model + "-term-0.fits", os.path.join(os.path.dirname(model), "predict-model.fits")
            )
        except FileExistsError:
            raise

    err_code = 0
    first_facet = 1
    model_name = os.path.join(tmpdir, "predict")
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
            "-channels-out",
            str(n_models),
            "-model-storage-manager",
            str(storage_manager),
            "-j",
            str(n_threads),
        ]
        if time_freq_smearing is not None:
            cmd.append("-apply-time-frequency-smearing")
        cmd.append("-parallel-reordering")
        cmd.append(str(n_threads))
        if first_facet:
            first_facet = 0
        else:
            cmd.append("-reuse-reordered")
        cmd.append("-save-reordered")
        cmd.append(msfile)
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
      wsclean_predict.py --region sector_1_facets_ds9.reg --msin small.ms
      only one MS will be processed (as this is run in scatter mode).
      The new MS name will be written to 'output_info' JSON file
    2) To get column names as one string
      wsclean_predict.py --region sector_1_facets_ds9.reg

    Returns
    -------
    int : 0 if successfull; non-zero otherwise

    Raises
    ------
    ValueError
        If no valid Measurement Set or DS9 region file are given
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
    parser.add_argument(
        "--predict_bandwidth", help="Model image bandwidth (Hz)", type=float, default=2e6
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

    if not os.path.exists(args.region):
        raise ValueError(f"DS9 region file {args.region!r} does not exist")

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

    if len(args.msin) == 0:
        # No MS names given, only output the facet names
        out_dict = {"patches": facet_names}
        with open(output_info, "w") as f:
            json.dump(out_dict, f)

        return 0
    else:
        if len(args.msin) > 1:
            raise ValueError(f"Multiple {args.msin!r}, can work with only one")
        if not os.path.exists(args.msin[0]):
            raise ValueError(f"Input measurement set {args.msin!r} does not exist")
        if not os.path.exists(args.skymodel):
            raise ValueError(f"Sky model file {args.skymodel!r} does not exist")
        if len(args.ra_dec) != 2:
            raise ValueError(f"Invalid RA Dec {args.ra_dec!r}")
        if len(args.frequency_bandwidth) != 2:
            raise ValueError(f"Invalid frequency and bandwidth {args.frequency_bandwidth!r}")
        if len(args.imsize) != 2:
            raise ValueError(f"Invalid image size {args.imsize!r}")

    # Always create a copy of msin to work with,
    # return this as output
    msname = make_writable(args.msin[0])
    out_dict = {"msout": msname, "patches": facet_names}
    with open(output_info, "w") as f:
        json.dump(out_dict, f)

    # determine optimal parameters for model image rendering
    n_pix, cell_size = optimal_imaging_parameters(
        msname, args.skymodel, args.ra_dec[0], args.ra_dec[1]
    )

    # draw model and predict
    return predict(
        msname,
        args.region,
        args.skymodel,
        args.ra_dec,
        args.frequency_bandwidth,
        args.imsize,
        args.cellsize,
        args.time_freq_smearing,
        args.storage_manager,
        args.predict_bandwidth,
        args.threads,
    )


if __name__ == "__main__":
    sys.exit(main())

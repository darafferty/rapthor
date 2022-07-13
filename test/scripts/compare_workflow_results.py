#!/usr/bin/env python3

"""
Script to compare the results of two workflow runs. The generated skymodels,
solutions files and images will be compared, for each major iteration.
"""

import argparse
import filecmp
import logging
import os
import subprocess
import sys

import astropy.io.fits
import coloredlogs
import lsmtool
import numpy as np
import rapthor.lib.image

# import traceback
import warnings

# Suppress UserWarning messages (matplotlib spits out a lot of them)
warnings.filterwarnings("ignore", category=UserWarning)

# warnings.showwarning

# _old_warn = warnings.warn
# def warn(*args, **kwargs):
#     tb = traceback.extract_stack()
#     _old_warn(*args, **kwargs)
#     print("".join(traceback.format_list(tb)[:-1]))
# warnings.warn = warn

# def init_logger():
#     """
#     Initialize our logger.
#     """
# handler = logging.StreamHandler()
# >>> bf = logging.Formatter('{asctime} {name} {levelname:8s} {message}',
# ...                        style='{')
# >>> handler.setFormatter(bf)
# >>> root.addHandler(handler)


# logging.basicConfig(format="{levelname}: {message}", style="{", level=logging.INFO)
coloredlogs.install(level="INFO", fmt="%(levelname)s [%(name)s]: %(message)s")
logger = logging.getLogger()
# logger = logging.getLogger("compare_workflow_results")


def check_all_files_present(dcmp):
    """
    Checks recursively that all files are present in both compared directories

    Parameters
    ----------
    dcmp : dircmp object
        The result of dircmp(left, right)

    Returns
    -------
    agree : bool
        True if all files are present, False if not
    """
    logger.debug("Comparing directory contents of '%s' and '%s'", dcmp.left, dcmp.right)
    agree = True
    if dcmp.left_only or dcmp.right_only:
        agree = False
        msg = "The following files/directories are present in '%s' but not in '%s': %s"
        if dcmp.left_only:
            logger.error(msg, dcmp.left, dcmp.right, dcmp.left_only)
        if dcmp.right_only:
            logger.error(msg, dcmp.right, dcmp.left, dcmp.right_only)
    for sub_dcmp in dcmp.subdirs.values():
        if not check_all_files_present(sub_dcmp):
            agree = False
    return agree


def fits_files_are_similar(file1, file2, rtol=1e-3, verbose=False):
    """
    Compare FITS files `file1` and `file2`, using `FITSDiff` in `astropy`.
    FITS files are considered similar if none of the values have a relative
    difference that exceeds `rtol`.
    :param file1: First FITS file
    :param file2: Second FITS file
    :param rtol: Relative tolerance threshold
    :param verbose: Be more verbose if True
    :return: True if files are similar, else False
    """
    logger.debug("Comparing '%s' and '%s'", file1, file2)
    diff = astropy.io.fits.FITSDiff(file1, file2, rtol=rtol)
    agree = diff.identical
    if agree:
        logger.info("FITS files '%s' and '%s' are similar", file1, file2)
    else:
        if verbose:
            logger.error("FITS files '%s' and '%s' differ (rtol: %s):\n%s", 
                         file1, file2, rtol, diff.report())
        else:
            logger.error("FITS files '%s' and '%s' differ (rtol: %s)", file1, file2, rtol)


def h5_files_are_similar(file1, file2, obj1=None, obj2=None, rtol=1e-3, verbose=False):
    """
    Check if HDF-5 `file1` and `file2` are similar. If `obj1` is not None and `obj2`
    is None, then `obj1` in `file1` is compared against `obj1` in `file2`. If both
    `obj1` and `obj2` are not None, then `obj1` in `file1` is compared against `obj2`
    in `file2`. Files are considered similar if none of the values have a relative
    difference that exceed `rtol`.
    This method uses the `h5diff` command that is part of the HDF5 Tools.
    :param file1: First HDF-5 file
    :param file2: Second HDF-5 file
    :param obj1: Object in file1 (and file2 if obj2 is `None`) to compare 
    :param obj2: Object in file2 to compare
    :param rtol: Relative tolerance threshold
    :param verbose: Be more verbose if files differ
    :return: True if files are similar, else False
    """
    logger.debug("Comparing '%s' and '%s'", file1, file2)
    command = ["h5diff"]
    # if verbose:
    #     command.extend(["--verbose"])
    if not verbose:
        command.extend(["--quiet"])
    command.extend([f"--relative={rtol}"])
    command.extend([file1, file2])
    if obj1:
        command.extend([obj1])
        if obj2:
            command.extend([obj2])
    logger.debug("Executing command: %s", command)
    cp = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        universal_newlines=True,
    )
    if cp.returncode == 0:
        logger.info("HDF5 files '%s' and '%s' are similar", file1, file2)
    if cp.returncode == 1:
        if verbose:
            logger.error("HDF5 files '%s' and '%s' differ (rtol: %s):\n%s", file1, file2, rtol, cp.stdout)
        else:
            logger.error("HDF5 files '%s' and '%s' differ (rtol: %s)", file1, file2, rtol)
    if cp.returncode == 2:
        if verbose:
            logger.error("Failed to compare files '%s' and '%s': %s", file1, file2, cp.stdout)
        else:
            logger.error("Failed to compare files '%s' and '%s'", file1, file2)
    return cp.returncode == 0


def image_files_are_similar(file1, file2, rtol=1e-3, verbose=False):
    """
    Compare two FITS image files, by comparing the RMS noise values.
    Files are considered similar if the the noise ratio differs less than
    `rtol` from 1.
    :param file1: Filename of first FITS image
    :param file2: Filename of second FITS image
    :param rtol: Relative tolerance threshold
    :param verbose: Be more verbose if image noise ratios differ
    :return: True if images are similar, else False
    """
    def image_noise(fname):
        """
        Calculate the RMS value of the noise in the image file `fname`
        """
        image = rapthor.lib.image.FITSImage(fname)
        image.calc_noise()
        return image.noise
    logger.debug("Comparing noise in images '%s' and '%s'", file1, file2)
    noise1 = image_noise(file1)
    noise2 = image_noise(file2)
    agree = np.isclose(noise1, noise2, rtol=rtol)
    if agree:
        logger.info("Image files '%s' and '%s' are similar", file1, file2)
    if not agree:
        if verbose:
            logger.error("Image files '%s' and '%s' differ (rtol: %s):\n"
                         "noise levels are %s and %s", file1, file2, rtol, noise1, noise2)
        else:
            logger.error("Image files '%s' and '%s' differ (rtol: %s)", file1, file2, rtol)
    return agree


def skymodel_files_are_similar(file1, file2, atol=1e-5, rtol=1e-3, verbose=False):
    """
    Compare two skymodel files. Skymodels are considered similar if their
    statistics are similar. This function uses LSMTool to determine the skymodel
    statistics. The skymodels are considered to be similar, if the relevant
    statistics differ less than `atol` or `rtol`.
    :param file1: Filename of the first skymodel 
    :param file2: Filename of the second skymodel
    :param atol: Absolute tolerance threshold
    :param rtol: Relative tolerance threshold
    :param verbose: Be more verbose if skymodels differ
    :return: True is skymodels are similar, else False
    """
    def compare_stats(stats, atol, rtol):
        """
        Compare statistics generated by LSMTool. Check if the flux ratio is
        close to 1, and that the position offsets are less than `atol`
        :param atol: Absolute tolerance used to compare position offsets
        :param rtol: Relative tolerance used to compare flux ratio
        :return: True if statistics are similar, else False
        """
        if not np.isclose(stats['meanRatio'], 1, rtol=rtol):
            return False
        if not np.isclose(stats['meanRAOffsetDeg'], 0, atol=atol):
            return False
        if not np.isclose(stats['meanDecOffsetDeg'], 0, atol=atol):
            return False
        return True
    logger.debug("Comparing skymodels '%s' and '%s'", file1, file2)
    sm1 = lsmtool.load(file1)
    sm2 = lsmtool.load(file2)
    stats = sm1.compare(sm2)
    if not stats:
        agree = False
    else:
        # We're only interested in a few statistics
        stats = {key: stats[key] for key in ('meanRatio', 'meanRAOffsetDeg', 'meanDecOffsetDeg')}
        agree = compare_stats(stats, atol, rtol)
    if agree:
        logger.info("Skymodels '%s' and '%s' are similar", file1, file2)
    else:
        if stats:
            if verbose:
                logger.error("Skymodels '%s' and '%s' differ (atol: %s, rtol: %s): %s", file1, file2, atol, rtol, stats)
            else:
                logger.error("Skymodels '%s' and '%s' differ (atol: %s, rtol: %s)", file1, file2, atol, rtol)
        else:
            logger.error("Failed to compare skymodels '%s' and '%s'", file1, file2)
    return agree


def compare_images(dcmp, rtol=1e-3, verbose=False):
    """
    Compare the image files present in `dircmp` object `dcmp`.
    Only files with extension `.fits` are considered to be image files.
    """
    logger.debug("Comparing directories '%s' and '%s'", dcmp.left, dcmp.right)
    agree = True
    if not check_all_files_present(dcmp):
        agree = False
    for fname in dcmp.common_files:
        if os.path.splitext(fname)[1] == ".fits":
            left = os.path.join(dcmp.left, fname)
            right = os.path.join(dcmp.right, fname)
            if not image_files_are_similar(left, right, rtol):
                agree = False
    for sub_dcmp in dcmp.subdirs.values():
        if not compare_images(sub_dcmp, rtol=rtol, verbose=verbose):
            agree = False
    return agree


def compare_skymodels(dcmp, atol=1e-6, rtol=1e-3, verbose=False):
    """
    Compare the skymodel files present in `dircmp` object `dcmp`.
    Only files with extension `.txt` are considered to be skymodel files.
    """
    logger.debug("Comparing directories '%s' and '%s'", dcmp.left, dcmp.right)
    agree = True
    if not check_all_files_present(dcmp):
        agree = False
    for fname in dcmp.common_files:
        if os.path.splitext(fname)[1] == ".txt":
            left = os.path.join(dcmp.left, fname)
            right = os.path.join(dcmp.right, fname)
            if not skymodel_files_are_similar(left, right, atol=atol, rtol=rtol, verbose=verbose):
                agree = False
    for sub_dcmp in dcmp.subdirs.values():
        if not compare_skymodels(sub_dcmp, atol=atol, rtol=rtol, verbose=verbose):
            agree = False
    return agree


def compare_solutions(dcmp, rtol=1e-3, verbose=False):
    """
    Compare the solution files present in `dircmp` object `dcmp`.
    Only files with extension `.h5` are considered to be solution files.
    """
    logger.debug("Comparing directories '%s' and '%s'", dcmp.left, dcmp.right)
    agree = True
    if not check_all_files_present(dcmp):
        agree = False
    for fname in dcmp.common_files:
        if os.path.splitext(fname)[1] == ".h5":
            left = os.path.join(dcmp.left, fname)
            right = os.path.join(dcmp.right, fname)
            if not h5_files_are_similar(left, right, rtol=rtol, verbose=verbose):
                agree = False
    for sub_dcmp in dcmp.subdirs.values():
        if not compare_solutions(sub_dcmp, rtol=rtol, verbose=verbose):
            agree = False
    return agree


def compare_intermediate_results(dcmp, rtol=1e-2, verbose=False):
    """
    Compare the intermediate results produced by each CWL workflow.
    """
    logger.info("Comparing intermediate results in '%s' and '%s'", dcmp.left, dcmp.right)
    agree = True
    for dname, sub_dcmp in dcmp.subdirs.items():
        print("****", dname, sub_dcmp.left, sub_dcmp.right)
        for fname in sub_dcmp.common_files:
            # print("  ==", fname)
            left = os.path.join(sub_dcmp.left, fname)
            right = os.path.join(sub_dcmp.right, fname)
            ext = os.path.splitext(fname)[1]
            if "fits" in ext:
                if "image" in dname or "mosaic" in dname:
                    if not image_files_are_similar(left, right, rtol=rtol, verbose=verbose):
                        agree = False
                else:
                    if not fits_files_are_similar(left, right, rtol=rtol, verbose=verbose):
                        agree = False
            if "h5" in ext:
                if not h5_files_are_similar(left, right, rtol=rtol, verbose=verbose):
                    agree = False
            if "sky" in ext or ext == ".txt":
                if "image" in dname:
                    if not skymodel_files_are_similar(left, right, rtol=rtol, verbose=verbose):
                        agree = False
    return agree


def parse_arguments():
    """
    Parse command-line arguments.
    :return: argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description="Compare the results of two Rapthor runs.\n",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("left", help="Directory containing results of the first run")
    parser.add_argument("right", help="Directory containing results of the second run")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print more details",
    )
    args = parser.parse_args()
    print("**** args:", args)
    return args



def main(args):
    """
    Compare the images, skymodels and solutions in the `args.left` directory to
    those in the `args.right` directory.
    """
    agree = True

    # if not compare_images(
    #     filecmp.dircmp(
    #         os.path.join(args.left, "images"), os.path.join(args.right, "images")
    #     ),
    #     rtol=0.01,
    #     verbose=args.verbose,
    # ):
    #     agree = False

    # if not compare_skymodels(
    #     filecmp.dircmp(
    #         os.path.join(args.left, "skymodels"), os.path.join(args.right, "skymodels")
    #     ),
    #     verbose=args.verbose,
    # ):
    #     agree = False

    # if not compare_solutions(
    #     filecmp.dircmp(
    #         os.path.join(args.left, "solutions"), os.path.join(args.right, "solutions")
    #     ),
    #     rtol=5e-3,
    #     verbose=args.verbose,
    # ):
    #     agree = False

    if not compare_intermediate_results(
        filecmp.dircmp(
            os.path.join(args.left, "pipelines"), os.path.join(args.right, "pipelines")
        ),
        # rtol=5e-3,
        verbose=args.verbose
    ):
        agree = False

    return 0 if agree else 1


if __name__ == "__main__":
    sys.exit(main(parse_arguments()))

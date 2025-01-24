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
import textwrap
import warnings

import astropy.io.fits
import coloredlogs
import lsmtool
import numpy as np
import rapthor.lib.fitsimage

# Suppress UserWarning messages (mostly generated by matplotlib)
warnings.filterwarnings("ignore", category=UserWarning)

# Global Logger object, will be initialized in init_logger()
logger = None


def check_all_files_present(dcmp):
    """
    Checks that all files are present in both compared directories.

    :param dircmp dcmp: the result of `dircmp(left, right)`
    :return bool: True if all files are present, False if not
    """
    logger.debug("Comparing directory contents of '%s' and '%s'", dcmp.left, dcmp.right)
    agree = True
    if dcmp.left_only or dcmp.right_only:
        agree = False
        msg = "The following files are present in '%s' but not in '%s': %s"
        if dcmp.left_only:
            logger.error(msg, dcmp.left, dcmp.right, dcmp.left_only)
        if dcmp.right_only:
            logger.error(msg, dcmp.right, dcmp.left, dcmp.right_only)
    return agree


def fits_files_are_similar(file1, file2, rtol, verbosity):
    """
    Compare FITS files ``file1`` and ``file2``, using `FITSDiff` in `astropy`.
    FITS files are considered similar if none of the values have a relative
    difference exceeding `rtol`.
    :param str file1: First FITS file
    :param str file2: Second FITS file
    :param float rtol: Relative tolerance threshold
    :param int verbosity: Verbosity level, higher is more verbose
    :return bool: True if FITS files are similar, else False
    """
    logger.debug("Comparing '%s' and '%s'", file1, file2)
    diff = astropy.io.fits.FITSDiff(file1, file2, rtol=rtol)
    agree = diff.identical
    if agree:
        logger.info("FITS files '%s' and '%s' are similar", file1, file2)
    else:
        if verbosity > 0:
            logger.error(
                "FITS files '%s' and '%s' differ (rtol=%s):\n%s",
                file1,
                file2,
                rtol,
                diff.report(),
            )
        else:
            logger.error(
                "FITS files '%s' and '%s' differ (rtol=%s)", file1, file2, rtol
            )
    return agree


def h5_files_are_similar(file1, file2, rtol, verbosity):
    """
    Check if HDF5 ``file1`` and ``file2`` are similar. Files are considered
    similar if none of the values have a relative difference exceeding ``rtol``.
    This method uses the `h5diff` command that is part of the HDF5 Tools.
    :param str file1: First HDF-5 file
    :param str file2: Second HDF-5 file
    :param float rtol: Relative tolerance threshold
    :param int verbosity: Verbosity level, higher is more verbose
    :return bool: True if HDF5 files are similar, else False
    """
    logger.debug("Comparing '%s' and '%s'", file1, file2)
    command = ["h5diff"]
    if verbosity < 0:
        command.extend(["--quiet"])
    elif verbosity > 1:
        command.extend(["--verbose"])
    command.extend([f"--relative={rtol}"])
    command.extend([file1, file2])
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
        if cp.stdout:
            logger.error(
                "HDF5 files '%s' and '%s' differ (rtol=%s)\n%s",
                file1,
                file2,
                rtol,
                cp.stdout,
            )
        else:
            logger.error(
                "HDF5 files '%s' and '%s' differ (rtol=%s)", file1, file2, rtol
            )
    if cp.returncode == 2:
        if cp.stdout:
            logger.error(
                "Failed to compare files '%s' and '%s': %s", file1, file2, cp.stdout
            )
        else:
            logger.error("Failed to compare files '%s' and '%s'", file1, file2)
    return cp.returncode == 0


def image_files_are_similar(file1, file2, rtol, verbosity):
    """
    Compare two FITS image files, by comparing the RMS noise values.
    Files are considered similar if the the noise ratio differs less than
    ``rtol`` from 1.
    :param str file1: Filename of first FITS image
    :param str file2: Filename of second FITS image
    :param float rtol: Relative tolerance threshold
    :param int verbosity: Verbosity level, higher is more verbose
    :return bool: True if images files are similar, else False
    """

    def image_noise(fname):
        """
        Calculate the RMS value of the noise in the image file ``fname``.
        :param str fname: Filename of the FITS image
        :return float: RMS noise in the image in Jy/b
        """
        image = rapthor.lib.fitsimage.FITSImage(fname)
        image.calc_noise()
        return image.noise

    logger.debug("Comparing noise in images '%s' and '%s'", file1, file2)
    noise1 = image_noise(file1)
    noise2 = image_noise(file2)
    agree = np.isclose(noise1, noise2, rtol=rtol)
    if agree:
        logger.info("Image files '%s' and '%s' are similar", file1, file2)
    if not agree:
        if verbosity > 0:
            logger.error(
                "Image files '%s' and '%s' differ (rtol=%s)\n"
                "Noise: %.3f mJy/b vs %.3f mJy/b",
                file1,
                file2,
                rtol,
                1000 * noise1,
                1000 * noise2,
            )
        else:
            logger.error(
                "Image files '%s' and '%s' differ (rtol=%s)", file1, file2, rtol
            )
    return agree


def skymodel_files_are_similar(file1, file2, atol, rtol, verbosity):
    """
    Compare two skymodel files. Skymodels are considered similar if their
    statistics are similar. This function uses LSMTool to determine the skymodel
    statistics. The skymodels are considered to be similar, if the relevant
    statistics differ less than ``atol`` or ``rtol``.
    :param str file1: Filename of the first skymodel
    :param str file2: Filename of the second skymodel
    :param float atol: Absolute tolerance threshold
    :param float rtol: Relative tolerance threshold
    :param int verbosity: Verbosity level, higher is more verbose
    :return bool: True is skymodels file are similar, else False
    """

    def compare_stats(stats, atol, rtol):
        """
        Compare statistics generated by LSMTool. Check if the flux ratio is
        close to 1, and that the position offsets are less than ``atol``
        :param float atol: Absolute tolerance used to compare position offsets
        :param float rtol: Relative tolerance used to compare flux ratio
        :return bool: True if statistics are similar, else False
        """
        if not np.isclose(stats["meanRatio"], 1, rtol=rtol):
            return False
        if not np.isclose(stats["meanRAOffsetDeg"], 0, atol=atol):
            return False
        if not np.isclose(stats["meanDecOffsetDeg"], 0, atol=atol):
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
        stats = {
            key: stats[key]
            for key in ("meanRatio", "meanRAOffsetDeg", "meanDecOffsetDeg")
        }
        agree = compare_stats(stats, atol, rtol)
    if agree:
        logger.info("Skymodel files '%s' and '%s' are similar", file1, file2)
    else:
        if stats:
            if verbosity > 0:
                logger.error(
                    "Skymodel files '%s' and '%s' differ (atol=%s, rtol=%s): %s",
                    file1,
                    file2,
                    atol,
                    rtol,
                    stats,
                )
            else:
                logger.error(
                    "Skymodel files '%s' and '%s' differ (atol=%s, rtol=%s)",
                    file1,
                    file2,
                    atol,
                    rtol,
                )
        else:
            logger.error("Failed to compare skymodel files '%s' and '%s'", file1, file2)
    return agree


def compare_results(dcmp, atol, rtol, verbosity):
    """
    Compare the results between the files produced by each CWL workflow.
    :param dircmp dcmp: the directory contents of the two workflows
    :param float atol: absolute tolerance threshold
    :param float rtol: relative tolerance threshold
    :param int verbosity: verbosity level, higher means more verbose
    """
    logger.info("*** Comparing results in '%s' and '%s' ***", dcmp.left, dcmp.right)
    agree = True
    for dname, sub_dcmp in dcmp.subdirs.items():
        logger.debug(
            "dname: %s, sub_dcmp.left: %s, sub_dcmp.right: %s",
            dname,
            sub_dcmp.left,
            sub_dcmp.right,
        )
        check_all_files_present(sub_dcmp)
        for fname in sub_dcmp.common_files:
            left = os.path.join(sub_dcmp.left, fname)
            right = os.path.join(sub_dcmp.right, fname)
            root, ext = os.path.splitext(fname)
            logger.debug("fname: %s, root: %s, ext: %s", fname, root, ext)
            if "fits" in ext:
                if "image" in dname or "mosaic" in dname:
                    if not image_files_are_similar(
                        left, right, rtol=rtol, verbosity=verbosity
                    ):
                        agree = False
                else:
                    if not fits_files_are_similar(
                        left, right, rtol=rtol, verbosity=verbosity
                    ):
                        agree = False
            if "h5" in ext:
                if not h5_files_are_similar(
                    left, right, rtol=rtol, verbosity=verbosity
                ):
                    agree = False
            if ("sky" in ext or ext == ".txt"):
                if not skymodel_files_are_similar(
                    left, right, atol=atol, rtol=rtol, verbosity=verbosity
                ):
                    agree = False
    return agree


def parse_arguments():
    """
    Parse command-line arguments.
    :return argparse.Namespace: parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """
            Compare the results of two Rapthor runs.
            By default, only the saved results (images, skymodels, and solutions) are
            compared, and not the intermediate results. If the `--all` option is given,
            the intermediate results are also compared. The user can also choose to
            compare just one category of results for comparison: saved images, saved
            skymodels, saved solutions, or intermediate results. Note that the
            intermediate results also contain images, skymodels and solutions.
            """
        ),
    )
    parser.add_argument("first", help="Directory containing results of the first run")
    parser.add_argument("second", help="Directory containing results of the second run")
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-6,
        help="Absolute tolerance threshold for comparing data (default=%(default)s)",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-3,
        help="Relative tolerance threshold for comparing data (default=%(default)s)",
    )
    verb_group = parser.add_mutually_exclusive_group()
    verb_group.add_argument(
        "-q", "--quiet", action="store_true", help="Quiet mode; only report errors"
    )
    verb_group.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Verbose mode; multiple -v options increase verbosity",
    )
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument(
        "--all",
        action="store_true",
        help="Compare all results (including intermediate results)",
    )
    data_group.add_argument(
        "--images", action="store_true", help="Compare saved images only"
    )
    data_group.add_argument(
        "--skymodels", action="store_true", help="Compare saved skymodels only"
    )
    data_group.add_argument(
        "--solutions", action="store_true", help="Compare saved solutions only"
    )
    data_group.add_argument(
        "--intermediate", action="store_true", help="Compare intermediate results only"
    )

    args = parser.parse_args()
    # print("**** args:", args)
    return args


def init_logger(verbosity):
    """
    Initialize a global logger object.
    Set the loglevel depending on the `verbosity`.
    :param int verbosity: Verbosity level, higher means more verbose logging
    """
    global logger
    logger = logging.getLogger()
    fmt = "%(levelname)s [%(name)s]: %(message)s"
    if verbosity == -1:
        coloredlogs.install(fmt=fmt, level="ERROR")
    elif verbosity == 0:
        coloredlogs.install(fmt=fmt, level="WARNING")
    elif verbosity == 1:
        coloredlogs.install(fmt=fmt, level="INFO")
    elif verbosity >= 2:
        coloredlogs.install(fmt=fmt, level="DEBUG")


def main(args):
    """
    Compare the images, skymodels and solutions in the ``args.first`` directory
    to those in the ``args.second`` directory.
    :param argparse.Namespace args: parsed command-line
    :return bool: True if all files in ``args.first`` are similar to those in
                  ``args.second``, else False.
    """
    # Set level of verbosity to -1 if command-line option `-q` is given, else
    # to the number of times the command-line option `-v` is given.
    verbosity = -1 if args.quiet else args.verbose

    # Initialize logger and set log level depending on verbosity level
    init_logger(verbosity)

    no_opts = not any(
        (args.all, args.images, args.skymodels, args.solutions, args.intermediate)
    )
    agree = True

    if args.images or args.all or no_opts:
        if not compare_results(
            filecmp.dircmp(
                os.path.join(args.first, "images"), os.path.join(args.second, "images")
            ),
            atol=args.atol,
            rtol=args.rtol,
            verbosity=verbosity,
        ):
            agree = False

    if args.skymodels or args.all or no_opts:
        if not compare_results(
            filecmp.dircmp(
                os.path.join(args.first, "skymodels"),
                os.path.join(args.second, "skymodels"),
            ),
            atol=args.atol,
            rtol=args.rtol,
            verbosity=verbosity,
        ):
            agree = False

    if args.solutions or args.all or no_opts:
        if not compare_results(
            filecmp.dircmp(
                os.path.join(args.first, "solutions"),
                os.path.join(args.second, "solutions"),
            ),
            atol=args.atol,
            rtol=args.rtol,
            verbosity=verbosity,
        ):
            agree = False

    if args.intermediate or args.all:
        if not compare_results(
            filecmp.dircmp(
                os.path.join(args.first, "pipelines"),
                os.path.join(args.second, "pipelines"),
            ),
            atol=args.atol,
            rtol=args.rtol,
            verbosity=verbosity,
        ):
            agree = False

    return agree


if __name__ == "__main__":
    success = main(parse_arguments())
    sys.exit(0 if success else 1)

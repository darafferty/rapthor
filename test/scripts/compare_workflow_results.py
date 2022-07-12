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
import h5py
import lsmtool
import numpy as np
import rapthor.lib.image

# import traceback
# import warnings

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
coloredlogs.install(level="INFO", fmt="%(levelname)s: %(message)s")
logger = logging.getLogger("compare_workflow_results")


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


# def compare_solution_files(left, right, atol=1e-3, rtol=1e-3):
#     """
#     Compare the values in the solution files `left` and `right`.
#     This method assumes that the h5 files are produced by the Rapthor pipeline,
#     hence have a certain known structure.
#     :param left: First solutions file
#     :param right: Second solutions file
#     :param atol: Absolute tolerance threshold
#     :param rtol: Relative tolerance threshold
#     :return: True if files are similar, else False.
#     """
#     agree = True
#     with h5py.File(left, "r") as h5_left, h5py.File(right, "r") as h5_right:
#         for solsetname in h5_left:
#             soltabnames = [
#                 name
#                 for name in h5_left[solsetname]
#                 if name not in ("source", "antenna")
#             ]
#             for soltabname in soltabnames:
#                 left_soltabval = h5_left[solsetname][soltabname]["val"]
#                 try:
#                     right_soltabval = h5_right[solsetname][soltabname]["val"]
#                 except KeyError:
#                     logger.error(
#                         "Table '%s' in set '%s' is present in file '%s', but not in "
#                         "file '%s'",
#                         soltabname,
#                         solsetname,
#                         left,
#                         right,
#                     )
#                     return False
#                 # matching_vals = right_soltabval
#                 # if 'freq' in h5_left[solsetname][soltabname].keys() and args.allow_frequency_subset:
#                 #     left_axes = left_soltabval.attrs['AXES'].decode('utf-8').split(',')
#                 #     freq_axis_index = left_axes.index('freq')
#                 #     left_soltabfreq = h5_left[solsetname][soltabname]['freq'][:]
#                 #     right_soltabfreq = h5_right[solsetname][soltabname]['freq'][:]
#                 #     matches = np.isclose(right_soltabfreq[:, np.newaxis], left_soltabfreq)
#                 #     matching_freq_indices = np.where(matches)[0]
#                 #     matching_vals = np.take(right_soltabval,
#                 #                             matching_freq_indices,
#                 #                             axis=freq_axis_index)
#                 # if not np.allclose(left_soltabval, matching_vals,

#                 # Calculate cosine of phase to avoid phase jump issues
#                 if 'phase' in soltabname:
#                     logger.debug("Calculate cosine to avoid phase jump issues")
#                     left_soltabval = np.cos(left_soltabval)
#                     right_soltabval = np.cos(right_soltabval)
#                 if np.allclose(
#                     left_soltabval,
#                     right_soltabval,
#                     rtol=rtol,
#                     atol=atol,
#                     equal_nan=True,
#                 ):
#                     logger.info("Values in solution table '%s' match", soltabname)
#                 else:
#                     agree = False
#                     logger.error(
#                         "Values in solution table '%s' do not match", soltabname
#                     )
#                     # if dump_vals:
#                     #     with open(f"left.{soltabname}.val", "w") as f:
#                     #         f.write(str(left_soltabval[:]))
#                     #     with open(f"right.{soltabname}.val", "w") as f:
#                     #         f.write(str(right_soltabval[:]))
#     return agree


def fits_files_are_similar(file1, file2, rtol=1e-3, verbose=False):
    """
    Compare FITS files `file1` and `file2`, using `FITSDiff` in `astropy`.
    """
    logger.debug("Comparing '%s' and '%s'", file1, file2)
    diff = astropy.io.fits.FITSDiff(file1, file2, rtol=rtol)
    agree = diff.identical
    if agree:
        logger.info("FITS files '%s' and '%s' are similar", file1, file2)
    else:
        if verbose:
            logger.error("FITS files '%s' and '%s' are different:\n%s", 
                         file1, file2, diff.report())
        else:
            logger.error("FITS files '%s' and '%s' are different", file1, file2)


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
    :param verbose: Print `h5diff` report
    :return: True if files are similar, else False.
    """
    logger.debug("Comparing '%s' and '%s'", file1, file2)
    command = ["h5diff"]
    if verbose:
        command.extend(["--verbose"])
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
        logger.error("HDF5 files '%s' and '%s' differ:\n%s", file1, file2, cp.stdout)
    if cp.returncode == 2:
        logger.error("Failed to compare files '%s' and '%s': %s", file1, file2, cp.stdout)
    return cp.returncode == 0


def image_noise_is_similar(file1, file2, rtol=1e-3, verbose=False):
    """
    Check if the RMS noise values of `image1` and `image2` are similar.
    :param file1: Filename of first FITS image
    :param file2: Filename of second FITS image
    :param rtol: Relative tolerance threshold
    :param verbose: Be more verbose if image noise ratios differ
    :return: True if noise ratio differs less than `rtol` from 1, else False
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
        logger.info("Image noise in '%s' and '%s' is similar", file1, file2)
    if not agree:
        if verbose:
            logger.error("Image noise in '%s' and '%s' is different: %f and %f",
                         file1, file2, noise1, noise2)
        else:
            logger.error("Image noise in '%s' and '%s' is different", file1, file2)
    return agree


def skymodels_are_similar(file1, file2, atol=1e-6, rtol=1e-3, verbose=False):
    """
    Check if the skymodels `sky1` and `sky2` are similar.
    Skymodels are considered similar if their statistics are similar.
    This function uses LSMTool to determine the skymodel statistics.
    The skymodels are considered to be similar, if the relevant statistics
    differ less than `atol` or `rtol`.
    :param file1: Filename of the first skymodel 
    :param file2: Filename of the second skymodel
    :param atol: Absolute tolerance threshold
    :param rtol: Relative tolerance threshold
    :param verbose: If True, be more verbose if skymodels differ
    :return: True is skymodels are similar, else False
    """
    def compare_stats(stats, atol, rtol):
        """
        Check if the flux ratio is close to 1, and that the position offsets
        are less than `atol`
        :param atol: Absolute tolerance used to compare position offsets
        :param rtol: Relative tolerance used to compare flux ratio
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
    agree = compare_stats(stats, atol, rtol)
    if agree:
        logger.info("Skymodels '%s' and '%s' are similar", file1, file2)
    else:
        if verbose:
            logger.error("Skymodels '%s' and '%s' differ: %s", file1, file2, stats)
        else:
            logger.error("Skymodels '%s' and '%s' differ", file1, file2)
    return agree


def compare_images(dcmp, rtol=1e-3, verbose=False):
    """
    """
    logger.debug("Comparing images in '%s' and '%s'", dcmp.left, dcmp.right)
    agree = True
    if not check_all_files_present(dcmp):
        agree = False
    for fname in dcmp.common_files:
        if os.path.splitext(fname)[1] == ".fits":
            left = os.path.join(dcmp.left, fname)
            right = os.path.join(dcmp.right, fname)
            if not image_noise_is_similar(left, right, rtol):
                agree = False
    for sub_dcmp in dcmp.subdirs.values():
        if not compare_images(sub_dcmp, rtol=rtol, verbose=verbose):
            agree = False
    return agree


def compare_skymodels(dcmp, atol=1e-6, rtol=1e-3, verbose=False):
    """
    Compare the skymodels present in the `dcmp`, a `dircmp` object
    """
    logger.info("Comparing directories '%s' and '%s'", dcmp.left, dcmp.right)
    agree = True
    if not check_all_files_present(dcmp):
        agree = False
    for fname in dcmp.common_files:
        if os.path.splitext(fname)[1] == ".txt":
            left = os.path.join(dcmp.left, fname)
            right = os.path.join(dcmp.right, fname)
            if not skymodels_are_similar(left, right, atol=atol, rtol=rtol, verbose=verbose):
                agree = False
    for sub_dcmp in dcmp.subdirs.values():
        if not compare_skymodels(sub_dcmp, atol=atol, rtol=rtol, verbose=verbose):
            agree = False
    return agree


def compare_solutions(dcmp, rtol=1e-3, verbose=False):
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


# def compare_lsm_stats(stats, atol, rtol):
#     """
#     Check if the flux ratio is close to 1, and that the position offsets
#     are less than `atol`
#     :param atol: Absolute tolerance used to compare position offsets
#     :param rtol: Relative tolerance used to compare flux ratio
#     """
#     if not np.isclose(stats['meanRatio'], 1, rtol=rtol):
#         return False
#     if not np.isclose(stats['meanRAOffsetDeg'], 0, atol=atol):
#         return False
#     if not np.isclose(stats['meanDecOffsetDeg'], 0, atol=atol):
#         return False
#     return True

# def image_noise(fname):
#     """
#     Calculate the RMS value of the noise in the image file `fname`
#     """
#     image = rapthor.lib.image.FITSImage(fname)
#     image.calc_noise()
#     return image.noise


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
                    if not image_noise_is_similar(left, right, rtol=rtol, verbose=verbose):
                        agree = False
                else:
                    if not fits_files_are_similar(left, right, rtol=rtol, verbose=verbose):
                        agree = False
            if "h5" in ext:
                if not h5_files_are_similar(left, right, rtol=rtol, verbose=verbose):
                    agree = False
            if "sky" in ext or ext == ".txt":
                if "image" in dname:
                    if not skymodels_are_similar(left, right, rtol=rtol, verbose=verbose):
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

    if not compare_images(
        filecmp.dircmp(
            os.path.join(args.left, "images"), os.path.join(args.right, "images")
        ),
        rtol=0.01,
        verbose=args.verbose,
    ):
        agree = False

    if not compare_skymodels(
        filecmp.dircmp(
            os.path.join(args.left, "skymodels"), os.path.join(args.right, "skymodels")
        ),
        verbose=args.verbose,
    ):
        agree = False

    if not compare_solutions(
        filecmp.dircmp(
            os.path.join(args.left, "solutions"), os.path.join(args.right, "solutions")
        ),
        rtol=5e-3,
        verbose=args.verbose,
    ):
        agree = False

    if not compare_intermediate_results(
        filecmp.dircmp(
            os.path.join(args.left, "pipelines"), os.path.join(args.right, "pipelines")
        )
    ):
        agree = False

    return 0 if agree else 1


if __name__ == "__main__":
    sys.exit(main(parse_arguments()))

#!/usr/bin/env python3
"""
Script to collect multiple H5parms containing screen solutions by concatenating
in time
"""

from argparse import ArgumentParser, RawTextHelpFormatter
import logging

from rapthor.execution.calibrate.screen_h5parms import (
    collect_screen_h5parms,
    parse_h5parm_file_list,
)


def main(h5parm_files, outh5parm_file, overwrite):
    """
    Collects multiple screen h5parms into a single h5parm by concatenating in
    time

    Parameters
    ----------
    h5parm_files : list
        Filenames of input screen h5parms. The h5parms should contian solutions
        with the same structure but for different times
    outh5parm_file : str
        Filename of the output h5parm
    overwrite : bool
        If True, overwrite existing output H5parm file
    """
    collect_screen_h5parms(h5parm_files, outh5parm_file, overwrite=overwrite)


if __name__ == "__main__":
    descriptiontext = "Collect multiple screen h5parms in time.\n"

    parser = ArgumentParser(description=descriptiontext, formatter_class=RawTextHelpFormatter)
    parser.add_argument("h5parm_files", nargs="+", help="List of input h5parms")
    parser.add_argument(
        "--outh5parm",
        "-o",
        default="output.h5",
        dest="outh5parm",
        help="Output h5parm name [default: output.h5]",
    )
    parser.add_argument(
        "--overwrite",
        "-c",
        default=False,
        action="store_true",
        help="Replace existing outh5parm file instead of appending to it (default=False)",
    )
    args = parser.parse_args()

    try:
        main(parse_h5parm_file_list(args.h5parm_files), args.outh5parm, args.overwrite)
    except ValueError as e:
        log = logging.getLogger("rapthor:collect_screen_h5parms")
        log.critical(e)

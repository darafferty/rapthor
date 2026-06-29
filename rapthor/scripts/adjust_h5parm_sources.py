#!/usr/bin/env python3
"""
Script to adjust h5parm source coordinates to match those in the sky model
"""

from argparse import ArgumentParser, RawTextHelpFormatter
import sys

from rapthor.execution.calibrate.h5parm_sources import adjust_h5parm_source_coordinates


def main(skymodel, h5parm_file, solset_name="sol000"):
    """
    Adjust the h5parm source coordinates to match those in the sky model

    Parameters
    ----------
    skymodel : str
        Filename of calibration sky model
    h5parm_file : str
        Filename of h5parm with cooresponding solutions to update with
        the facet positions
    solset_name : str, optional
        Name of the solution set to adjust
    """
    adjust_h5parm_source_coordinates(skymodel, h5parm_file, solset_name=solset_name)


if __name__ == "__main__":
    descriptiontext = "Adjust the h5parm source coordinates to match those in the sky model.\n"

    parser = ArgumentParser(description=descriptiontext, formatter_class=RawTextHelpFormatter)
    parser.add_argument("skymodel", help="Filename of input sky model")
    parser.add_argument(
        "h5parm_file",
        help="Filename of h5parm with corresponding solutions to update",
    )
    parser.add_argument("--solset_name", help="Name of solution set", type=str, default="sol000")
    args = parser.parse_args()
    try:
        main(args.skymodel, args.h5parm_file, solset_name=args.solset_name)
    except ValueError as err:
        sys.exit(f"ERROR: {err}")

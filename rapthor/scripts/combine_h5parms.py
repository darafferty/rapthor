#!/usr/bin/env python3
"""CLI wrapper for combining two h5parm solution files."""

import logging
from argparse import ArgumentParser, RawTextHelpFormatter

from rapthor.execution.calibrate.h5parm_combination import combine_h5parms


def main(
    h5parm1,
    h5parm2,
    outh5parm,
    mode,
    reweight=False,
    cal_names=None,
    cal_fluxes=None,
):
    """Combine two h5parm solution files."""
    combine_h5parms(
        h5parm1,
        h5parm2,
        outh5parm,
        mode,
        reweight=reweight,
        cal_names=cal_names,
        cal_fluxes=cal_fluxes,
    )


if __name__ == "__main__":
    descriptiontext = "Combine two h5parms.\n"

    parser = ArgumentParser(description=descriptiontext, formatter_class=RawTextHelpFormatter)
    parser.add_argument("h51", help="Filename of input h5 1")
    parser.add_argument("h52", help="Filename of input h5 2")
    parser.add_argument("outh5", help="Filename of the output h5")
    parser.add_argument("mode", help="Mode to use")
    parser.add_argument("--reweight", help="Reweight solutions", type=str, default="False")
    parser.add_argument("--cal_names", help="Names of calibrators", type=str, default="")
    parser.add_argument("--cal_fluxes", help="Flux densities of calibrators", type=str, default="")
    args = parser.parse_args()

    try:
        main(
            args.h51,
            args.h52,
            args.outh5,
            args.mode,
            reweight=args.reweight,
            cal_names=args.cal_names,
            cal_fluxes=args.cal_fluxes,
        )
    except ValueError as error:
        log = logging.getLogger("rapthor:combine_h5parms")
        log.critical(error)

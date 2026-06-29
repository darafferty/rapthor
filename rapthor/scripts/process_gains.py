#!/usr/bin/env python3
"""CLI wrapper for processing calibration gain solutions."""

from argparse import ArgumentParser, RawTextHelpFormatter

from rapthor.execution.calibrate.gain_processing import process_gain_solutions


def main(
    h5parmfile,
    solsetname="sol000",
    ampsoltabname="amplitude000",
    phasesoltabname="phase000",
    ref_id=None,
    smooth=False,
    normalize=False,
    flag=False,
    lowampval=None,
    highampval=None,
    max_station_delta=0.0,
    scale_delta_with_dist=False,
    phase_center=None,
):
    """Process gain solutions."""
    process_gain_solutions(
        h5parmfile,
        solsetname=solsetname,
        ampsoltabname=ampsoltabname,
        phasesoltabname=phasesoltabname,
        ref_id=ref_id,
        smooth=smooth,
        normalize=normalize,
        flag=flag,
        lowampval=lowampval,
        highampval=highampval,
        max_station_delta=max_station_delta,
        scale_delta_with_dist=scale_delta_with_dist,
        phase_center=phase_center,
    )


if __name__ == "__main__":
    descriptiontext = "Process gain solutions.\n"

    parser = ArgumentParser(description=descriptiontext, formatter_class=RawTextHelpFormatter)
    parser.add_argument("h5parmfile", help="Filename of input h5parm")
    parser.add_argument("--solsetname", help="Solset name", type=str, default="sol000")
    parser.add_argument(
        "--ampsoltabname", help="Amplitude soltab name", type=str, default="amplitude000"
    )
    parser.add_argument("--phasesoltabname", help="Phase soltab name", type=str, default="phase000")
    parser.add_argument("--ref_id", help="Reference station", type=int, default=0)
    parser.add_argument(
        "--normalize", help="Normalize amplitude solutions", type=str, default="False"
    )
    parser.add_argument("--smooth", help="Smooth amplitude solutions", type=str, default="False")
    parser.add_argument("--flag", help="Flag amplitude solutions", type=str, default="False")
    parser.add_argument(
        "--lowampval", help="Low threshold for amplitude flagging", type=float, default=None
    )
    parser.add_argument(
        "--highampval", help="High threshold for amplitude flagging", type=float, default=None
    )
    parser.add_argument(
        "--max_station_delta",
        help="Max difference of median from unity allowed for station normalizations",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--scale_delta_with_dist",
        help="Scale max difference with distance",
        type=str,
        default="False",
    )
    parser.add_argument(
        "--phase_center_ra", help="RA of phase center in degrees", type=float, default=0.0
    )
    parser.add_argument(
        "--phase_center_dec", help="Dec of phase center in degrees", type=float, default=0.0
    )
    args = parser.parse_args()
    phase_center = (args.phase_center_ra, args.phase_center_dec)
    main(
        args.h5parmfile,
        solsetname=args.solsetname,
        ampsoltabname=args.ampsoltabname,
        phasesoltabname=args.phasesoltabname,
        ref_id=args.ref_id,
        smooth=args.smooth,
        normalize=args.normalize,
        flag=args.flag,
        lowampval=args.lowampval,
        highampval=args.highampval,
        max_station_delta=args.max_station_delta,
        scale_delta_with_dist=args.scale_delta_with_dist,
        phase_center=phase_center,
    )

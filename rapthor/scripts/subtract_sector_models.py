#!/usr/bin/env python3
"""CLI wrapper for subtracting sector model data."""

from argparse import ArgumentParser, RawTextHelpFormatter

from rapthor.execution.predict.sector_model_subtraction import subtract_sector_models
from rapthor.lib import miscellaneous as misc


def main(
    msin,
    model_list,
    msin_column="DATA",
    model_column="DATA",
    out_column="DATA",
    nr_outliers=0,
    nr_bright=0,
    use_compression=False,
    peel_outliers=False,
    peel_bright=False,
    reweight=True,
    starttime=None,
    solint_sec=None,
    solint_hz=None,
    weights_colname="CAL_WEIGHT",
    gainfile="",
    uvcut_min=80.0,
    uvcut_max=1e6,
    phaseonly=True,
    dirname=None,
    quiet=True,
    infix="",
):
    """Subtract sector model data."""
    subtract_sector_models(
        msin,
        model_list,
        msin_column=msin_column,
        model_column=model_column,
        out_column=out_column,
        nr_outliers=nr_outliers,
        nr_bright=nr_bright,
        use_compression=use_compression,
        peel_outliers=peel_outliers,
        peel_bright=peel_bright,
        reweight=reweight,
        starttime=starttime,
        solint_sec=solint_sec,
        solint_hz=solint_hz,
        weights_colname=weights_colname,
        gainfile=gainfile,
        uvcut_min=uvcut_min,
        uvcut_max=uvcut_max,
        phaseonly=phaseonly,
        dirname=dirname,
        quiet=quiet,
        infix=infix,
    )


if __name__ == "__main__":
    descriptiontext = "Subtract sector model data.\n"

    parser = ArgumentParser(description=descriptiontext, formatter_class=RawTextHelpFormatter)
    parser.add_argument("msin", help="Filename of input MS data file")
    parser.add_argument("msmod", help="Filename of input MS model data file")
    parser.add_argument("--msin_column", help="Name of msin column", type=str, default="DATA")
    parser.add_argument("--model_column", help="Name of msmod column", type=str, default="DATA")
    parser.add_argument("--out_column", help="Name of output column", type=str, default="DATA")
    parser.add_argument("--nr_outliers", help="Number of outlier sectors", type=int, default=0)
    parser.add_argument("--nr_bright", help="Number of bright-source sectors", type=int, default=0)
    parser.add_argument("--use_compression", help="Use compression", type=str, default="False")
    parser.add_argument("--peel_outliers", help="Peel outliers", type=str, default="False")
    parser.add_argument("--peel_bright", help="Peel bright sources", type=str, default="False")
    parser.add_argument("--reweight", help="Reweight", type=str, default="True")
    parser.add_argument("--starttime", help="Start time in MVT", type=str, default=None)
    parser.add_argument("--solint_sec", help="Solution interval in s", type=float, default=None)
    parser.add_argument("--solint_hz", help="Solution interval in Hz", type=float, default=None)
    parser.add_argument(
        "--weights_colname", help="Name of weight column", type=str, default="CAL_WEIGHT"
    )
    parser.add_argument("--gainfile", help="Filename of gain file", type=str, default="")
    parser.add_argument("--uvcut_min", help="Min uv cut in lambda", type=float, default=80.0)
    parser.add_argument("--uvcut_max", help="Max uv cut in lambda", type=float, default=1e6)
    parser.add_argument("--phaseonly", help="Reweight with phases only", type=str, default="True")
    parser.add_argument("--dirname", help="Name of gain file directory", type=str, default=None)
    parser.add_argument("--quiet", help="Quiet", type=str, default="True")
    parser.add_argument("--infix", help="Infix for output files", type=str, default="")
    args = parser.parse_args()

    main(
        args.msin,
        misc.string2list(args.msmod),
        msin_column=args.msin_column,
        model_column=args.model_column,
        out_column=args.out_column,
        nr_outliers=args.nr_outliers,
        nr_bright=args.nr_bright,
        use_compression=args.use_compression,
        peel_outliers=args.peel_outliers,
        peel_bright=args.peel_bright,
        reweight=args.reweight,
        starttime=args.starttime,
        solint_sec=args.solint_sec,
        solint_hz=args.solint_hz,
        weights_colname=args.weights_colname,
        gainfile=args.gainfile,
        uvcut_min=args.uvcut_min,
        uvcut_max=args.uvcut_max,
        phaseonly=args.phaseonly,
        dirname=args.dirname,
        quiet=args.quiet,
        infix=args.infix,
    )

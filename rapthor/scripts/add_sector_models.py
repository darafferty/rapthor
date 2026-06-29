#!/usr/bin/env python3
"""CLI wrapper for adding sector model data."""

from argparse import ArgumentParser, RawTextHelpFormatter

from rapthor.execution.predict.sector_model_addition import add_sector_models
from rapthor.lib import miscellaneous as misc


def main(
    msin,
    msmod_list,
    msin_column="DATA",
    model_column="DATA",
    out_column="MODEL_DATA",
    use_compression=False,
    starttime=None,
    quiet=True,
    infix="",
):
    """Add sector model data."""
    add_sector_models(
        msin,
        msmod_list,
        msin_column=msin_column,
        model_column=model_column,
        out_column=out_column,
        use_compression=use_compression,
        starttime=starttime,
        quiet=quiet,
        infix=infix,
    )


if __name__ == "__main__":
    descriptiontext = "Add sector model data.\n"

    parser = ArgumentParser(description=descriptiontext, formatter_class=RawTextHelpFormatter)
    parser.add_argument("msin", help="Filename of input MS data file")
    parser.add_argument("msmod", help="Filename of input MS model data file")
    parser.add_argument("--msin_column", help="Name of msin column", type=str, default="DATA")
    parser.add_argument(
        "--model_column", help="Name of input model data column", type=str, default="DATA"
    )
    parser.add_argument(
        "--out_column", help="Name of output model data column", type=str, default="MODEL_DATA"
    )
    parser.add_argument("--use_compression", help="Use compression", type=str, default="False")
    parser.add_argument("--starttime", help="Start time in MVT", type=str, default=None)
    parser.add_argument("--quiet", help="Quiet", type=str, default="True")
    parser.add_argument("--infix", help="Infix for output files", type=str, default="")
    args = parser.parse_args()

    main(
        args.msin,
        misc.string2list(args.msmod),
        msin_column=args.msin_column,
        model_column=args.model_column,
        out_column=args.out_column,
        use_compression=args.use_compression,
        starttime=args.starttime,
        quiet=args.quiet,
        infix=args.infix,
    )

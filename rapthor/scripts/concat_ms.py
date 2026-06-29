#!/usr/bin/env python3
"""
Script to concatenate MS files
"""

from argparse import ArgumentParser, RawTextHelpFormatter
import sys

from rapthor.execution.concatenate.measurement_sets import concat_ms


def main():
    """
    Concatentates two or more input Measurement Sets into one output Measurement Set.

    Returns
    -------
    int : 0 if successfull; non-zero otherwise.
    """
    descriptiontext = "Concatenate Measurement Sets.\n"
    parser = ArgumentParser(description=descriptiontext, formatter_class=RawTextHelpFormatter)
    parser.add_argument("msin", nargs="+", help="List of input Measurement Sets")
    parser.add_argument(
        "--data_colname", help="Data column to be concatenated", type=str, default="DATA"
    )
    parser.add_argument("--msout", help="Output Measurement Set", type=str, default="concat.ms")
    parser.add_argument(
        "--concat_property",
        help="Property over which to concatenate: time or frequency",
        type=str,
        default="frequency",
    )
    parser.add_argument(
        "--overwrite", help="Overwrite existing output file", type=bool, default=False
    )

    args = parser.parse_args()
    return concat_ms(
        args.msin,
        args.msout,
        args.data_colname,
        concat_property=args.concat_property,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    sys.exit(main())

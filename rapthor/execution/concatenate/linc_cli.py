"""Command-line adapter for concatenating LINC Measurement Sets."""

import argparse
from typing import Optional, Sequence

from rapthor.execution.concatenate.measurement_sets import concat_linc_measurement_sets


def build_parser() -> argparse.ArgumentParser:
    """Build the ``concat_linc_files`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="concat_linc_files",
        description="Concatenate LINC Measurement Sets.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "input_path", help="Full path to the directory with the LINC Measurement Sets"
    )
    parser.add_argument("output_file", help="Output Measurement Set path")
    parser.add_argument(
        "--overwrite",
        help="Overwrite an existing output Measurement Set",
        action="store_true",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the ``concat_linc_files`` command."""
    args = build_parser().parse_args(argv)
    try:
        return concat_linc_measurement_sets(
            args.input_path,
            args.output_file,
            overwrite=args.overwrite,
        )
    except Exception as err:
        raise SystemExit(f"Error: {err}") from err


if __name__ == "__main__":
    raise SystemExit(main())

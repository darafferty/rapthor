"""Command-line adapter for preparing sky models for a run without internet access."""

import argparse
from typing import Optional, Sequence

from rapthor.execution.image.catalog_preparation import (
    DEFAULT_RADIUS_DEG,
    parset_options,
    prepare_catalogs,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the ``fetch_skymodels`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="fetch_skymodels",
        description=(
            "Download the sky models for the image diagnostics and the flux-scale "
            "normalization, for use in a run without internet access."
        ),
    )
    parser.add_argument("ms_file", help="Measurement Set whose pointing centers the downloads")
    parser.add_argument("output_dir", help="Directory for the sky models")
    parser.add_argument(
        "--radius",
        help="Radius of the downloads in degrees (default: %(default)s)",
        type=float,
        default=DEFAULT_RADIUS_DEG,
    )
    parser.add_argument(
        "--overwrite",
        help="Replace sky models that are already present",
        action="store_true",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the ``fetch_skymodels`` command."""
    args = build_parser().parse_args(argv)
    try:
        catalogs = prepare_catalogs(
            args.ms_file, args.output_dir, radius_deg=args.radius, overwrite=args.overwrite
        )
    except (OSError, ValueError) as err:
        raise SystemExit(f"Error: {err}") from err
    if catalogs.failures:
        raise SystemExit(
            f"Error: could not download {', '.join(catalogs.failures)}. Rerun this command to "
            "try again; the sky models that were downloaded are kept."
        )
    print(parset_options(catalogs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

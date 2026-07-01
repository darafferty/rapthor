"""Command-line adapter for image-cube source catalog generation."""

from argparse import ArgumentParser, Namespace
from typing import Optional, Sequence

from rapthor.execution.image.cubes import (
    DEFAULT_CUBE_CATALOG_ADAPTIVE_THRESH,
    DEFAULT_CUBE_CATALOG_NCORES,
    DEFAULT_CUBE_CATALOG_RMSBOX,
    DEFAULT_CUBE_CATALOG_RMSBOX_BRIGHT,
    DEFAULT_CUBE_CATALOG_THRESHISL,
    DEFAULT_CUBE_CATALOG_THRESHPIX,
    make_catalog_from_image_cube,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> Namespace:
    """Parse command-line arguments for image-cube catalog generation."""
    parser = ArgumentParser(description="Generate a PyBDSF source catalog from an image cube.")
    parser.add_argument("cube_image", help="Input FITS image cube")
    parser.add_argument("cube_beams", help="Beam metadata sidecar file")
    parser.add_argument("cube_frequencies", help="Frequency metadata sidecar file")
    parser.add_argument("output_catalog", help="Output FITS source catalog")
    parser.add_argument(
        "--threshisl",
        type=float,
        default=DEFAULT_CUBE_CATALOG_THRESHISL,
        help="Island threshold",
    )
    parser.add_argument(
        "--threshpix",
        type=float,
        default=DEFAULT_CUBE_CATALOG_THRESHPIX,
        help="Peak pixel threshold",
    )
    parser.add_argument(
        "--rmsbox",
        default=str(DEFAULT_CUBE_CATALOG_RMSBOX),
        help="RMS box as a tuple string",
    )
    parser.add_argument(
        "--rmsbox_bright",
        default=str(DEFAULT_CUBE_CATALOG_RMSBOX_BRIGHT),
        help="Bright-source RMS box as a tuple string",
    )
    parser.add_argument(
        "--adaptive_thresh",
        type=float,
        default=DEFAULT_CUBE_CATALOG_ADAPTIVE_THRESH,
        help="Adaptive threshold",
    )
    parser.add_argument(
        "--ncores",
        type=int,
        default=DEFAULT_CUBE_CATALOG_NCORES,
        help="PyBDSF worker cores",
    )
    return parser.parse_args(argv)


def run(argv: Optional[Sequence[str]] = None) -> None:
    """Run the image-cube catalog CLI adapter."""
    args = parse_args(argv)
    make_catalog_from_image_cube(
        args.cube_image,
        args.cube_beams,
        args.cube_frequencies,
        args.output_catalog,
        threshisl=args.threshisl,
        threshpix=args.threshpix,
        rmsbox=args.rmsbox,
        rmsbox_bright=args.rmsbox_bright,
        adaptive_thresh=args.adaptive_thresh,
        ncores=args.ncores,
    )


if __name__ == "__main__":
    run()

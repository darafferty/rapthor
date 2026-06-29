"""Command-line adapter for image-cube source catalog generation."""

from argparse import ArgumentParser, Namespace
from typing import Optional, Sequence

from rapthor.execution.image.cubes import make_catalog_from_image_cube


def parse_args(argv: Optional[Sequence[str]] = None) -> Namespace:
    """Parse command-line arguments for image-cube catalog generation."""
    parser = ArgumentParser(description="Generate a PyBDSF source catalog from an image cube.")
    parser.add_argument("cube_image", help="Input FITS image cube")
    parser.add_argument("cube_beams", help="Beam metadata sidecar file")
    parser.add_argument("cube_frequencies", help="Frequency metadata sidecar file")
    parser.add_argument("output_catalog", help="Output FITS source catalog")
    parser.add_argument("--threshisl", type=float, default=3.0, help="Island threshold")
    parser.add_argument("--threshpix", type=float, default=5.0, help="Peak pixel threshold")
    parser.add_argument("--rmsbox", default="(150, 50)", help="RMS box as a tuple string")
    parser.add_argument(
        "--rmsbox_bright",
        default="(35, 7)",
        help="Bright-source RMS box as a tuple string",
    )
    parser.add_argument("--adaptive_thresh", type=float, default=75.0, help="Adaptive threshold")
    parser.add_argument("--ncores", type=int, default=8, help="PyBDSF worker cores")
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

#!/usr/bin/env python3
"""Script to restore a skymodel into an image."""

from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path

from rapthor.execution.image.restoration import restore_skymodel


def main(source_catalog: Path, reference_image: Path, output_image: Path) -> None:
    """
    Restore a skymodel into an image.

    Parameters
    ----------
    source_catalog : Path
        Source catalog path.
    reference_image : Path
        Reference image path.
    output_image : Path
        Output image path.
    """
    restore_skymodel(source_catalog, reference_image, output_image)


if __name__ == "__main__":
    descriptiontext = (
        "Restore a skymodel text file into an image.\n"
        "The restored image will have the same dimensions and WCS as the reference image."
    )

    parser = ArgumentParser(description=descriptiontext, formatter_class=RawTextHelpFormatter)
    parser.add_argument("source_catalog", help="Filename of input FITS source catalog", type=Path)
    parser.add_argument(
        "reference_image",
        help="Filename of image to use as reference for restoration",
        type=Path,
    )
    parser.add_argument(
        "output_image",
        help=(
            "Filename of output restored image. If a compressed FITS extension is used, "
            "the output will be compressed."
        ),
        type=Path,
    )
    args = parser.parse_args()
    main(args.source_catalog, args.reference_image, args.output_image)

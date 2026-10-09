"""Command-line adapter for image skymodel filtering."""

import ast
from argparse import ArgumentParser, Namespace, RawTextHelpFormatter
from typing import Optional, Sequence, Union

from rapthor.execution.image.skymodel_filter import (
    DEFAULT_ADAPTIVE_THRESH,
    DEFAULT_FILTER_BY_MASK,
    DEFAULT_NCORES,
    DEFAULT_RMSBOX,
    DEFAULT_RMSBOX_BRIGHT,
    DEFAULT_SOURCE_FINDER,
    DEFAULT_THRESHISL,
    DEFAULT_THRESHPIX,
    filter_image_skymodel,
)
from rapthor.lib.miscellaneous import string2list

RmsBox = Union[str, tuple[float, float], tuple[int, int]]


def main(
    flat_noise_image: str,
    true_sky_image: str,
    true_sky_skymodel: str,
    apparent_sky_skymodel: str,
    output_root: str,
    vertices_file: str,
    beam_ms: Sequence[str],
    *,
    bright_true_sky_skymodel: Optional[str] = None,
    threshisl: float = DEFAULT_THRESHISL,
    threshpix: float = DEFAULT_THRESHPIX,
    rmsbox: RmsBox = DEFAULT_RMSBOX,
    rmsbox_bright: RmsBox = DEFAULT_RMSBOX_BRIGHT,
    adaptive_thresh: float = DEFAULT_ADAPTIVE_THRESH,
    filter_by_mask: bool = DEFAULT_FILTER_BY_MASK,
    ncores: int = DEFAULT_NCORES,
    source_finder: str = DEFAULT_SOURCE_FINDER,
) -> None:
    """Filter an image skymodel using the execution helper."""
    filter_image_skymodel(
        flat_noise_image,
        true_sky_image,
        true_sky_skymodel,
        apparent_sky_skymodel,
        output_root,
        vertices_file,
        beam_ms,
        bright_true_sky_skymodel=bright_true_sky_skymodel,
        threshisl=threshisl,
        threshpix=threshpix,
        rmsbox=rmsbox,
        rmsbox_bright=rmsbox_bright,
        adaptive_thresh=adaptive_thresh,
        filter_by_mask=filter_by_mask,
        ncores=ncores,
        source_finder=source_finder,
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> Namespace:
    """Parse command-line arguments for skymodel filtering."""
    parser = ArgumentParser(
        description="Filter and group a sky model with an image.\n",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "flat_noise_image", help="Filename of input flat-noise image without beam correction"
    )
    parser.add_argument(
        "true_sky_image", help="Filename of input true-sky image with beam correction"
    )
    parser.add_argument("true_sky_skymodel", help="Filename of input true-sky sky model")
    parser.add_argument("apparent_sky_skymodel", help="Filename of input apparent-sky sky model")
    parser.add_argument("output_root", help="Root of output files")
    parser.add_argument("vertices_file", help="Filename of vertices file")
    parser.add_argument("beam_ms", help="MS filename(s) to use for beam attenuation")
    parser.add_argument(
        "--bright_true_sky_skymodel",
        help="Filename of input bright-source true-sky sky model",
        default=None,
    )
    parser.add_argument(
        "--threshisl",
        help="Island threshold",
        type=float,
        default=DEFAULT_THRESHISL,
    )
    parser.add_argument(
        "--threshpix",
        help="Peak pixel threshold",
        type=float,
        default=DEFAULT_THRESHPIX,
    )
    parser.add_argument(
        "--rmsbox",
        help='Rms box width and step (e.g., "(60, 20)")',
        default=str(DEFAULT_RMSBOX),
    )
    parser.add_argument(
        "--rmsbox_bright",
        help='Rms box for bright sources, width and step (e.g., "(60, 20)")',
        default=str(DEFAULT_RMSBOX_BRIGHT),
    )
    parser.add_argument(
        "--adaptive_thresh",
        help="Adaptive threshold",
        type=float,
        default=DEFAULT_ADAPTIVE_THRESH,
    )
    parser.add_argument(
        "--filter_by_mask",
        help="Filter sources by mask",
        type=ast.literal_eval,
        default=DEFAULT_FILTER_BY_MASK,
    )
    parser.add_argument(
        "--ncores",
        help="Max number of cores to use",
        type=int,
        default=DEFAULT_NCORES,
    )
    parser.add_argument(
        "--source_finder",
        help='Source finder to use, either "sofia" or "bdsf"',
        default=DEFAULT_SOURCE_FINDER,
    )
    return parser.parse_args(argv)


def run(argv: Optional[Sequence[str]] = None) -> None:
    """Run the skymodel filtering CLI adapter."""
    args = parse_args(argv)
    main(
        args.flat_noise_image,
        args.true_sky_image,
        args.true_sky_skymodel,
        args.apparent_sky_skymodel,
        args.output_root,
        args.vertices_file,
        string2list(args.beam_ms),
        bright_true_sky_skymodel=args.bright_true_sky_skymodel,
        threshisl=args.threshisl,
        threshpix=args.threshpix,
        rmsbox=args.rmsbox,
        rmsbox_bright=args.rmsbox_bright,
        adaptive_thresh=args.adaptive_thresh,
        filter_by_mask=args.filter_by_mask,
        ncores=args.ncores,
        source_finder=args.source_finder,
    )


if __name__ == "__main__":
    run()

#!/usr/bin/env python3
"""
Script to make a template image for mosaicking
"""

from argparse import ArgumentParser, RawTextHelpFormatter

from rapthor.lib import miscellaneous as misc
from rapthor.execution.mosaic.images import make_mosaic_template


def main(input_image_list, vertices_file_list, output_image, skip=False, padding=1.1):
    """
    Make a mosaic template image

    Parameters
    ----------
    input_image_list : list
        List of filenames of input images to mosaic
    vertices_file_list : list
        List of filenames of input vertices files
    output_image : str
        Filename of output image
    skip : bool
        If True, skip all processing
    padding : float
        Fraction with which to increase the final mosaic size
    """
    input_image_list = misc.string2list(input_image_list)
    vertices_file_list = misc.string2list(vertices_file_list)
    make_mosaic_template(
        input_image_list,
        vertices_file_list,
        output_image,
        skip=misc.string2bool(skip),
        padding=padding,
    )


if __name__ == "__main__":
    descriptiontext = "Make a template image for mosaicking.\n"

    parser = ArgumentParser(description=descriptiontext, formatter_class=RawTextHelpFormatter)
    parser.add_argument("input_image_list", help="Filenames of input image")
    parser.add_argument("vertices_file_list", help="Filenames of input vertices files")
    parser.add_argument("output_image", help="Filename of output template image")
    parser.add_argument("--skip", help="Skip processing", type=str, default="False")
    parser.add_argument("--padding", help="Padding factor", type=float, default=1.2)
    args = parser.parse_args()
    main(
        args.input_image_list,
        args.vertices_file_list,
        args.output_image,
        skip=args.skip,
        padding=args.padding,
    )

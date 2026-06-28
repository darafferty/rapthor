#!/usr/bin/env python3
"""Script to make a mosaic from FITS images."""

from argparse import ArgumentParser, RawTextHelpFormatter

from rapthor.lib import miscellaneous as misc
from rapthor.execution.mosaic.images import make_mosaic


def main(input_image_list, template_image, output_image, skip=False):
    """
    Make a mosaic image

    Parameters
    ----------
    input_image_list : list
        List of filenames of input FITS images to mosaic
    template_image : str
        Filename of mosaic template FITS image
    output_image : str
        Filename of output FITS image
    skip : bool
        If True, just copy input image and skip all other processing
    """
    input_image_list = misc.string2list(input_image_list)
    make_mosaic(
        input_image_list,
        template_image,
        output_image,
        skip=misc.string2bool(skip),
    )


if __name__ == "__main__":
    descriptiontext = "Make a mosaic image.\n"

    parser = ArgumentParser(description=descriptiontext, formatter_class=RawTextHelpFormatter)
    parser.add_argument("input_image_list", help="Filenames of input image")
    parser.add_argument("template_image", help="Filename of input template image")
    parser.add_argument("output_image", help="Filename of output template image")
    parser.add_argument("--skip", help="Skip processing", type=str, default="False")
    args = parser.parse_args()
    main(args.input_image_list, args.template_image, args.output_image, skip=args.skip)

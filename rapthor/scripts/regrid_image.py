#!/usr/bin/env python3
"""
Script to regrid a FITS image
"""

from argparse import ArgumentParser, RawTextHelpFormatter

from rapthor.lib import miscellaneous as misc
from rapthor.execution.mosaic.images import regrid_image


def main(input_image, template_image, vertices_file, output_image, skip=False):
    """
    Regrid a FITS image

    Parameters
    ----------
    input_image : str
        Filename of input FITS image to regrid
    template_image : str
        Filename of mosaic template FITS image
    vertices_file : str
        Filename of file with vertices
    output_image : str
        Filename of output FITS image
    skip : bool
        If True, skip all processing
    """
    regrid_image(
        input_image,
        template_image,
        vertices_file,
        output_image,
        skip=misc.string2bool(skip),
    )


if __name__ == "__main__":
    descriptiontext = "Regrid an image to match a template image.\n"

    parser = ArgumentParser(description=descriptiontext, formatter_class=RawTextHelpFormatter)
    parser.add_argument("input_image", help="Filenames of input image")
    parser.add_argument("template_image", help="Filenames of input template image")
    parser.add_argument("vertices_file", help="Filename of input vertices files")
    parser.add_argument("output_image", help="Filename of output regridded image")
    parser.add_argument("--skip", help="Skip processing", type=str, default="False")
    args = parser.parse_args()
    main(
        args.input_image,
        args.template_image,
        args.vertices_file,
        args.output_image,
        skip=args.skip,
    )

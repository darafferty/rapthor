#!/usr/bin/env python3
"""
Script to compress FITS images using fpack
"""
from argparse import ArgumentParser, RawTextHelpFormatter
import os
import subprocess
import sys


def main(image_filenames, output_directory='.'):
    """
    Compress FITS images

    Note: the filenmanes of the compressed images are constructed
    by adding the extension ".fz" to the input filenames (e.g.,
    an input of "image_1.fits" results in the output file
    "image_1.fits.fz")

    Parameters
    ----------
    image_filenames : list of str
        Filenames of the input FITS images
    output_directory : str, optional
        Path to directory where the output compressed images are
        written
    """
    # Run fpack on each input file. We do each file separately since the output filename
    # argument ("-O") can only be used with a single input file
    for input_image in image_filenames:
        output_image = os.path.join(output_directory, os.path.basename(input_image) + ".fz")
        cmd = ["fpack", "-O", output_image, input_image]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as err:
            print(err, file=sys.stderr)
            return err.returncode


if __name__ == '__main__':
    descriptiontext = "Compress FITS images.\n"

    parser = ArgumentParser(description=descriptiontext, formatter_class=RawTextHelpFormatter)
    parser.add_argument('image_filenames', nargs="+", help='Filenames of input FITS images')
    parser.add_argument('--output_directory', help='Output directory', type=str, default='.')
    args = parser.parse_args()
    main(args.image_filenames, output_directory=args.output_directory)

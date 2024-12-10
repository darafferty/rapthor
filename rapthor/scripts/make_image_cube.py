#!/usr/bin/env python3
"""
Script to make a FITS image cube
"""
from argparse import ArgumentParser, RawTextHelpFormatter
from rapthor.lib.fitsimage import FITSCube
from rapthor.lib import miscellaneous as misc


def main(input_image_filenames, output_image_filename, output_beams_filename=None,
         output_frequencies_filename=None):
    """
    Make a FITS image cube

    The associated beam and frequency values for each channel are also
    written out to files. These values can be used in PyBDSF to specify
    the beam and frequencies of the image cube.

    Parameters
    ----------
    input_image_filenames : str or list of str
        List of filenames of input images to mosaic
    output_image_filename : str
        Filename of output FITS image cube
    output_beams_filename : str, optional
        Filename of output text file with channel beams. The beams are written one
        per line as follows:
            (major axis, minor axis, position angle)
        with all values being in degrees
    output_frequencies_filename : str, optional
        Filename of output text file with channel frequencies. The frequencies are
        written one per line in Hz
    """
    input_image_filenames = misc.string2list(input_image_filenames)

    # Make the cube and write it out, along with the beams and frequencies
    image = FITSCube(input_image_filenames)
    image.write(output_image_filename)

    if output_beams_filename is None:
        output_beams_filename = output_image_filename + '_beams.txt'
    image.write_beams(output_beams_filename)

    if output_frequencies_filename is None:
        output_frequencies_filename = output_image_filename + '_frequencies.txt'
    image.write_frequencies(output_frequencies_filename)


if __name__ == '__main__':
    descriptiontext = "Make a FITS image cube.\n"

    parser = ArgumentParser(description=descriptiontext, formatter_class=RawTextHelpFormatter)
    parser.add_argument('input_image_filenames', help='Filenames of input channel images')
    parser.add_argument('output_image_filename', help='Filename of output image cube FITS file')
    parser.add_argument('--output_beams_filename', help='Filename of output cube beams text file',
                        type=str, default=None)
    parser.add_argument('--output_frequencies_filename', help='Filename of output cube frequencies text file',
                        type=str, default=None)
    args = parser.parse_args()
    main(args.input_image_filenames, args.output_image_filename,
         output_beams_filename=args.output_beams_filename,
         output_frequencies_filename=args.output_frequencies_filename)

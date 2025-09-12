#!/usr/bin/env python3
"""
Script to blank regions (with zeros or NaNs) in a fits image. Can also be used to make
a clean mask
"""
import logging
from argparse import ArgumentParser, RawTextHelpFormatter

from astropy import wcs
from astropy.io import fits as pyfits
from lsmtool.utils import rasterize
from lsmtool.io import read_vertices

from rapthor.lib import miscellaneous as misc


def main(output_image, input_image=None, vertices_file=None, reference_ra_deg=None,
         reference_dec_deg=None, cellsize_deg=None, imsize=None):
    """
    Blank a region in an image

    Parameters
    ----------
    output_image : str
        Filename of output image
    input_image : str, optional
        Filename of input image/mask to blank
    vertices_file : str, optional
        Filename of file with vertices
    reference_ra_deg : float, optional
        RA for center of output mask image
    reference_dec_deg : float, optional
        Dec for center of output mask image
    cellsize_deg : float, optional
        Size of a pixel in degrees
    imsize : int, optional
        Size of image as "xsize ysize"
    """
    if input_image is None:
        print('Input image not given. Making empty image...')
        make_blank_image = True
        if reference_ra_deg is not None and reference_dec_deg is not None:
            reference_ra_deg = float(reference_ra_deg)
            reference_dec_deg = float(reference_dec_deg)
            ximsize = int(imsize.split(',')[0])
            yimsize = int(imsize.split(',')[1])
            misc.make_template_image(output_image, reference_ra_deg, reference_dec_deg,
                                     ximsize=ximsize, yimsize=yimsize,
                                     cellsize_deg=float(cellsize_deg), fill_val=1)
        else:
            raise ValueError('ERROR: a reference position must be given to make an empty template image')
    else:
        make_blank_image = False

    if vertices_file is not None:
        # Construct polygon
        if make_blank_image:
            header = pyfits.getheader(output_image, 0)
        else:
            header = pyfits.getheader(input_image, 0)
        vertices = read_vertices(vertices_file, wcs.WCS(header))

        if make_blank_image:
            hdu = pyfits.open(output_image, memmap=False)
        else:
            hdu = pyfits.open(input_image, memmap=False)
        data = hdu[0].data

        # Rasterize the poly
        data[0, 0, :, :] = rasterize(vertices, data[0, 0, :, :])

        hdu[0].data = data
        hdu.writeto(output_image, overwrite=True)


if __name__ == '__main__':
    descriptiontext = "Blank regions of an image.\n"

    parser = ArgumentParser(description=descriptiontext, formatter_class=RawTextHelpFormatter)
    parser.add_argument('output_image_file', help='Filename of output image')
    parser.add_argument('input_image_file', help='Filename of input image', nargs='?', default=None)
    parser.add_argument('--vertices_file', help='Filename of vertices file', type=str, default=None)
    parser.add_argument('--reference_ra_deg', help='Reference RA', type=float, default=None)
    parser.add_argument('--reference_dec_deg', help='Reference Dec', type=float, default=None)
    parser.add_argument('--cellsize_deg', help='Cellsize', type=float, default=None)
    parser.add_argument('--imsize', help='Image size', type=str, default=None)
    args = parser.parse_args()
    try:
        main(args.output_image_file, args.input_image_file, vertices_file=args.vertices_file,
             reference_ra_deg=args.reference_ra_deg, reference_dec_deg=args.reference_dec_deg,
             cellsize_deg=args.cellsize_deg, imsize=args.imsize)
    except ValueError as e:
        log = logging.getLogger('rapthor:blank_image')
        log.critical(e)

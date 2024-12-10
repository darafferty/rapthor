#!/usr/bin/env python3
"""
Script to make a mosiac from FITS images
"""
from argparse import ArgumentParser, RawTextHelpFormatter
from rapthor.lib import miscellaneous as misc
from astropy.io import fits as pyfits
import numpy as np
import shutil
import os


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
    skip = misc.string2bool(skip)
    if skip:
        if os.path.exists(output_image):
            os.remove(output_image)
        shutil.copyfile(input_image_list[0], output_image)
        return

    # Load template and sector images and add them to mosaic
    regrid_hdr = pyfits.open(template_image)[0].header
    isum = pyfits.open(template_image)[0].data
    xslices = [slice(0, int(isum.shape[0] / 2.0)),
               slice(int(isum.shape[0] / 2.0), isum.shape[0])]
    yslices = [slice(0, int(isum.shape[1] / 2.0)),
               slice(int(isum.shape[1] / 2.0), isum.shape[1])]
    for xs in xslices:
        for ys in yslices:
            wsum = np.zeros_like(isum[xs, ys])
            for sector_image in input_image_list:
                r = pyfits.open(sector_image)[0].section[xs, ys]
                w = np.ones_like(r)
                w[~np.isfinite(r)] = 0
                r[~np.isfinite(r)] = 0
                isum[xs, ys] += r
                wsum += w
            isum[xs, ys] /= wsum
    del wsum, r, w
    isum[~np.isfinite(isum)] = np.nan
    hdu = pyfits.PrimaryHDU(header=regrid_hdr, data=isum)
    hdu.writeto(output_image, overwrite=True)


if __name__ == '__main__':
    descriptiontext = "Make a mosaic image.\n"

    parser = ArgumentParser(description=descriptiontext, formatter_class=RawTextHelpFormatter)
    parser.add_argument('input_image_list', help='Filenames of input image')
    parser.add_argument('template_image', help='Filename of input template image')
    parser.add_argument('output_image', help='Filename of output template image')
    parser.add_argument('--skip', help='Skip processing', type=str, default='False')
    args = parser.parse_args()
    main(args.input_image_list, args.template_image, args.output_image,
         skip=args.skip)

#!/usr/bin/env python3
"""
Script to make a ds9 region file for use with WSClean and faceting
"""
from argparse import ArgumentParser, RawTextHelpFormatter
from rapthor.lib.facet import make_ds9_region_file, read_skymodel


def main(skymodel, ra_mid, dec_mid, width_ra, width_dec, region_file):
    """
    Make a ds9 region file

    Parameters
    ----------
    skymodel : str
        Filename of calibration sky model
    ra_mid : float
        RA in degrees of bounding box center
    dec_mid : float
        Dec in degrees of bounding box center
    width_ra : float
        Width of bounding box in RA in degrees, corrected to Dec = 0
    width_dec : float
        Width of bounding box in Dec in degrees
    region_file : str
        Filename of output ds9 region file
    """
    # Read the facets from the input sky model
    facets = read_skymodel(skymodel, ra_mid, dec_mid, width_ra, width_dec)

    # Make the ds9 region file
    make_ds9_region_file(facets, region_file)


if __name__ == '__main__':
    descriptiontext = "Blank regions of an image.\n"

    parser = ArgumentParser(description=descriptiontext, formatter_class=RawTextHelpFormatter)
    parser.add_argument('skymodel', help='Filename of input sky model')
    parser.add_argument('ra_mid', help='RA of midpoint in degrees', type=float, default=None)
    parser.add_argument('dec_mid', help='Dec of midpoint in degrees', type=float, default=None)
    parser.add_argument('width_ra', help='Width in RA in degrees', type=float, default=None)
    parser.add_argument('width_dec', help='Width in Dec in degrees', type=float, default=None)
    parser.add_argument('region_file', help='Filename of output ds9 region file', type=str, default=None)
    args = parser.parse_args()
    main(args.skymodel, args.ra_mid, args.dec_mid, args.width_ra, args.width_dec, args.region_file)

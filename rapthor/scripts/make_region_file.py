#!/usr/bin/env python3
"""
Script to make a ds9 region file for use with WSClean and faceting
"""
import argparse
from argparse import RawTextHelpFormatter
from rapthor.lib import facet
from rapthor.lib import miscellaneous as misc
import lsmtool
from math import floor, ceil


def normalize_ra(num):
    """
    Normalize RA to be in the range [0, 360).

    Based on https://github.com/phn/angles/blob/master/angles.py

    Parameters
    ----------
    num : float
        The RA in degrees to be normalized.

    Returns
    -------
    res : float
        RA in degrees in the range [0, 360).
    """
    lower = 0.0
    upper = 360.0
    res = num
    if num > upper or num == lower:
        num = lower + abs(num + upper) % (abs(lower) + abs(upper))
    if num < lower or num == upper:
        num = upper - abs(num - lower) % (abs(lower) + abs(upper))
    res = lower if num == upper else num

    return res


def normalize_dec(num):
    """
    Normalize Dec to be in the range [-90, 90].

    Based on https://github.com/phn/angles/blob/master/angles.py

    Parameters
    ----------
    num : float
        The Dec in degrees to be normalized.

    Returns
    -------
    res : float
        Dec in degrees in the range [-90, 90].
    """
    lower = -90.0
    upper = 90.0
    res = num
    total_length = abs(lower) + abs(upper)
    if num < -total_length:
        num += ceil(num / (-2 * total_length)) * 2 * total_length
    if num > total_length:
        num -= floor(num / (2 * total_length)) * 2 * total_length
    if num > upper:
        num = total_length - num
    if num < lower:
        num = -total_length - num
    res = num

    return res


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
    # Set the position of the calibration patches to those of
    # the input sky model
    skymod = lsmtool.load(skymodel)
    source_dict = skymod.getPatchPositions()
    name_cal = []
    ra_cal = []
    dec_cal = []
    for k, v in source_dict.items():
        name_cal.append(k)
        # Make sure RA is between [0, 360) deg and Dec between [-90, 90]
        ra = normalize_ra(v[0].value)
        dec = normalize_dec(v[1].value)
        ra_cal.append(ra)
        dec_cal.append(dec)

    # Do the tessellation
    facet_points, facet_polys = facet.make_facet_polygons(ra_cal, dec_cal, ra_mid, dec_mid, width_ra, width_dec)
    facet_names = []
    for facet_point in facet_points:
        # For each facet, match the correct name. Some patches in the sky model may have
        # been filtered out if they lie outside the bounding box
        for ra, dec, name in zip(ra_cal, dec_cal, name_cal):
            if misc.approx_equal(ra, facet_point[0], tol=1e-6) and misc.approx_equal(dec, facet_point[1], tol=1e-6):
                # Note: some versions of ds9 have problems when there is an underscore in any of
                # the names, so replace any underscores with "-"
                facet_names.append(name.replace('_', '-'))
                break

    # Make the ds9 region file
    facet.make_ds9_region_file(facet_points, facet_polys, region_file, names=facet_names)


if __name__ == '__main__':
    descriptiontext = "Blank regions of an image.\n"

    parser = argparse.ArgumentParser(description=descriptiontext, formatter_class=RawTextHelpFormatter)
    parser.add_argument('skymodel', help='Filename of input sky model')
    parser.add_argument('ra_mid', help='RA of midpoint in degrees', type=float, default=None)
    parser.add_argument('dec_mid', help='Dec of midpoint in degrees', type=float, default=None)
    parser.add_argument('width_ra', help='Width in RA in degrees', type=float, default=None)
    parser.add_argument('width_dec', help='Width in Dec in degrees', type=float, default=None)
    parser.add_argument('region_file', help='Filename of output ds9 region file', type=str, default=None)
    args = parser.parse_args()
    main(args.skymodel, args.ra_mid, args.dec_mid, args.width_ra, args.width_dec, args.region_file)

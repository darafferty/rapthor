#!/usr/bin/env python3
"""
Script to calculate flux-scale normalization corrections
"""
import argparse
from argparse import RawTextHelpFormatter
from astropy.io import fits
from rapthor.lib import miscellaneous as misc
from astropy.coordinates import SkyCoord, match_coordinates_sky
import astropy.units as u


def main(source_catalog, ra, dec, output_h5parm, radius_cut=3.0, major_axis_cut=10/3600,
         neighbor_cut=30/3600):
    """
    Calculate flux-scale normalization corrections

    Parameters
    ----------
    source_catalog : str
        Filename of of the input FITS source catalog
    ra : float
        RA of the image center in degrees
    dec : float
        Dec of the image center in degrees
    output_h5parm : str
        Filename of the output H5parm
    radius_cut : float, optional
        Maximum radius in degrees from the image center. Sources that lie at
        radii larger than this value are excluded from the analysis
    major_axis_cut : float, optional
        Maximum major axis size in degrees. Sources with sizes larger than this
        value are excluded from the analysis
    neighbor_cut : float, optional
        Minimum distance to nearest neighbor in degrees. Sources with neighbors
        closer than this value are excluded from the analysis
    """
    # Read in the source catalog
    hdul = fits.open(source_catalog)
    data = hdul[1].data

    # Filter the sources to keep only:
    #  - sources within 3 deg of phase center
    #  - sources with major axes less than 10 arcsec
    #  - sources that have no neighbors within 30 arcsec
    source_ras = misc.normalize_ra(data['RA'])
    source_decs = misc.normalize_dec(data['DEC'])
    source_coords = SkyCoord(ra=source_ras*u.degree, dec=source_decs*u.degree)
    center_coord = SkyCoord(ra=ra*u.degree, dec=dec*u.degree)
    source_distances = [sep.value for sep in center_coord.separation(source_coords)]
    _, separation, _ = match_coordinates_sky(source_coords, source_coords, nthneighbor=2)
    neighbor_distances = [sep.value for sep in separation]

    radius_filter = source_distances < radius_cut
    major_axis_filter = data['DC_Maj'] < major_axis_cut
    neighbor_filter = neighbor_distances > neighbor_cut
    data = data[radius_filter & major_axis_filter & neighbor_filter]

    # TODO: calculate normalizations
    # TODO: write H5parm file. Note: this is a dummy file for now
    with open(output_h5parm, 'w') as f:
        f.writelines([''])


if __name__ == '__main__':
    descriptiontext = "Calculate flux-scale normalization corrections.\n"

    parser = argparse.ArgumentParser(description=descriptiontext, formatter_class=RawTextHelpFormatter)
    parser.add_argument('source_catalog', help='Filename of input FITS source catalog')
    parser.add_argument('ra', help='RA of image center in degrees', type=float)
    parser.add_argument('dec', help='Dec of image center in degrees', type=float)
    parser.add_argument('output_h5parm', help='Filename of output H5parm file with the normalization corrections')

    args = parser.parse_args()
    main(args.source_catalog, args.ra, args.dec, args.output_h5parm)

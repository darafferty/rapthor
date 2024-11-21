#!/usr/bin/env python3
"""
Script to calculate flux-scale normalization corrections
"""
import argparse
from argparse import RawTextHelpFormatter
from astropy.io import fits
from astropy.coordinates import SkyCoord, match_coordinates_sky
import astropy.units as u
import lsmtool
import numpy as np
from rapthor.lib import miscellaneous as misc
import sys


def fit_source_sed(rapthor_fluxes, rapthor_frequencies, survey_fluxes, survey_frequencies, output_frequencies):
    pass
    # Fit the external catalog fluxes to get "true" normalizations and spectral indices

    # Fit Rapthor fluxes to get observed normalizations and spectral indices

    # Derive corrections per channel needed to adjust the observed SEDs to the true ones


def main(source_catalog, ra, dec, output_h5parm, radius_cut=3.0, major_axis_cut=10/3600,
         neighbor_cut=30/3600):
    """
    Calculate flux-scale normalization corrections

    Parameters
    ----------
    source_catalog : str
        Filename of the input FITS source catalog. This catalog should be
        created by PyBDSF from an image cube with the spectral-index mode
        activated
    ra : float
        RA of the image center in degrees
    dec : float
        Dec of the image center in degrees
    output_h5parm : str
        Filename of the output H5parm
    radius_cut : float, optional
        Radius cut in degrees. Sources that lie at radii from the image
        center larger than this value are excluded from the analysis
    major_axis_cut : float, optional
        Major-axis size cut in degrees. Sources with major axes larger
        than this value are excluded from the analysis
    neighbor_cut : float, optional
        Nearest-neighbor distance cut in degrees. Sources with neighbors
        closer than this value are excluded from the analysis
    """
    # Read in the source catalog
    hdul = fits.open(source_catalog)
    data = hdul[1].data

    # Find the number of frequency channels and the total bandwidth covered
    n_chan = 0
    while True:
        freq_col = f'Freq_ch{n_chan+1}'
        if freq_col in data:
            n_chan += 1
        else:
            break
    min_frequency = np.min(data['Freq_ch1'])  # Hz
    max_frequency = np.max(data[f'Freq_ch{n_chan}'])  # Hz

    # Filter the sources to keep only:
    #  - sources within radius_cut degrees of phase center
    #  - sources with major axes less than major_axis_cut degrees
    #  - sources that have no neighbors within neighbor_cut degrees
    source_coords = SkyCoord(ra=np.array([misc.normalize_ra(source_ra)
                                          for source_ra in data['RA']])*u.degree,
                             dec=np.array([misc.normalize_dec(source_dec)
                                           for source_dec in data['DEC']])*u.degree)
    center_coord = SkyCoord(ra=ra*u.degree, dec=dec*u.degree)
    source_distances = np.array([sep.value for sep in center_coord.separation(source_coords)])
    _, separation, _ = match_coordinates_sky(source_coords, source_coords, nthneighbor=2)
    neighbor_distances = np.array([sep.value for sep in separation])

    radius_filter = source_distances < radius_cut
    major_axis_filter = data['DC_Maj'] < major_axis_cut
    neighbor_filter = neighbor_distances > neighbor_cut
    print(f"Number of sources before applying cuts: {data['RA'].size}")
    data = data[radius_filter & major_axis_filter & neighbor_filter]
    source_coords = source_coords[radius_filter & major_axis_filter & neighbor_filter]
    print(f"Number of sources after applying cuts: {data['RA'].size}")

    # Cross match sources with external catalogs
    survey_catalogs = []
    surveys = ['vlssr', 'wenss']
    frequencies = [74e6, 327e6]  # Hz
    do_normalization = True
    for survey, frequency in zip(surveys, frequencies):
        # Download sky model(s), using a 5-deg radius to ensure the field is
        # fully covered
        try:
            skymodel = lsmtool.load(survey, VOPosition=[ra, dec], VORadius=5.0)
        except (OSError, ConnectionError) as e:
            print(f'A problem occurred when downloading the {survey} catalog. '
                  'Error was: {}. Flux normalization will be skipped.'.format(e))
            do_normalization = False
        if not len(skymodel):
            print(f'No sources foundin the {survey} catalog for this field. '
                  'Flux normalization will be skipped.')
            do_normalization = False
        if not do_normalization:
            continue
        skymodel.write(f'{survey}.fits', format='fits')
        hdul = fits.open(f'{survey}.fits')
        survey_data = hdul[1].data
        survey_coords = SkyCoord(ra=np.array([misc.normalize_ra(survey_ra)
                                              for survey_ra in survey_data['RA']])*u.degree,
                                 dec=np.array([misc.normalize_dec(survey_dec)
                                               for survey_dec in survey_data['DEC']])*u.degree)

        # Cross match with the Rapthor sources
        match_ind, separation, _ = match_coordinates_sky(source_coords, survey_coords)

        # Save the flux densities
        if survey == 'wenss':
            flux_correction = 0.9
        else:
            flux_correction = 1
        survey_catalogs.append({'survey': survey, 'flux': survey_data['I'][match_ind]*flux_correction,
                                'frequency': frequency})

    output_frequencies = np.arange(min_frequency, max_frequency+1e5, 1e5)
    if do_normalization:
        # Make arrays of flux density vs. frequency for each source, for both
        # the observed fluxes and the catalog fluxes, and find the corrections
        n_sources = len(source_coords)
        corrections = np.zeros((n_sources, len(output_frequencies)))
        survey_frequencies = [sc['frequency'] for sc in survey_catalogs]  # Hz
        for i in range(n_sources):
            rapthor_fluxes = []
            rapthor_frequencies = []
            for ch_ind in range(n_chan):
                if not np.isnan(data[f'Total_flux_ch{ch_ind+1}']):
                    rapthor_fluxes.append(data[f'Total_flux_ch{ch_ind+1}'])  # Jy
                    rapthor_frequencies.append(data[f'Freq_ch{ch_ind+1}'])  # Hz
            survey_fluxes = [sc['flux'][i] for sc in survey_catalogs]  # Jy
            corrections.append(fit_source_sed(rapthor_fluxes, rapthor_frequencies, survey_fluxes,
                                              survey_frequencies, output_frequencies))

        # For each output frequency, find the average correction over all sources
        # (weighted by source flux density)
    else:
        avg_corrections = np.ones(len(output_frequencies))

    # Write corrections to the output H5parm file as amplitude corrections
    # (corrected data = data / amp^2)
    with open(output_h5parm, 'w') as f:
        f.writelines([''])


if __name__ == '__main__':
    descriptiontext = "Calculate flux-scale normalization corrections.\n"

    parser = argparse.ArgumentParser(description=descriptiontext, formatter_class=RawTextHelpFormatter)
    parser.add_argument('source_catalog', help='Filename of input FITS source catalog')
    parser.add_argument('ra', help='RA of image center in degrees', type=float)
    parser.add_argument('dec', help='Dec of image center in degrees', type=float)
    parser.add_argument('output_h5parm', help='Filename of output H5parm file with the normalization corrections')
    parser.add_argument('--radius_cut', help='Radius cut in degrees', type=float, default=3.0)
    parser.add_argument('--major_axis_cut', help='Major-axis size cut in degrees', type=float, default=10/3600)
    parser.add_argument('--neighbor_cut', help='Nearest-neighbor distance cut in degrees', type=float, default=30/3600)

    args = parser.parse_args()
    main(args.source_catalog, args.ra, args.dec, args.output_h5parm, radius_cut=args.radius_cut,
         major_axis_cut=args.major_axis_cut, neighbor_cut=args.neighbor_cut)

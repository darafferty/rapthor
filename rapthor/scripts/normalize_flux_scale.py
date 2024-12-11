#!/usr/bin/env python3
"""
Script to calculate flux-scale normalization corrections
"""
from argparse import ArgumentParser, RawTextHelpFormatter
from astropy.io import fits
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.modeling import models, fitting
import astropy.units as u
import lsmtool
import numpy as np
from rapthor.lib import miscellaneous as misc


def fit_sed(fluxes, errors, frequencies):
    """
    Fit a spectral energy distribution (SED)

    Parameters
    ----------
    fluxes : numpy array
        Array of SED flux densities in Jy
    errors : numpy array
        Array of 1-sigma errors on the SED flux densities in Jy
    frequencies : numpy array
        Array of SED frequencies in Hz for each flux density, ordered
        from low to high

    Returns
    -------
    sed_fit : function
        Fit function that returns the flux density of the SED fit at a given
        frequency in Hz
    """
    # Filter out zero fluxes
    valid_ind = fluxes > 0
    fluxes = fluxes[valid_ind]
    errors = errors[valid_ind]
    frequencies = frequencies[valid_ind]

    # Fit the SED
    if len(fluxes) < 2:
        # No fit possible, return 0
        def sed_fit_lt_2(freq):
            return 0.0
        sed_fit = sed_fit_lt_2

    elif len(fluxes) == 2:
        # Fit a simple powerlaw fit with spectral index alpha
        alpha = np.log10(fluxes[0] / fluxes[1]) / np.log10(frequencies[0] / frequencies[1])

        def sed_fit_2(freq):
            return fluxes[0] * (frequencies[0] / freq) ** -alpha
        sed_fit = sed_fit_2

    else:
        # Fit a powerlaw to the fluxes, weighted by their errors
        fitter = fitting.LMLSQFitter()
        powerlaw_init = models.Linear1D(slope=-0.8, intercept=np.log10(fluxes[0]))
        weights = [min(1e3, 1/err) if err > 0 else 1e3 for err in errors]
        powerlaw_fit = fitter(powerlaw_init, np.log10(frequencies), np.log10(fluxes), weights=weights)

        def sed_fit_gt_2(freq):
            return 10**(powerlaw_fit(np.log10(freq)))
        sed_fit = sed_fit_gt_2

    return sed_fit


def find_normalizations(rapthor_fluxes, rapthor_errors, rapthor_frequencies,
                        survey_fluxes, survey_errors, survey_frequencies,
                        output_frequencies):
    """
    Fit source spectral energy distributions (SEDs) to get flux-scale
    normalizations

    Parameters
    ----------
    rapthor_fluxes : numpy array
        Array of Rapthor source flux densities in Jy
    rapthor_errors : numpy array
        Array of 1-sigma errors on the Rapthor source flux densities in Jy
    rapthor_frequencies : numpy array
        Array of source frequencies in Hz for each Rapthor flux density
    survey_fluxes : numpy array
        Array of survey source flux densities in Jy
    survey_errors : numpy array
        Array of 1-sigma errors on the survey source flux densities in Jy
    survey_frequencies : numpy array
        Array of source frequencies in Hz for each survey flux density
    output_frequencies : numpy array
        Array of output frequencies in Hz at which corrections will be
        calculated

    Returns
    -------
    normalizations : numpy array
        Array of normalizations, one per output frequency, that will result in
        observed flux density * correction / true flux density = 1
    """
    # Fit the external survey SED
    survey_fit = fit_sed(survey_fluxes, survey_errors, survey_frequencies)

    # Fit the Rapthor SED
    rapthor_fit = fit_sed(rapthor_fluxes, rapthor_errors, rapthor_frequencies)

    # Derive normalizations per frequency needed to adjust the Rapthor SED to
    # match the survey one
    normalizations = np.array([survey_fit(freq/1e8) / rapthor_fit(freq/1e8)
                               if (survey_fit(freq/1e8) > 0 and rapthor_fit(freq/1e8) > 0)
                               else np.nan for freq in output_frequencies])

    return normalizations


def main(source_catalog, ra, dec, output_h5parm, radius_cut=3.0, major_axis_cut=10/3600,
         neighbor_cut=30/3600, spurious_match_cut=10/3600):
    """
    Calculate flux-scale normalization corrections

    Parameters
    ----------
    source_catalog : str
        Filename of the input FITS source catalog. This catalog should have been
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
    spurious_match_cut : float, optional
        Distance cut in degrees for spurious matches. Sources with matches in
        the survey catalogs with distances greater than this value are excluded
        from the analysis
    """
    # Read in the source catalog
    with fits.open(source_catalog) as hdul:
        data = hdul[1].data

    # Find the number of frequency channels and the total bandwidth covered
    n_chan = len([colname for colname in data.columns.names if 'Freq_ch' in colname])
    if n_chan == 0:
        raise ValueError('No channel frequency columns were found in the input source catalog. '
                         'Please run PyBDSF with the spectral-index mode activated.')
    min_frequency = np.nanmin(data['Freq_ch1'])  # Hz
    max_frequency = np.nanmax(data[f'Freq_ch{n_chan}'])  # Hz

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
    frequencies = [74e6, 327e6]  # Hz, ordered from low to high
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
            print(f'No sources found in the {survey} catalog for this field. '
                  'Flux normalization will be skipped.')
            do_normalization = False
        if not do_normalization:
            continue
        skymodel.write(f'{survey}.fits', format='fits', clobber=True)
        with fits.open(f'{survey}.fits') as hdul:
            survey_data = hdul[1].data
        survey_coords = SkyCoord(ra=np.array([misc.normalize_ra(survey_ra)
                                              for survey_ra in survey_data['RA']])*u.degree,
                                 dec=np.array([misc.normalize_dec(survey_dec)
                                               for survey_dec in survey_data['DEC']])*u.degree)

        # Cross match the survey sources with the Rapthor sources
        match_ind, separation, _ = match_coordinates_sky(source_coords, survey_coords)

        # Check each Rapthor source, rejecting distant matches that are likely to be
        # spurious (e.g., due to the true source not being present in the survey catalog)
        # and keeping only the closest match
        survey_fluxes = []
        for dist, ind in zip(separation,  match_ind):
            all_matches_ind = np.where(match_ind == ind)[0]
            if dist.value > np.min(separation.value[all_matches_ind]) or dist.value > spurious_match_cut:
                # Reject match by setting its survey flux to 0 (which will be ignored
                # during SED fitting)
                survey_fluxes.append(0.0)
            else:
                survey_fluxes.append(survey_data['I'][ind])

        # Save the catalog details for use in SED fitting
        if survey == 'wenss':
            flux_correction = 0.9  # adjust to Scaife and Heald (2012) flux scale
            flux_err = 3.6e-3  # Jy (reported average rms noise level)
        elif survey == 'vlssr':
            flux_correction = 1  # already on Scaife and Heald (2012) flux scale
            flux_err = 0.1  # Jy (reported average rms noise level)
        survey_catalogs.append({'survey': survey, 'flux': np.array(survey_fluxes)*flux_correction,
                                'flux_err': flux_err, 'frequency': frequency})

    output_frequencies = np.arange(min_frequency, max_frequency+1e5, 1e5)
    if do_normalization:
        # Make arrays of flux density vs. frequency for each source, for both
        # the observed fluxes and the catalog fluxes, and find the corrections
        n_sources = len(source_coords)
        corrections = np.zeros((n_sources, len(output_frequencies)))
        survey_frequencies = np.array([sc['frequency'] for sc in survey_catalogs])  # Hz
        for i in range(n_sources):
            rapthor_fluxes = []
            rapthor_errors = []
            rapthor_frequencies = []
            for ch_ind in range(n_chan):
                if not np.isnan(data[f'Total_flux_ch{ch_ind+1}'][i]):
                    rapthor_fluxes.append(data[f'Total_flux_ch{ch_ind+1}'][i])  # Jy
                    rapthor_errors.append(data[f'E_Total_flux_ch{ch_ind+1}'][i])  # Jy
                    rapthor_frequencies.append(data[f'Freq_ch{ch_ind+1}'][i])  # Hz
            rapthor_fluxes = np.array(rapthor_fluxes)
            rapthor_errors = np.array(rapthor_errors)
            rapthor_frequencies = np.array(rapthor_frequencies)
            survey_fluxes = np.array([sc['flux'][i] for sc in survey_catalogs])  # Jy
            survey_errors = np.array([sc['flux_err'] for sc in survey_catalogs])  # Jy
            corrections[i, :] = find_normalizations(rapthor_fluxes, rapthor_errors, rapthor_frequencies,
                                                    survey_fluxes, survey_errors, survey_frequencies,
                                                    output_frequencies)

        # For each output frequency, find the average correction over all sources
        # (weighted by source flux density)
        avg_corrections = np.nanmean(corrections, axis=0)
    else:
        # If normalization cannot be done, just set all corrections to 1
        avg_corrections = np.ones(len(output_frequencies))

    # TODO: [RAP-792] Write corrections to the output H5parm file as amplitude corrections
    # (corrected data = data / amp^2)
    with open(output_h5parm, 'w') as f:
        f.writelines([''])  # placeholder


if __name__ == '__main__':
    descriptiontext = "Calculate flux-scale normalization corrections.\n"

    parser = ArgumentParser(description=descriptiontext, formatter_class=RawTextHelpFormatter)
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

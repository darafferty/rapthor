#!/usr/bin/env python3
"""
Script to calculate flux-scale normalization corrections
"""
from argparse import ArgumentParser, RawTextHelpFormatter
from astropy.io import fits
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.modeling import models, fitting
import astropy.units as u
import casacore.tables as pt
from losoto.h5parm import h5parm
import lsmtool
import numpy as np
from pathlib import Path
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
        fitter = fitting.LinearLSQFitter()
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
        Array of normalization corrections, one per output frequency, that will
        result in (true flux density / observed flux density) * correction = 1
    """
    # Fit the external survey SED
    survey_fit = fit_sed(survey_fluxes, survey_errors, survey_frequencies)

    # Fit the Rapthor SED
    rapthor_fit = fit_sed(rapthor_fluxes, rapthor_errors, rapthor_frequencies)

    # Derive normalizations per frequency needed to adjust the Rapthor SED to
    # match the survey one
    normalizations = np.array([rapthor_fit(freq) / survey_fit(freq)
                               if (survey_fit(freq) > 0 and rapthor_fit(freq) > 0)
                               else np.nan for freq in output_frequencies])

    return normalizations


def create_normalization_h5parm(antenna_file, field_file, h5parm_file, frequencies,
                                normalizations, solset_name='sol000',
                                soltab_name='amp000'):
    """
    Writes normalization corrections to an H5parm file

    The corrections are written as amplitudes such that, when applied by DP3,
    the corrected data = data / amp^2 (i.e., amp = sqrt(normalizations))

    Parameters
    ----------
    antenna_file : str
        Filename of the antenna table (e.g., from a representative MS file)
    field_file : str
        Filename of the field table (e.g., from a representative MS file)
    h5parm_file : str
        Filename of the output H5parm file
    frequencies : array
        Array of frequencies corresponding to the normalization corrections
    normalizations : array
        Array of normalization corrections, one per frequency, that will result
        in (true flux density / observed flux density) * correction = 1
    solset_name : str, optional
        Name of the output solution set
    soltab_name : str, optional
        Name of the output solution table
    """
    with h5parm(h5parm_file, readonly=False) as ouput_h5parm:
        # Create the solution set
        solset = ouput_h5parm.makeSolset(solset_name)

        # Get the station info and make the output antenna table
        with pt.table(antenna_file, ack=False) as antennaTable:
            antennaNames = antennaTable.getcol('NAME')
            antennaPositions = antennaTable.getcol('POSITION')
        antennaTable = solset.obj._f_get_child('antenna')
        antennaTable.append(list(zip(*(antennaNames, antennaPositions))))

        # Get the field info and make the output source table
        with pt.table(field_file, ack=False) as fieldTable:
            phaseDir = fieldTable.getcol('PHASE_DIR')
        pointing = phaseDir[0, 0, :]
        sourceTable = solset.obj._f_get_child('source')
        sourceTable.append([('pointing', pointing)])

        # Create the output solution table
        amps = np.sqrt(normalizations)  # so that corrected data = data / normalizations
        weights = np.ones(normalizations.shape)
        soltab = solset.makeSoltab('amplitude', soltab_name, axesNames=['freq'],
                                   axesVals=[frequencies], vals=amps,
                                   weights=weights)

        # Add a CREATE entry to the solution table history
        soltab.addHistory('CREATE (by normalize_flux_scale.py)')


def main(source_catalog, ra, dec, ms_file, output_h5parm, radius_cut=3.0,
         major_axis_cut=30/3600, neighbor_cut=30/3600, spurious_match_cut=30/3600,
         min_sources=5):
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
    ms_file : str
        Filename of the MS file used for imaging (needed for antenna and field tables)
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
    min_sources : int, optional
        The minimum number of souces required for the normalization correction
        calculation
    """
    # Read in the source catalog
    with fits.open(source_catalog) as hdul:
        data = hdul[1].data

    # Find the number of frequency channels and the total bandwidth covered
    n_chan = len([colname for colname in data.columns.names if colname.startswith('Freq_ch')])
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

    # To find the distance to the nearest neighbor of each source, cross match
    # the source catalog with itself and take the second-closest match using
    # nthneighbor = 2 (the closest match, returned by nthneighbor = 1, will
    # always be each source matched to itself and hence at a distance of 0)
    _, separation, _ = match_coordinates_sky(source_coords, source_coords, nthneighbor=2)
    neighbor_distances = np.array([sep.value for sep in separation])

    # Apply the cuts
    do_normalization = True
    radius_filter = source_distances < radius_cut
    major_axis_filter = data['DC_Maj'] < major_axis_cut
    neighbor_filter = neighbor_distances > neighbor_cut
    print(f"Number of sources before applying cuts: {data['RA'].size}")
    data = data[radius_filter & major_axis_filter & neighbor_filter]
    source_coords = source_coords[radius_filter & major_axis_filter & neighbor_filter]
    n_sources = len(source_coords)
    print(f"Number of sources after applying cuts: {n_sources}")
    if n_sources < min_sources:
        print('Too few sources remain after applying cuts. Flux normalization will be skipped.')
        do_normalization = False

    # Cross match sources with external catalogs
    if do_normalization:
        survey_catalogs = []
        surveys = ['vlssr', 'wenss']  # the survey names
        frequencies = [74e6, 327e6]  # the survey reference frequencies in Hz, ordered from low to high
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

    # Fit the source SEDs to find the corrections in 0.1 MHz channels. We use 0.1 MHz as
    # it matches the typical channel width of data used with Rapthor
    #
    # TODO: Test whether a coarser grid would work (it just needs to be fine enough to
    # capture the frequency behavior of the corrections sufficiently well)
    output_frequencies = np.arange(min_frequency, max_frequency+1e5, 1e5)
    if do_normalization:
        # Make arrays of flux density vs. frequency for each source, for both
        # the observed fluxes and the catalog fluxes, and find the corrections
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
        #
        # Check the number of valid fits first, and if too few, skip the normalization.
        # This check assumes that the corrections from valid fits do not contain any NaNs
        valid_fits = np.all(~np.isnan(corrections), axis=1)
        n_valid = np.where(valid_fits)[0].size
        if n_valid < min_sources:
            print('Too few sources with successful SED fits. Flux normalization will be skipped.')
            avg_corrections = np.ones(len(output_frequencies))
        else:
            valid_corrections = corrections[valid_fits]
            if weight_by_flux:
                weights = data['Total_flux'][valid_corrections]
            else:
                weights = np.ones(n_valid)
            avg_corrections = np.average(corrections[valid_corrections], axis=0,
                                         weights=weights)
    else:
        # If normalization cannot be done, just set all corrections to 1
        avg_corrections = np.ones(len(output_frequencies))

    # Write corrections to the output H5parm file as amplitude corrections
    antenna_file = Path(ms_file).joinpath('ANTENNA')
    field_file = Path(ms_file).joinpath('FIELD')
    create_normalization_h5parm(antenna_file, field_file, output_h5parm, frequencies,
                                avg_corrections)


if __name__ == '__main__':
    descriptiontext = "Calculate flux-scale normalization corrections.\n"

    parser = ArgumentParser(description=descriptiontext, formatter_class=RawTextHelpFormatter)
    parser.add_argument('source_catalog', help='Filename of input FITS source catalog')
    parser.add_argument('ra', help='RA of image center in degrees', type=float)
    parser.add_argument('dec', help='Dec of image center in degrees', type=float)
    parser.add_argument('ms_file', help='Filename of imaging MS file')
    parser.add_argument('output_h5parm', help='Filename of output H5parm file with the normalization corrections')
    parser.add_argument('--radius_cut', help='Radius cut in degrees', type=float, default=3.0)
    parser.add_argument('--major_axis_cut', help='Major-axis size cut in degrees', type=float, default=30/3600)
    parser.add_argument('--neighbor_cut', help='Nearest-neighbor distance cut in degrees', type=float, default=30/3600)
    parser.add_argument('--spurious_match_cut', help='Spurious match distance cut in degrees', type=float, default=30/3600)
    parser.add_argument('--min_sources', help='Minimum number of souces required for normalization calculation', type=int, default=5)

    args = parser.parse_args()
    main(args.source_catalog, args.ra, args.dec, args.ms_file, args.output_h5parm,
         radius_cut=args.radius_cut, major_axis_cut=args.major_axis_cut,
         neighbor_cut=args.neighbor_cut, spurious_match_cut=args.spurious_match_cut,
         min_sources=args.min_sources)

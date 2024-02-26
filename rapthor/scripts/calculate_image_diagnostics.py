#!/usr/bin/env python3
"""
Script to calculate various image diagnostics
"""
import argparse
from argparse import RawTextHelpFormatter
import lsmtool
import numpy as np
from rapthor.lib import miscellaneous as misc
from rapthor.lib.facet import SquareFacet, filter_skymodel, read_ds9_region_file
from rapthor.lib.fitsimage import FITSImage
from rapthor.lib.observation import Observation
import casacore.tables as pt
from astropy.utils import iers
from astropy.table import Table
import astropy.units as u
from astropy.coordinates import SkyCoord, Angle
from astropy.coordinates import match_coordinates_sky
from astropy.stats import sigma_clip
import json
import requests
import time


# Turn off astropy's IERS downloads to fix problems in cases where compute
# node does not have internet access
iers.conf.auto_download = False


def download_panstarrs(ra, dec, radius, output_filename, max_tries=5):
    """
    Downloads a catalog of Pan-STARRS sources around the given position

    Parameters
    ----------
    ra : float
        RA of cone search center in deg
    dec : float
        Dec of cone search center in deg
    radius : float
        Radius of cone in deg (must be <= 0.5)
    output_filename : str
        Filename for the output catalog
    max_tries : int, optional
        Maximum number of download tries to do if problems such as timeouts are
        encountered

    Returns
    -------
    output_catalog : str
        Filename of the output makesourcedb catalog of Pan-STARRS matches
    """
    if radius > 0.5:
        raise ValueError('The search radius must be <= 0.5 deg')

    # Construct the URL and search parameters
    baseurl = 'https://catalogs.mast.stsci.edu/api/v0.1/panstarrs'
    release = 'dr1'  # the release with the mean data
    table = 'mean'  # the main catalog, with the mean data
    cat_format = 'csv'
    url = f'{baseurl}/{release}/{table}.{cat_format}'
    search_params = {'ra': ra,
                     'dec': dec,
                     'radius': radius,
                     'nDetections.min': '5',  # require detection in at least 5 epochs
                     'columns': ['objID', 'ramean', 'decmean']  # get only the info we need
                     }

    # Download the catalog
    for tries in range(1, 1 + max_tries):
        timedout = False
        try:
            result = requests.get(url, params=search_params, timeout=300)
        except requests.exceptions.Timeout:
            timedout = True
        if timedout or not result.ok:
            if tries == max_tries:
                raise IOError('Download failed after {} attempts.'.format(max_tries))
            else:
                print('Attempt #{0:d} failed. Attempting {1:d} more times.'.format(tries, max_tries - tries))
                time.sleep(5)

    # Convert the catalog to makesourcedb format and write to the output file
    lines = result.text.split('\n')[1:]  # split and remove header line
    for line in lines:
        # Add dummy entries for type and Stokes I flux density
        line += ',POINT,0.0'
    header = 'FORMAT = Name, Ra, Dec, Type, I'

    with open(output_filename, 'w') as f:
        f.writelines(header)
        f.writelines(lines)


def cross_match(coords_1, coords_2, radius=0.1):
    """
    Returns the matches of two coordinate arrays

    Parameters
    ----------
    coords_1 : np.array
        Array of [RA, Dec] values in deg
    coords_2 : np.array
        Array of [RA, Dec] values in deg
    radius : float or str, optional
        Radius in degrees (if float) or 'value unit' (if str; e.g., '30 arcsec')

    Returns
    -------
    matches1, matches2 : np.array, np.array
        matches1 is the array of indices of coords_1 that have matches in coords_2
        within the specified radius. matches2 is the array of indices of coords_2
        for the same sources.

    """
    # Do the cross matching
    catalog1 = SkyCoord(coords_1[0], coords_1[1], unit=(u.degree, u.degree))
    catalog2 = SkyCoord(coords_2[0], coords_2[1], unit=(u.degree, u.degree))
    idx, d2d, d3d = match_coordinates_sky(catalog1, catalog2)

    # Remove matches outside given radius
    try:
        radius = '{0} degree'.format(float(radius))
    except ValueError:
        pass
    radius = Angle(radius).degree
    matches1 = np.where(d2d.value <= radius)[0]
    matches2 = idx[matches1]

    # Select closest match only
    filter = []
    for i in range(len(matches1)):
        mind = np.where(matches2 == matches2[i])[0]
        nMatches = len(mind)
        if nMatches > 1:
            mradii = d2d.value[matches1][mind]
            if d2d.value[matches1][i] == np.min(mradii):
                filter.append(i)
        else:
            filter.append(i)
    matches1 = matches1[filter]
    matches2 = matches2[filter]

    return matches1, matches2


def main(flat_noise_image, flat_noise_rms_image, true_sky_image, true_sky_rms_image,
         input_catalog, input_skymodel, obs_ms, diagnostics_file, output_root,
         facet_region_file=None, photometry_comparison_skymodel=None,
         astrometry_comparison_skymodel=None):
    """
    Calculate various image diagnostics

    Parameters
    ----------
    flat_noise_image : str
        Filename of the flat-noise image
    flat_noise_rms_image : str
        Filename of the background RMS image derived from the flat-noise image
    true_sky_image : str
        Filename of the true-sky image
    true_sky_rms_image : str
        Filename of the background RMS image derived from the true-sky image
    input_catalog : str
        Filename of the input PyBDSF FITS catalog derived from the LOFAR image
    input_skymodel : str
        Filename of input sky model produced during imaging
    obs_ms : list of str
        List of MS files to use to derive the theoretical image noise and
        other properties of the observation
    diagnostics_file : str
        Filename of the input JSON file containing image diagnostics derived
        by the sky model filtering script
    output_root : str
        Root of the filename for the output files
    facet_region_file : str, optional
        Filename of the facet region file (in ds9 format) that defines the
        facets used in imaging
    photometry_comparison_skymodel : str, optional
        Filename of the sky model to use for the photometry (flux scale)
        comparison (in makesourcedb format). If not given, a TGSS model is
        downloaded
    astrometry_comparison_skymodel : str, optional
        Filename of the sky model to use for the astrometry comparison (in
        makesourcedb format). If not given, a Pan-STARRS model is downloaded
    """
    # Select the best MS
    if isinstance(obs_ms, str):
        obs_ms = misc.string2list(obs_ms)
    if len(obs_ms) > 1:
        ms_times = []
        for ms in obs_ms:
            tab = pt.table(ms, ack=False)
            ms_times.append(np.mean(tab.getcol('TIME')))
            tab.close()
        ms_times_sorted = sorted(ms_times)
        mid_time = ms_times_sorted[int(len(ms_times)/2)]
        beam_ind = ms_times.index(mid_time)
    else:
        beam_ind = 0

    # Read in the images and diagnostics
    img_flat_noise = FITSImage(flat_noise_image)
    rms_img_flat_noise = FITSImage(flat_noise_rms_image)
    img_true_sky = FITSImage(true_sky_image)
    rms_img_true_sky = FITSImage(true_sky_rms_image)
    with open(diagnostics_file, 'r') as fp:
        cwl_output = json.load(fp)

    # Collect some diagnostic numbers from the images. Note: we ensure all
    # non-integer numbers are float, as, e.g., np.float32 is not supported by json.dump()
    obs_list = [Observation(ms) for ms in obs_ms]
    theoretical_rms, unflagged_fraction = misc.calc_theoretical_noise(obs_list)  # Jy/beam
    dynamic_range_global_true_sky = float(img_true_sky.max_value / rms_img_true_sky.min_value)
    dynamic_range_local_true_sky = float(np.nanmax(rms_img_flat_noise.img_data / rms_img_true_sky.img_data))
    dynamic_range_global_flat_noise = float(img_flat_noise.max_value / rms_img_flat_noise.min_value)
    dynamic_range_local_flat_noise = float(np.nanmax(img_flat_noise.img_data / rms_img_flat_noise.img_data))
    cwl_output.update({'theoretical_rms': theoretical_rms,
                       'unflagged_data_fraction': unflagged_fraction,
                       'min_rms_true_sky': rms_img_true_sky.min_value,
                       'max_rms_true_sky': rms_img_true_sky.max_value,
                       'mean_rms_true_sky': rms_img_true_sky.mean_value,
                       'median_rms_true_sky': rms_img_true_sky.median_value,
                       'dynamic_range_global_true_sky': dynamic_range_global_true_sky,
                       'dynamic_range_local_true_sky': dynamic_range_local_true_sky,
                       'min_rms_flat_noise': rms_img_flat_noise.min_value,
                       'max_rms_flat_noise': rms_img_flat_noise.max_value,
                       'mean_rms_flat_noise': rms_img_flat_noise.mean_value,
                       'median_rms_flat_noise': rms_img_flat_noise.median_value,
                       'dynamic_range_global_flat_noise': dynamic_range_global_flat_noise,
                       'dynamic_range_local_flat_noise': dynamic_range_local_flat_noise,
                       'freq': img_true_sky.freq,
                       'beam_fwhm': img_true_sky.beam})

    # Load input sky model and photometry comparison model
    s_in = lsmtool.load(input_skymodel)
    if photometry_comparison_skymodel is None:
        # Download a TGSS sky model around the midpoint of the input sky model,
        # using a 5-deg radius to ensure the field is fully covered
        _, _, midRA, midDec = s_in._getXY()
        try:
            s_comp_photometry = lsmtool.load('tgss', VOPosition=[midRA, midDec], VORadius=5.0)
        except OSError:
            # Comparison catalog not downloaded successfully
            s_comp_photometry = None
    else:
        s_comp_photometry = lsmtool.load(photometry_comparison_skymodel)

    # Do the photometry check
    if s_comp_photometry and s_in:
        # Write the comparison catalog to FITS file for use with astropy
        catalog_comp_filename = output_root + '.comparison.fits'
        s_comp_photometry.table.columns['Ra'].format = None
        s_comp_photometry.table.columns['Dec'].format = None
        s_comp_photometry.table.columns['I'].format = None
        s_comp_photometry.write(catalog_comp_filename, format='fits', clobber=True)
        if 'ReferenceFrequency' in s_comp_photometry.getColNames():
            freq_comp = np.mean(s_comp_photometry.getColValues('ReferenceFrequency'))
        else:
            freq_comp = s_comp_photometry.table.meta['ReferenceFrequency']

        # Read in PyBDSF-derived and comparison catalogs and filter to keep
        # only comparison sources within a radius of FWHM / 2 of phase center
        catalog = Table.read(input_catalog, format='fits')
        catalog_comp = Table.read(catalog_comp_filename, format='fits')
        obs = obs_list[beam_ind]
        phase_center = SkyCoord(ra=obs.ra*u.degree, dec=obs.dec*u.degree)
        coords_comp = SkyCoord(ra=catalog_comp['Ra'], dec=catalog_comp['Dec'])
        separation = phase_center.separation(coords_comp)
        sec_el = 1.0 / np.sin(obs.mean_el_rad)
        fwhm_deg = 1.1 * ((3.0e8 / img_true_sky.freq) / obs.diam) * 180 / np.pi * sec_el
        catalog_comp = catalog_comp[separation < fwhm_deg/2*u.degree]

        # Cross match the coordinates and keep only matches that have a
        # separation of 5 arcsec or less
        #
        # Note: we require at least 10 sources for the comparison, as using
        # fewer can give unreliable estimates
        min_number = 10
        if len(catalog_comp) >= min_number:
            coords = SkyCoord(ra=catalog['RA'], dec=catalog['DEC'])
            coords_comp = SkyCoord(ra=catalog_comp['Ra'], dec=catalog_comp['Dec'])
            idx, sep2d, _ = match_coordinates_sky(coords_comp, coords)
            constraint = sep2d < 5*u.arcsec
            catalog_comp = catalog_comp[constraint]
            catalog = catalog[idx[constraint]]

            # Find the mean flux ratio (input / comparison). We use the total island
            # flux density from PyBDSF as it gives less scatter than the Gaussian
            # fluxes when comparing to a lower-resolution catalog (such as TGSS).
            # Lastly, a correction factor is used to correct for the potentially
            # different frequencies of the LOFAR image and the comparison catalog
            if len(catalog_comp) >= min_number:
                alpha = -0.7  # adopt a typical spectral index
                freq_factor = (freq_comp / img_true_sky.freq)**alpha
                ratio = catalog['Isl_Total_flux'] / catalog_comp['I'] * freq_factor
                meanRatio = np.mean(ratio)
                stdRatio = np.std(ratio)
                clippedRatio = sigma_clip(ratio)
                meanClippedRatio = np.mean(clippedRatio)
                stdClippedRatio = np.std(clippedRatio)
                stats = {'meanRatio_pybdsf': meanRatio,
                         'stdRatio_pybdsf': stdRatio,
                         'meanClippedRatio_pybdsf': meanClippedRatio,
                         'stdClippedRatio_pybdsf': stdClippedRatio}
                cwl_output.update(stats)

    # Do the astrometry check by comparing to Pan-STARRS positions
    max_search_cone_radius = 0.5  # deg; Pan-STARRS search limit
    if facet_region_file is not None:
        facets = read_ds9_region_file(facet_region_file)
    else:
        # Use a single rectangular facet centered on the image
        ra = image_ra
        dec = image_dec
        width = min(max_search_cone_radius*2, image_width)
        facets = [SquareFacet('field', ra, dec, width)]

    # Loop over the facets, performing the astrometry checks for each
    if astrometry_comparison_skymodel:
        s_comp_astrometry = lsmtool.load(astrometry_comparison_skymodel)
    else:
        s_comp_astrometry = None
    for facet in facets:
        if s_comp_astrometry is None:
            try:
                output_filename = f'{facet.name}.panstarrs.txt'
                download_panstarrs(facet.ra, facet.dec, min(max_search_cone_radius, facet.size/2),
                                   output_filename)
                s_comp_astrometry = lsmtool.load(output_filename)
            except IOError:
                # Comparison catalog not downloaded successfully
                continue
        facet_skymodel = filter_skymodel(facet.polygon, s_in.copy(), facet.wcs)

        # Check if there is a sufficient number of sources to do the comparison with.
        # If there is, do it and append the resulting diagnostics dict to the
        # existing one
        #
        # Note: the various ratios are all calculated as (s_in / s_comp) and the
        # differences as (s_in - s_comp). If there are no successful matches,
        # the compare() method returns None
        if s_comp_astrometry and len(s_comp_astrometry) >= min_number:
            flux_astrometry_diagnostics = facet_skymodel.compare(s_comp_astrometry,
                                                                 radius='5 arcsec',
                                                                 excludeMultiple=True,
                                                                 make_plots=False)
            if flux_astrometry_diagnostics is not None:
                cwl_output.update(flux_astrometry_diagnostics)

    # Write out the diagnostics
    with open(output_root+'.image_diagnostics.json', 'w') as fp:
        json.dump(cwl_output, fp)


if __name__ == '__main__':
    descriptiontext = "Analyze source catalogs.\n"

    parser = argparse.ArgumentParser(description=descriptiontext, formatter_class=RawTextHelpFormatter)
    parser.add_argument('flat_noise_image', help='Filename of flat-noise FITS image')
    parser.add_argument('flat_noise_rms_image', help='Filename of flat-noise FITS image')
    parser.add_argument('true_sky_image', help='Filename of flat-noise FITS image')
    parser.add_argument('true_sky_rms_image', help='Filename of flat-noise FITS image')
    parser.add_argument('input_catalog', help='Filename of input PyBDSF FITS catalog')
    parser.add_argument('input_skymodel', help='Filename of input sky model')
    parser.add_argument('obs_ms', help='Filename of observation MS')
    parser.add_argument('diagnostics_file', help='Filename of diagnostics JSON file')
    parser.add_argument('output_root', help='Root of output files')

    args = parser.parse_args()
    main(args.flat_noise_image, args.flat_noise_rms_image, args.true_sky_image, args.true_sky_rms_image,
         args.input_catalog, args.input_skymodel, args.obs_ms, args.diagnostics_file, args.output_root)

#!/usr/bin/env python3
"""
Script to calculate various image diagnostics
"""
import argparse
from argparse import RawTextHelpFormatter
import lsmtool
import numpy as np
from rapthor.lib import miscellaneous as misc
from rapthor.lib.fitsimage import FITSImage
from rapthor.lib.observation import Observation
import casacore.tables as pt
from astropy.utils import iers
from astropy.table import Table
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import match_coordinates_sky
from astropy.stats import sigma_clip
from astropy.io.registry import IORegistryError
import json


# Turn off astropy's IERS downloads to fix problems in cases where compute
# node does not have internet access
iers.conf.auto_download = False


def main(flat_noise_image, flat_noise_rms_image, true_sky_image, true_sky_rms_image,
         input_catalog, input_skymodel, obs_ms, diagnostics_file, output_root,
         comparison_skymodel=None):
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
    comparison_skymodel : str, optional
        Filename of the sky model to use for flux scale and astrometry
        comparisons. If not given, a TGSS model is downloaded
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
    theoretical_rms, unflagged_fraction = misc.calc_theoretical_noise(obs_ms)  # Jy/beam
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

    # Load input sky model and comparison model
    s_in = lsmtool.load(input_skymodel)
    if comparison_skymodel is None:
        # Download a TGSS sky model around the midpoint of the input sky model,
        # using a 5-deg radius to ensure the field is fully covered
        _, _, midRA, midDec = s_in._getXY()
        try:
            s_comp = lsmtool.load('tgss', VOPosition=[midRA, midDec], VORadius=5.0)
        except ConnectionError:
            # Comparison catalog not downloaded successfully
            s_comp = None
    else:
        s_comp = lsmtool.load(comparison_skymodel)

    # Do cross matching of the comparison model with the PyBDSF-derived one
    if s_comp and s_in:
        # Write the comparison catalog to FITS file for use with astropy
        catalog_comp_filename = output_root+'.comparison.fits'
        s_comp.table.columns['Ra'].format = None
        s_comp.table.columns['Dec'].format = None
        s_comp.table.columns['I'].format = None
        s_comp.write(catalog_comp_filename, format='fits', clobber=True)
        if 'ReferenceFrequency' in s_comp.getColNames():
            freq_comp = np.mean(s_comp.getColValues('ReferenceFrequency'))
        else:
            freq_comp = s_comp.table.meta['ReferenceFrequency']

        # Read in PyBDSF-derived and comparison catalogs and filter to keep
        # only comparison sources within a radius of FWHM / 2 of phase center
        catalog = Table.read(input_catalog, format='fits')
        catalog_comp = Table.read(catalog_comp_filename, format='fits')
        obs = Observation(obs_ms[beam_ind])
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

        # Group the comparison sky model into sources using a FWHM of 40 arcsec,
        # the approximate TGSS resolution
        s_comp.group('threshold', FWHM='40.0 arcsec', threshold=0.05)

        # Keep POINT-only sources
        source_type = s_comp.getColValues('Type')
        patch_names = s_comp.getColValues('Patch')
        non_point_patch_names = set(patch_names[np.where(source_type != 'POINT')])
        ind = []
        for patch_name in non_point_patch_names:
            ind.extend(s_comp.getRowIndex(patch_name))
        s_comp.remove(np.array(ind))

        # Check if there is a sufficient number of sources to do the comparison with.
        # If there is, do it and append the resulting diagnostics dict to the
        # existing one
        #
        # Note: the various ratios are all calculated as (s_in / s_comp) and the
        # differences as (s_in - s_comp). If there are no successful matches,
        # the compare() method returns None
        if (s_comp
                and len(s_in.getPatchNames()) >= min_number
                and len(s_comp.getPatchNames()) >= min_number):
            flux_astrometry_diagnostics = s_in.compare(s_comp, radius='5 arcsec',
                                                       excludeMultiple=True, make_plots=False)
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

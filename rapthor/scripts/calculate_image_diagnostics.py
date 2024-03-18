#!/usr/bin/env python3
"""
Script to calculate various image diagnostics
"""
import argparse
from argparse import RawTextHelpFormatter
import lsmtool
import numpy as np
from rapthor.lib import miscellaneous as misc
from rapthor.lib.facet import SquareFacet, read_ds9_region_file, make_wcs, radec2xy
from rapthor.lib.fitsimage import FITSImage
from rapthor.lib.observation import Observation
import casacore.tables as pt
from astropy.utils import iers
from astropy.table import Table
import astropy.units as u
from astropy.coordinates import SkyCoord
import json
from astropy.visualization.wcsaxes import WCSAxes
import tempfile
import matplotlib
if matplotlib.get_backend() != 'Agg':
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import glob
import shutil


# Turn off astropy's IERS downloads to fix problems in cases where compute
# node does not have internet access
iers.conf.auto_download = False


def plot_astrometry_offsets(facets, field_ra, field_dec, output_file, plot_labels=False):
    """
    Plots the astrometry offsets across the field

    Note: The arrows indicate the direction and magnitude of the
    corrections that should be applied to the LOFAR positions in
    each facet to obtain a mean offset of zero

    Parameters
    ----------
    facets : list
        List of Facet objects
    field_ra : float
        RA in degrees of the field
    field_dec : float
        Dec in degrees of the field
    output_file : str
        Filename for the output plot
    plot_labels : bool, optional
        If True, plot the facet labels
    """
    wcs = make_wcs(field_ra, field_dec)
    ra_offsets = []
    dec_offsets = []
    facet_patches = []
    facet_ra = []
    facet_dec = []
    facet_names = []
    for facet in facets:
        # Note: the offsets are calculated as (LOFAR model value) - (comparison model
        # value); e.g., a positive Dec offset indicates that the LOFAR sources
        # lie on average to the North of the comparison source positions. Therefore, we
        # multiply by -1 to obtain the correct arrow directions in the quiver plot
        if facet.astrometry_diagnostics:
            ra_offsets.append(-1 * facet.astrometry_diagnostics['meanClippedRAOffsetDeg'])
            dec_offsets.append(-1 * facet.astrometry_diagnostics['meanClippedDecOffsetDeg'])
            facet_ra.append(facet.ra)
            facet_dec.append(facet.dec)
            facet_names.append(facet.name)
            facet_patches.append(facet.get_matplotlib_patch(wcs=wcs))

    # Set up the figure
    fig = plt.figure(1, figsize=(7.66, 7))
    plt.clf()
    ax = WCSAxes(fig, [0.16, 0.1, 0.8, 0.8], wcs=wcs)
    fig.add_axes(ax)
    RAAxis = ax.coords['ra']
    RAAxis.set_axislabel('RA', minpad=0.75)
    RAAxis.set_major_formatter('hh:mm:ss')
    DecAxis = ax.coords['dec']
    DecAxis.set_axislabel('Dec', minpad=0.75)
    DecAxis.set_major_formatter('dd:mm:ss')
    ax.coords.grid(color='black', alpha=0.5, linestyle='solid')
    ax.set_title('Positional Offsets (arrows indicate direction and magnitude of correction)')

    # Plot the facet polygons
    x, y = radec2xy(wcs, facet_ra, facet_dec)
    for i, patch in enumerate(facet_patches):
        ax.add_patch(patch)
        if plot_labels:
            ax.annotate(facet_names[i], (x[i], y[i]), va='top', ha='center')

    # Plot the offsets. The arrows indicate the direction and magnitude of the
    # correction that should be applied to the LOFAR positions
    x_offsets = np.array(ra_offsets) / wcs.wcs.cdelt[0]
    y_offsets = np.array(dec_offsets) / wcs.wcs.cdelt[1]
    quiver_plot = plt.quiver(x, y, x_offsets, y_offsets, units='xy', angles='xy',
                             scale=2/3600, color='blue')
    plt.quiverkey(quiver_plot, 0.1, 0.95, 1/3600/wcs.wcs.cdelt[0], '1 arcsec',
                  labelpos='N', coordinates='figure')
    plt.savefig(output_file, format='pdf')


def fits_to_makesourcedb(catalog, reference_freq, flux_colname='Isl_Total_flux'):
    """
    Converts a PyBDSF catalog to a makesourcedb sky model

    Note: the resulting makesourcedb catalog is a minimal catalog suitable for
    use in photometry and astrometry comparisons and should not be used for
    calibration

    Parameters
    ----------
    catalog : astropy Table object
        Input PyBDSF catalog

    Returns
    -------
    skymodel : LSMTool skymodel object
        The makesourcedb sky model
    """
    # Convert the result to makesourcedb format and write to a tempfile
    out_lines = [f'FORMAT = Name, Type, Ra, Dec, I, ReferenceFrequency={reference_freq}\n']
    for (name, ra, dec, flux) in zip(catalog['Source_id'], catalog['RA'],
                                     catalog['DEC'], catalog[flux_colname]):
        out_lines.append(f'{name}, POINT, {ra}, {dec}, {flux}, \n')

    skymodel_file = tempfile.NamedTemporaryFile()
    with open(skymodel_file.name, 'w') as f:
        f.writelines(out_lines)
    skymodel = lsmtool.load(skymodel_file.name)

    return skymodel


def check_photometry(obs, input_catalog, freq, min_number, comparison_skymodel=None):
    """
    Calculate and plot various photometry diagnostics

    Parameters
    ----------
    obs : Observation object
        Representative observation, used to derive pointing, etc.
    input_catalog : str
        Filename of the input PyBDSF FITS catalog derived from the LOFAR image
    freq : float
        Frequency in Hz of the LOFAR image
    min_number : int
        Minimum number of matched sources required for the comparisons
    comparison_skymodel : str, optional
        Filename of the sky model to use for the photometry (flux scale)
        comparison (in makesourcedb format). If not given, a TGSS model is
        downloaded

    Returns
    -------
    photometry_diagnostics : dict
        Photometry diagnositcs
    """
    # Load photometry comparison model
    if comparison_skymodel:
        try:
            s_comp_photometry = lsmtool.load(comparison_skymodel)
            name_comp_photometry = 'User supplied catalog'
        except OSError as e:
            # Comparison catalog not loaded successfully
            s_comp_photometry = None
            print('Comparison sky model could not be loaded. Error was: {}. Trying default '
                  'sky model instead...'.format(e))
    else:
        s_comp_photometry = None
    if s_comp_photometry is None:
        try:
            # Download a TGSS sky model around the phase center, using a 5-deg radius
            # to ensure the field is fully covered
            s_comp_photometry = lsmtool.load('tgss', VOPosition=[obs.ra, obs.dec], VORadius=5.0)
            name_comp_photometry = 'TGSS'
        except OSError as e:
            # Comparison catalog not loaded successfully
            print('Comparison sky model could not be loaded. Error was: {}. Skipping photometry '
                  'check...'.format(e))
            s_comp_photometry = None
            photometry_diagnostics = None

    # Do the photometry check
    if s_comp_photometry:
        # Filter the input PyBDSF FITS catalog as needed for the photometry check
        # Sources are filtered to keep only those that:
        #   - lie within the FWHM of the primary beam (to exclude sources with
        #     uncertain primary beam corrections)
        #   - have deconvolved major axis < 10 arcsec (to exclude extended sources
        #     that may be poorly modeled)
        catalog = Table.read(input_catalog, format='fits')
        phase_center = SkyCoord(ra=obs.ra*u.degree, dec=obs.dec*u.degree)
        coords_comp = SkyCoord(ra=catalog['RA'], dec=catalog['DEC'])
        separation = phase_center.separation(coords_comp)
        sec_el = 1.0 / np.sin(obs.mean_el_rad)
        fwhm_deg = 1.1 * ((3.0e8 / freq) / obs.diam) * 180 / np.pi * sec_el
        catalog = catalog[separation < fwhm_deg/2*u.degree]
        major_axis = catalog['DC_Maj']  # degrees
        catalog = catalog[major_axis < 10/3600]

        # Convert the filtered catalog to a minimal sky model for use with LSMTool
        # and do the comparison
        s_pybdsf = fits_to_makesourcedb(catalog, freq)
        s_comp_photometry.group('every')
        if len(s_pybdsf) >= min_number:
            photometry_diagnostics = s_pybdsf.compare(s_comp_photometry, radius='5 arcsec',
                                                      excludeMultiple=True, make_plots=True,
                                                      name1='LOFAR', name2=name_comp_photometry)
        else:
            photometry_diagnostics = None
            print(f'Fewer than {min_number} sources found in the LOFAR image meet '
                  'the photometry cuts (major axis < 10" and located inside the FWHM '
                  'of the primary beam"). Skipping photometry check...')

    return photometry_diagnostics


def check_astrometry(obs, input_catalog, image, facet_region_file, min_number,
                     output_root, comparison_skymodel=None):
    """
    Calculate and plot various astrometry diagnostics

    Parameters
    ----------
    obs : Observation object
        Representative observation, used to derive pointing, etc.
    input_catalog : str
        Filename of the input PyBDSF FITS catalog derived from the LOFAR image
    image : FITSImage object
        The LOFAR image, used to derive frequency, coverage, etc.
    facet_region_file : str, optional
        Filename of the facet region file (in ds9 format) that defines the
        facets used in imaging
    min_number : int
        Minimum number of matched sources required for the comparisons
    output_root : str
        Root of the filename for the output files
    comparison_skymodel : str, optional
        Filename of the sky model to use for the photometry (flux scale)
        comparison (in makesourcedb format). If not given, a Pan-STARRS model is
        downloaded

    Returns
    -------
    astrometry_diagnostics : dict
        Astrometry diagnositcs
    """
    # Load and filter the input PyBDSF FITS catalog as needed for the astrometry check
    # Sources are filtered to keep only those that:
    #   - have deconvolved major axis < 10 arcsec (to exclude extended sources
    #     that may be poorly modeled)
    #   - have errors on RA and Dec of < 2 arcsec (to exclude sources
    #     with high positional uncertainties)
    catalog = Table.read(input_catalog, format='fits')
    major_axis = catalog['DC_Maj']  # degrees
    catalog = catalog[major_axis < 10/3600]
    ra_error = catalog['E_RA']  # degrees
    catalog = catalog[ra_error < 2/3600]
    dec_error = catalog['E_DEC']  # degrees
    catalog = catalog[dec_error < 2/3600]

    # Do the astrometry check
    if len(catalog) >= min_number:
        max_search_cone_radius = 0.5  # deg; Pan-STARRS search limit
        if facet_region_file is not None and os.path.isfile(facet_region_file):
            facets = read_ds9_region_file(facet_region_file)
        else:
            # Use a single rectangular facet centered on the phase center
            ra = obs.ra
            dec = obs.dec
            image_width = max(image.img_data.shape[-2:]) * abs(image.img_hdr['CDELT1'])
            width = min(max_search_cone_radius*2, image_width)
            facets = [SquareFacet('field', ra, dec, width)]

        # Convert the filtered catalog into a minimal sky model for use with LSMTool
        s_pybdsf = fits_to_makesourcedb(catalog, image.freq)

        # Loop over the facets, performing the astrometry checks for each
        if comparison_skymodel:
            try:
                s_comp_astrometry = lsmtool.load(comparison_skymodel)
                s_comp_astrometry.group('every')
            except OSError as e:
                # Comparison catalog not loaded successfully
                s_comp_astrometry = None
                print('Comparison sky model could not be loaded. Error was: {}. Trying default '
                      'sky model instead...'.format(e))
        else:
            s_comp_astrometry = None

        astrometry_diagnostics = {'facet_name': [],
                                  'meanRAOffsetDeg': [],
                                  'stdRAOffsetDeg': [],
                                  'meanClippedRAOffsetDeg': [],
                                  'stdClippedRAOffsetDeg': [],
                                  'meanDecOffsetDeg': [],
                                  'stdDecOffsetDeg': [],
                                  'meanClippedDecOffsetDeg': [],
                                  'stdClippedDecOffsetDeg': []}
        for facet in facets:
            facet.set_skymodel(s_pybdsf.copy())
            facet.find_astrometry_offsets(s_comp_astrometry, min_number=min_number)
            if len(facet.astrometry_diagnostics):
                astrometry_diagnostics['facet_name'].append(facet.name)
                astrometry_diagnostics['meanRAOffsetDeg'].append(facet.astrometry_diagnostics['meanRAOffsetDeg'])
                astrometry_diagnostics['stdRAOffsetDeg'].append(facet.astrometry_diagnostics['stdRAOffsetDeg'])
                astrometry_diagnostics['meanClippedRAOffsetDeg'].append(facet.astrometry_diagnostics['meanClippedRAOffsetDeg'])
                astrometry_diagnostics['stdClippedRAOffsetDeg'].append(facet.astrometry_diagnostics['stdClippedRAOffsetDeg'])
                astrometry_diagnostics['meanDecOffsetDeg'].append(facet.astrometry_diagnostics['meanDecOffsetDeg'])
                astrometry_diagnostics['stdDecOffsetDeg'].append(facet.astrometry_diagnostics['stdDecOffsetDeg'])
                astrometry_diagnostics['meanClippedDecOffsetDeg'].append(facet.astrometry_diagnostics['meanClippedDecOffsetDeg'])
                astrometry_diagnostics['stdClippedDecOffsetDeg'].append(facet.astrometry_diagnostics['stdClippedDecOffsetDeg'])

        # Save and plot the per-facet offsets
        if len(astrometry_diagnostics['facet_name']):
            with open(output_root+'.astrometry_offsets.json', 'w') as fp:
                json.dump(astrometry_diagnostics, fp)
            ra = obs.ra
            dec = obs.dec
            plot_astrometry_offsets(facets, ra, dec, output_root+'.astrometry_offsets.pdf')

            # Calculate mean offsets
            mean_astrometry_diagnostics = {'meanRAOffsetDeg': np.mean(astrometry_diagnostics['meanRAOffsetDeg']),
                                           'stdRAOffsetDeg': np.mean(astrometry_diagnostics['stdRAOffsetDeg']),
                                           'meanClippedRAOffsetDeg': np.mean(astrometry_diagnostics['meanClippedRAOffsetDeg']),
                                           'stdClippedRAOffsetDeg': np.mean(astrometry_diagnostics['stdClippedRAOffsetDeg']),
                                           'meanDecOffsetDeg': np.mean(astrometry_diagnostics['meanDecOffsetDeg']),
                                           'stdDecOffsetDeg': np.mean(astrometry_diagnostics['stdDecOffsetDeg']),
                                           'meanClippedDecOffsetDeg': np.mean(astrometry_diagnostics['meanClippedDecOffsetDeg']),
                                           'stdClippedDecOffsetDeg': np.mean(astrometry_diagnostics['stdClippedDecOffsetDeg'])}
        else:
            # Write dummy files
            mean_astrometry_diagnostics = None
            with open(output_root+'.astrometry_offsets.json', 'w') as fp:
                fp.writelines('Astrometry diagnostics could not be determined. Please see the logs for details')
            with open(output_root+'.astrometry_offsets.pdf', 'w') as fp:
                fp.writelines('Astrometry diagnostics could not be determined. Please see the logs for details')
    else:
        mean_astrometry_diagnostics = None
        print(f'Fewer than {min_number} sources found in the LOFAR image meet the '
              'astrometry cuts (major axis < 10" with positional errors < 2"). '
              'Skipping the astromety check...')

    return mean_astrometry_diagnostics


def main(flat_noise_image, flat_noise_rms_image, true_sky_image, true_sky_rms_image,
         input_catalog, input_skymodel, obs_ms, diagnostics_file, output_root,
         facet_region_file=None, photometry_comparison_skymodel=None,
         astrometry_comparison_skymodel=None, min_number=5):
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
    min_number : int, optional
        Minimum number of matched sources required for the comparisons
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

    # Do the photometry check and update the ouput dict
    result = check_photometry(obs_list[beam_ind], input_catalog, img_true_sky.freq,
                              min_number, comparison_skymodel=photometry_comparison_skymodel)
    if result is not None:
        cwl_output.update(result)

    # Do the astrometry check and update the ouput dict
    result = check_astrometry(obs_list[beam_ind], input_catalog, img_true_sky, facet_region_file,
                              min_number, output_root, comparison_skymodel=astrometry_comparison_skymodel)
    if result is not None:
        cwl_output.update(result)

    # Write out the full diagnostics
    with open(output_root+'.image_diagnostics.json', 'w') as fp:
        json.dump(cwl_output, fp)

    # Adjust plot filenames as needed to include output_root (those generated by LSMTool
    # lack it)
    diagnostic_plots = glob.glob(os.path.join('.', '*.pdf'))
    for src_filename in diagnostic_plots:
        if not os.path.basename(src_filename).startswith(output_root):
            dst_filename = os.path.join('.', f'{output_root}.' + os.path.basename(src_filename))
            if os.path.exists(dst_filename):
                os.remove(dst_filename)
            shutil.copy(src_filename, dst_filename)


if __name__ == '__main__':
    descriptiontext = "Calculate image photometry and astrometry diagnostics.\n"

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
    parser.add_argument('--facet_region_file', help='Filename of ds9 facet region file', type=str, default=None)
    parser.add_argument('--photometry_comparison_skymodel', help='Filename of photometry sky model', type=str, default=None)
    parser.add_argument('--astrometry_comparison_skymodel', help='Filename of astrometry sky model', type=str, default=None)
    parser.add_argument('--min_number', help='Minimum number of sources for diagnostics', type=int, default=5)

    args = parser.parse_args()
    main(args.flat_noise_image, args.flat_noise_rms_image, args.true_sky_image, args.true_sky_rms_image,
         args.input_catalog, args.input_skymodel, args.obs_ms, args.diagnostics_file, args.output_root,
         facet_region_file=args.facet_region_file, photometry_comparison_skymodel=args.photometry_comparison_skymodel,
         astrometry_comparison_skymodel=args.astrometry_comparison_skymodel, min_number=args.min_number)

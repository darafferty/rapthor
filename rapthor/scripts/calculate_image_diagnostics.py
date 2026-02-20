#!/usr/bin/env python3
"""
Script to calculate various image diagnostics
"""

import contextlib
import glob
import json
import logging
import os
import shutil
import tempfile
from argparse import ArgumentParser, RawTextHelpFormatter
from collections import defaultdict

import astropy.units as u
import casacore.tables as pt
import lsmtool
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.utils import iers
from astropy.visualization.wcsaxes import WCSAxes
from lsmtool.operations_lib import make_wcs

from rapthor.lib import miscellaneous as misc
from rapthor.lib.facet import SquareFacet, read_ds9_region_file
from rapthor.lib.fitsimage import FITSImage
from rapthor.lib.observation import Observation

if matplotlib.get_backend() != 'Agg':
    matplotlib.use('Agg')


# Turn off astropy's IERS downloads to fix problems in cases where compute
# node does not have internet access
iers.conf.auto_download = False

# Initialize logger
logger = logging.getLogger('rapthor:calculate_image_diagnostics')


@contextlib.contextmanager
def safe_load_skymodel(skymodel, message, post, **kws):
    """
    Context manager to catch loading errors and log an appropriate message

    Parameters
    ----------
    skymodel : str
        Skymodel to load.
    message : str
        Message to log if loading fails
    post : str
        Message to log after the error message and error details

    Yields
    -------
    lsmtool.SkyModel
        Loaded skymodel object
    """
    try:
        yield lsmtool.load(skymodel, **kws)
    except OSError as error:
        logger.info(
            '%s The error was: \n%s\n%s',
            message,
            error,
            post,
        )


def plot_astrometry_offsets(
    facets, field_ra, field_dec, output_file, plot_labels=False
):
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
    wcs = make_wcs(field_ra, field_dec, misc.WCS_PIXEL_SCALE)
    ra_offsets = []
    dec_offsets = []
    facet_patches = []
    facet_ra = []
    facet_dec = []
    facet_names = []
    for facet in facets:
        # Note: the offsets are calculated as (LOFAR model value) - (comparison
        # model value); e.g., a positive Dec offset indicates that the LOFAR
        # sources lie on average to the North of the comparison source
        # positions. Therefore, we multiply by -1 to obtain the correct arrow
        # directions in the quiver plot
        if facet.astrometry_diagnostics:
            ra_offsets.append(
                -1 * facet.astrometry_diagnostics['meanClippedRAOffsetDeg']
            )
            dec_offsets.append(
                -1 * facet.astrometry_diagnostics['meanClippedDecOffsetDeg']
            )
            facet_ra.append(facet.ra)
            facet_dec.append(facet.dec)
            facet_names.append(facet.name)
            facet_patches.append(facet.get_matplotlib_patch(wcs=wcs))

    # Set up the figure. We use various values that should produce a reasonably
    # sized figure with readable labels in most cases
    fig = plt.figure(
        1, figsize=(7.66, 7)
    )  # increase x size to ensure room for Dec label
    plt.clf()
    ax = WCSAxes(
        fig, [0.16, 0.1, 0.8, 0.8], wcs=wcs
    )  # set start x to give room for Dec label
    fig.add_axes(ax)
    RAAxis = ax.coords['ra']
    RAAxis.set_axislabel('RA', minpad=0.75)
    RAAxis.set_major_formatter('hh:mm:ss')
    DecAxis = ax.coords['dec']
    DecAxis.set_axislabel('Dec', minpad=0.75)
    DecAxis.set_major_formatter('dd:mm:ss')
    ax.coords.grid(color='black', alpha=0.5, linestyle='solid')
    ax.set_title(
        'Positional Offsets (arrows indicate direction and magnitude of '
        'correction)'
    )

    # Plot the facet polygons
    x, y = wcs.wcs_world2pix(facet_ra, facet_dec, misc.WCS_ORIGIN)
    for i, patch in enumerate(facet_patches):
        ax.add_patch(patch)
        if plot_labels:
            ax.annotate(facet_names[i], (x[i], y[i]), va='top', ha='center')

    # Plot the offsets. The arrows indicate the direction and magnitude of the
    # correction that should be applied to the LOFAR positions
    x_offsets = np.array(ra_offsets) / wcs.wcs.cdelt[0]
    y_offsets = np.array(dec_offsets) / wcs.wcs.cdelt[1]
    quiver_plot = plt.quiver(
        x,
        y,
        x_offsets,
        y_offsets,
        units='xy',
        angles='xy',
        scale=2 / 3600,
        color='blue',
    )
    plt.quiverkey(
        quiver_plot,
        0.1,
        0.95,
        1 / 3600 / wcs.wcs.cdelt[0],
        '1 arcsec',
        labelpos='N',
        coordinates='figure',
    )
    plt.savefig(output_file, format='pdf')


def fits_to_makesourcedb(
    catalog, reference_freq, flux_colname='Isl_Total_flux'
):
    """
    Converts a PyBDSF catalog to a makesourcedb sky model

    Note: the resulting makesourcedb catalog is a minimal catalog suitable for
    use in photometry and astrometry comparisons and should not be used for
    calibration

    Parameters
    ----------
    catalog : astropy Table object
        Input PyBDSF catalog
    reference_freq : float
        The reference frequency in Hz for the input catalog at which the flux
        densities were measured
    flux_colname : str, optional
        The name of the column in the input catalog that contains the flux
        density values

    Returns
    -------
    skymodel : LSMTool skymodel object
        The makesourcedb sky model
    """
    # Convert the result to makesourcedb format and write to a tempfile
    out_lines = [
        f'FORMAT = Name, Type, Ra, Dec, I, ReferenceFrequency={reference_freq}\n',
        *(
            f'{name}, POINT, {ra}, {dec}, {flux}, \n'
            for name, ra, dec, flux in zip(
                catalog['Source_id'],
                catalog['RA'],
                catalog['DEC'],
                catalog[flux_colname],
            )
        ),
    ]
    skymodel_file = tempfile.NamedTemporaryFile()
    with open(skymodel_file.name, 'w') as f:
        f.writelines(out_lines)

    return lsmtool.load(skymodel_file.name)


def check_photometry(
    obs,
    input_catalog,
    freq,
    min_number,
    comparison_skymodel=None,
    comparison_surveys=('TGSS', 'LOTSS'),
    backup_survey='NVSS',
):
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
        comparison (in makesourcedb format). If not given (or if it cannot be
        loaded), models are downloaded from the surveys defined by
        comparison_surveys
    comparison_surveys : list, optional
        A list giving the names of surveys to use for the photometry comparison
        when a sky model is not supplied with the comparison_skymodel argument
        (each name must be one of the VO services supported by LSMTool: see
        https://lsmtool.readthedocs.io/en/latest/lsmtool.html#lsmtool.load for
        the supported services)
    backup_survey : str, optional
        Survey name to use if the queries fail for all surveys given by
        comparison_surveys (as with comparison_surveys, the survey name must be
        one of the VO services supported by LSMTool). Ideally, a survey with
        full sky coverage should be used for this purpose. Set to None to
        disable

    Returns
    -------
    photometry_diagnostics : dict
        Photometry diagnostics. An empty dict is returned if the comparison
        could not be done successfully
    """
    # Load and filter the input PyBDSF FITS catalog as needed for the
    # photometry check. Sources are filtered to keep only those that:
    #   - lie within the FWHM of the primary beam (to exclude sources with
    #     uncertain primary beam corrections)
    #   - have deconvolved major axis < 10 arcsec (to exclude extended sources
    #     that may be poorly modeled)
    catalog = Table.read(input_catalog, format='fits')
    if len(catalog) == 0:
        logger.info(
            'No sources found in the LOFAR image. Skipping the photometry '
            'check...'
        )
        return {}

    phase_center = SkyCoord(ra=obs.ra * u.degree, dec=obs.dec * u.degree)
    coords_comp = SkyCoord(ra=catalog['RA'], dec=catalog['DEC'])
    separation = phase_center.separation(coords_comp)
    sec_el = 1.0 / np.sin(obs.mean_el_rad)
    fwhm_deg = 1.1 * ((3.0e8 / freq) / obs.diam) * 180 / np.pi * sec_el
    catalog = catalog[separation < fwhm_deg / 2 * u.degree]
    major_axis = catalog['DC_Maj']  # degrees
    catalog = catalog[major_axis < 10 / 3600]

    # Check number of sources against min_number set by user. If too few, return
    if len(catalog) < min_number:
        logger.info(
            'Fewer than %s sources found in the LOFAR image that '
            'meet the photometry cuts (major axis < 10" and located inside the '
            'FWHM of the primary beam"). Skipping the photometry check...',
            min_number,
        )
        return {}

    # Load photometry survey skymodels, fall back to backup if needed
    comparison_surveys = list(comparison_surveys)
    comparison_skymodels = load_photometry_surveys(
        obs, comparison_skymodel, comparison_surveys, backup_survey
    )

    # Convert the filtered catalog to a minimal sky model for use with LSMTool
    # and do the comparison for each survey.
    #
    # Note: LSMTool makes the plots with the same root filenames for each
    # (successful) comparison, so we need to rename them appropriately before
    # continuing to the next survey

    # Do the photometry check
    successful_surveys = []
    photometry_diagnostics = {}
    for survey, comparison_skymodel in zip(
        comparison_surveys, comparison_skymodels
    ):
        survey = survey.strip().upper()

        # Check if we're dealing with the backup survey and use it only if none
        # of the other surveys succeeded (the backup survey is always appended
        # to the end of comparison_surveys, so the other comparisons will have
        # been attempted if this point is reached)
        if survey == backup_survey:
            if successful_surveys:
                # Backup not needed, so skip further processing for it
                continue

            # Backup needed
            logger.info(
                'The backup survey catalog "%s" will be used for the photometry'
                ' check, as the queries for all other survey catalogs were '
                'unsuccessful',
                backup_survey,
            )

        if statistics := compare_photometry_survey(
            catalog, survey, comparison_skymodel, freq
        ):
            # Comparison succeeded
            successful_surveys.append(survey)

            # Append survey name to the diagnostic plots generated by LSMTool and
            # remove other, unneeded plots
            rename_plots(survey)

            # Save the diagnostics for the comparison
            photometry_diagnostics |= statistics

    return photometry_diagnostics


def load_photometry_surveys(
    observation, comparison_skymodel, comparison_surveys, backup_survey
):
    # Load photometry comparison model
    comparison_skymodels = []
    if comparison_skymodel:
        with safe_load_skymodel(
            comparison_skymodel,
            'Comparison sky model could not be loaded.',
            'Trying to download sky model(s) instead...',
        ) as skymodel:
            comparison_skymodels.append(skymodel)
            comparison_surveys = ['USER_SUPPLIED']
            logger.info(
                'Using the supplied comparison sky model for the photometry '
                'check'
            )

    if not comparison_skymodels:
        # Download sky model(s) given by comparison_surveys around the phase
        # center, using a 5-deg radius to ensure the field is fully covered
        if not comparison_surveys:
            logger.info(
                'A comparison sky model is not available and a list of '
                'comparison surveys was not supplied. Skipping photometry '
                'check...'
            )
            return {}

        if backup_survey is not None:
            if backup_survey in comparison_surveys:
                logger.info(
                    'The backup survey "%s" is already included in '
                    'comparison_surveys. Disabling the backup',
                    backup_survey,
                )
                backup_survey = None
            else:
                logger.info(
                    'Using "%s" as the backup survey catalog for the '
                    'photometry check',
                    backup_survey,
                )
                comparison_surveys.append(backup_survey)

        for survey in comparison_surveys:
            with safe_load_skymodel(
                survey,
                'A problem occurred when downloading the %s catalog for use'
                ' in the photometry check.',
                'Skipping this survey...',
                VOPosition=[observation.ra, observation.dec],
                VORadius=5.0,
            ) as skymodel:
                comparison_skymodels.append(skymodel)

    return comparison_skymodels


def compare_photometry_survey(catalog, survey, comparison_skymodel, freq):

    # Convert the output and compare, using the total flux from the
    # Gaussian fits ('Total_flux') to be consistent with the flux-scale
    # normalization
    comparison_skymodel.group('every')
    reference_skymodel = fits_to_makesourcedb(
        catalog, freq, flux_colname='Total_flux'
    )

    if result := reference_skymodel.compare(
        comparison_skymodel,
        radius='5 arcsec',
        excludeMultiple=True,
        make_plots=True,
        name1='LOFAR',
        name2=survey,
    ):
        # Append survey name to the diagnostic plots generated by LSMTool and
        # remove other, unneeded plots
        rename_plots(survey)

        # Save the diagnostics for the comparison
        return {
            f'{key}_{survey}': result[key]
            for key in (
                'meanRatio',
                'stdRatio',
                'meanClippedRatio',
                'stdClippedRatio',
            )
        }

    # Comparison failed due to insufficient matches. Continue with the
    # next comparison model (if any)
    logger.info(
        'The photometry check with the %s catalog could not be done due to '
        'insufficient matches. Skipping this survey...',
        survey,
    )


def rename_plots(survey):
    # Append survey name to the diagnostic plots generated by LSMTool and
    # remove other, unneeded plots
    for plot in [
        'flux_ratio_vs_distance',
        'flux_ratio_vs_flux',
        'flux_ratio_sky',
    ]:
        os.rename(f'{plot}.pdf', f'{plot}_{survey}.pdf')

    # other plots not related to photometry check
    for plot in ['positional_offsets_sky']:
        if os.path.exists(f'{plot}.pdf'):
            os.remove(f'{plot}.pdf')


def check_astrometry(
    obs,
    input_catalog,
    image,
    facet_region_file,
    min_number,
    output_root,
    comparison_skymodel=None,
):
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
        comparison (in makesourcedb format). If not given, a Pan-STARRS model
        is downloaded

    Returns
    -------
    mean_astrometry_diagnostics : dict
        Mean astrometry diagnostics. An empty dict is returned if the
        comparison could not be done successfully
    """
    # Load and filter the input PyBDSF FITS catalog as needed for the
    # astrometry check. Sources are filtered to keep only those that:
    #   - have deconvolved major axis < 10 arcsec (to exclude extended sources
    #     that may be poorly modeled)
    #   - have errors on RA and Dec of < 2 arcsec (to exclude sources
    #     with high positional uncertainties)
    catalog = Table.read(input_catalog, format='fits')
    if len(catalog) == 0:
        logger.info(
            'No sources found in the LOFAR image. Skipping the astrometry '
            'check...'
        )
        return {}

    major_axis = catalog['DC_Maj']  # degrees
    catalog = catalog[major_axis < 10 / 3600]
    ra_error = catalog['E_RA']  # degrees
    catalog = catalog[ra_error < 2 / 3600]
    dec_error = catalog['E_DEC']  # degrees
    catalog = catalog[dec_error < 2 / 3600]

    # Check number of sources against min_number set by user. If too few, return
    if len(catalog) < min_number:
        logger.info(
            'Fewer than %d sources found in the LOFAR image meet the astrometry'
            ' cuts (major axis < 10" with positional errors < 2"). Skipping the'
            ' astrometry check...',
            min_number,
        )
        return {}

    if facet_region_file is not None and os.path.isfile(facet_region_file):
        facets = read_ds9_region_file(facet_region_file)
    else:
        # Use a single rectangular facet centered on the phase center
        ra = obs.ra
        dec = obs.dec
        image_width = max(image.img_data.shape[-2:]) * abs(
            image.img_hdr['CDELT1']
        )
        # Do the astrometry check
        max_search_cone_radius = 0.5  # deg; Pan-STARRS search limit
        width = min(max_search_cone_radius * 2, image_width)
        facets = [SquareFacet('field', ra, dec, width)]

    # Convert the filtered catalog into a minimal sky model for use with LSMTool
    s_pybdsf = fits_to_makesourcedb(catalog, image.freq)

    # Loop over the facets, performing the astrometry checks for each
    astrometry_skymodel = None
    if comparison_skymodel:
        with safe_load_skymodel(
            comparison_skymodel,
            'Comparison sky model could not be loaded.',
            'Trying default sky model instead...'
        ) as astrometry_skymodel:
            astrometry_skymodel.group('every')
            logger.info(
                'Using the supplied comparison sky model for the astrometry '
                'check'
            )

    astrometry_keys = (
        'meanRAOffsetDeg',
        'stdRAOffsetDeg',
        'meanClippedRAOffsetDeg',
        'stdClippedRAOffsetDeg',
        'meanDecOffsetDeg',
        'stdDecOffsetDeg',
        'meanClippedDecOffsetDeg',
        'stdClippedDecOffsetDeg',
    )
    astrometry_diagnostics = defaultdict(list)
    for facet in facets:
        facet.set_skymodel(s_pybdsf.copy())
        facet.find_astrometry_offsets(astrometry_skymodel, min_number=min_number)
        if facet.astrometry_diagnostics:
            astrometry_diagnostics['facet_name'].append(facet.name)
            for key in astrometry_keys:
                astrometry_diagnostics[key].append(
                    facet.astrometry_diagnostics[key]
                )

    if not astrometry_diagnostics['facet_name']:
        return {}

    with open(f'{output_root}.astrometry_offsets.json', 'w') as fp:
        json.dump(dict(astrometry_diagnostics), fp)

    plot_astrometry_offsets(
        facets, obs.ra, obs.dec, f'{output_root}.astrometry_offsets.pdf'
    )

    # Calculate mean offsets
    return {
        key: np.mean(astrometry_diagnostics[key]) for key in astrometry_keys
    }


def main(
    flat_noise_image,
    flat_noise_rms_image,
    true_sky_image,
    true_sky_rms_image,
    input_catalog,
    obs_ms,
    obs_starttime,
    obs_ntimes,
    diagnostics_file,
    output_root,
    facet_region_file=None,
    photometry_comparison_skymodel=None,
    photometry_comparison_surveys=['TGSS', 'LOTSS'],
    photometry_backup_survey='NVSS',
    astrometry_comparison_skymodel=None,
    min_number=5,
):
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
    obs_ms : list of str
        List of MS files to use to derive the theoretical image noise and
        other properties of the observation
    obs_starttime : list of str
        List of start times in casacore MVTime format for each input MS
    obs_ntimes : list of int
        List of nuber of time slots for each input MS
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
        comparison (in makesourcedb format). If not given, models are
        downloaded from the surveys defined by photometry_comparison_surveys
    photometry_comparison_surveys : list, optional
        A list giving the names of surveys to use for the photometry comparison
        (each must be one of the VO services supported by LSMTool: see
        https://lsmtool.readthedocs.io/en/latest/lsmtool.html#lsmtool.load for
        the supported services). If not given, the list is set to
        ['TGSS', 'LOTSS']
    photometry_backup_survey : str, optional
        Survey name to use if the queries fail for all surveys given by
        comparison_surveys (as with comparison_surveys, the survey name must be
        one of the VO services supported by LSMTool). Ideally, a survey with
        full sky coverage should be used for this purpose
    astrometry_comparison_skymodel : str, optional
        Filename of the sky model to use for the astrometry comparison (in
        makesourcedb format). If not given, a Pan-STARRS model is downloaded
    min_number : int, optional
        Minimum number of matched sources required for the comparisons
    """
    # Select the best MS
    if isinstance(obs_ms, str):
        obs_ms = misc.string2list(obs_ms)

    if isinstance(obs_starttime, str):
        obs_starttime = misc.string2list(obs_starttime)

    if isinstance(obs_ntimes, str):
        obs_ntimes = misc.string2list(obs_ntimes)
        obs_ntimes = [int(ntimes) for ntimes in obs_ntimes]

    if len(obs_ms) > 1:
        ms_times = []
        for ms in obs_ms:
            tab = pt.table(ms, ack=False)
            ms_times.append(np.mean(tab.getcol('TIME')))
            tab.close()
        ms_times_sorted = sorted(ms_times)
        mid_time = ms_times_sorted[len(ms_times) // 2]
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
    obs_list = []
    for ms, starttime, ntimes in zip(
        obs_ms, obs_starttime, obs_ntimes, strict=True
    ):
        starttime_mjd = misc.convert_mvt2mjd(starttime)  # MJD sec
        endtime_mjd = (
            starttime_mjd + ntimes * Observation(ms).timepersample
        )  # MJD sec
        obs_list.append(Observation(ms, starttime_mjd, endtime_mjd))

    theoretical_rms, unflagged_fraction = misc.calc_theoretical_noise(
        obs_list, use_lotss_estimate=True
    )  # Jy/beam
    dynamic_range_global_true_sky = float(
        img_true_sky.max_value / rms_img_true_sky.min_value
    )
    dynamic_range_local_true_sky = float(
        np.nanmax(rms_img_flat_noise.img_data / rms_img_true_sky.img_data)
    )
    dynamic_range_global_flat_noise = float(
        img_flat_noise.max_value / rms_img_flat_noise.min_value
    )
    dynamic_range_local_flat_noise = float(
        np.nanmax(img_flat_noise.img_data / rms_img_flat_noise.img_data)
    )
    cwl_output.update(
        {
            'theoretical_rms': theoretical_rms,
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
            'beam_fwhm': img_true_sky.beam,
        }
    )

    # Do the photometry check and update the ouput dict
    result = check_photometry(
        obs_list[beam_ind],
        input_catalog,
        img_true_sky.freq,
        min_number,
        comparison_skymodel=photometry_comparison_skymodel,
        comparison_surveys=photometry_comparison_surveys,
        backup_survey=photometry_backup_survey,
    )
    cwl_output.update(result)

    # Do the astrometry check and update the ouput dict
    result = check_astrometry(
        obs_list[beam_ind],
        input_catalog,
        img_true_sky,
        facet_region_file,
        min_number,
        output_root,
        comparison_skymodel=astrometry_comparison_skymodel,
    )
    cwl_output.update(result)

    # Write out the full diagnostics
    with open(f'{output_root}.image_diagnostics.json', 'w') as fp:
        json.dump(cwl_output, fp)

    # Adjust plot filenames as needed to include output_root (those generated by LSMTool
    # lack it)
    diagnostic_plots = glob.glob(os.path.join('.', '*.pdf'))
    for src_filename in diagnostic_plots:
        if not os.path.basename(src_filename).startswith(output_root):
            dst_filename = os.path.join(
                '.', f'{output_root}.{os.path.basename(src_filename)}'
            )
            if os.path.exists(dst_filename):
                os.remove(dst_filename)
            shutil.copy(src_filename, dst_filename)


if __name__ == '__main__':
    descriptiontext = 'Calculate image photometry and astrometry diagnostics.\n'

    parser = ArgumentParser(
        description=descriptiontext, formatter_class=RawTextHelpFormatter
    )
    parser.add_argument(
        'flat_noise_image', help='Filename of flat-noise FITS image'
    )
    parser.add_argument(
        'flat_noise_rms_image', help='Filename of flat-noise RMS FITS image'
    )
    parser.add_argument(
        'true_sky_image', help='Filename of true sky FITS image'
    )
    parser.add_argument(
        'true_sky_rms_image', help='Filename of true sky RMS FITS image'
    )
    parser.add_argument(
        'input_catalog', help='Filename of input PyBDSF FITS catalog'
    )
    parser.add_argument('obs_ms', help='Filename of observation MS')
    parser.add_argument('obs_starttime', help='Start time of observation')
    parser.add_argument(
        'obs_ntimes', help='Number of time slots of observation'
    )
    parser.add_argument(
        'diagnostics_file', help='Filename of diagnostics JSON file'
    )
    parser.add_argument('output_root', help='Root of output files')
    parser.add_argument(
        '--facet_region_file',
        help='Filename of ds9 facet region file',
        type=str,
        default=None,
    )
    parser.add_argument(
        '--photometry_comparison_skymodel',
        help='Filename of photometry sky model',
        type=str,
        default=None,
    )
    parser.add_argument(
        '--photometry_comparison_surveys',
        help='List of photometry surveys to use when '
        'photometry_comparison_skymodel is not given',
        type=list,
        default=['TGSS', 'LOTSS'],
    )
    parser.add_argument(
        '--photometry_backup_survey',
        help='Name of photometry survey to use as backup '
        'if all queries to photometry_comparison_surveys fail',
        type=str,
        default='NVSS',
    )
    parser.add_argument(
        '--astrometry_comparison_skymodel',
        help='Filename of astrometry sky model',
        type=str,
        default=None,
    )
    parser.add_argument(
        '--min_number',
        help='Minimum number of sources for diagnostics',
        type=int,
        default=5,
    )

    args = parser.parse_args()
    main(
        args.flat_noise_image,
        args.flat_noise_rms_image,
        args.true_sky_image,
        args.true_sky_rms_image,
        args.input_catalog,
        args.obs_ms,
        args.obs_starttime,
        args.obs_ntimes,
        args.diagnostics_file,
        args.output_root,
        facet_region_file=args.facet_region_file,
        photometry_comparison_skymodel=args.photometry_comparison_skymodel,
        photometry_comparison_surveys=args.photometry_comparison_surveys,
        photometry_backup_survey=args.photometry_backup_survey,
        astrometry_comparison_skymodel=args.astrometry_comparison_skymodel,
        min_number=args.min_number,
    )

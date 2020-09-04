"""
Module that holds all parset-related functions
"""
import sys
import os
import glob
import logging
import configparser
from rapthor._logging import set_log_file
from rapthor.cluster import find_executables, get_compute_nodes
from astropy.coordinates import Angle
import platform

log = logging.getLogger('rapthor:parset')


def parset_read(parset_file, use_log_file=True, skip_cluster=False):
    """
    Read a rapthor-formatted parset file and return dict of parameters

    Parameters
    ----------
    parset_file : str
        Filename of rapthor-formated parset file
    use_log_file : bool, optional
        Use a log file as well as outputing to the screen

    Returns
    -------
    parset_dict : dict
        Dict of parset parameters
    """
    if not os.path.isfile(parset_file):
        log.critical("Missing parset file ({}), I don't know what to do :'(".format(parset_file))
        sys.exit(1)

    log.info("Reading parset file: {}".format(parset_file))
    parset = configparser.RawConfigParser()
    parset.read(parset_file)

    # Handle global parameters
    parset_dict = get_global_options(parset)

    # Handle calibration parameters
    parset_dict['calibration_specific'].update(get_calibration_options(parset))

    # Handle imaging parameters
    parset_dict['imaging_specific'].update(get_imaging_options(parset))

    # Handle cluster-specific parameters
    if not skip_cluster:
        parset_dict['cluster_specific'].update(get_cluster_options(parset))

    # Set up working directory. All output will be placed in this directory
    if not os.path.isdir(parset_dict['dir_working']):
        os.mkdir(parset_dict['dir_working'])
    try:
        os.chdir(parset_dict['dir_working'])
        for subdir in ['logs', 'pipelines', 'regions', 'skymodels', 'images',
                       'solutions', 'scratch']:
            subdir_path = os.path.join(parset_dict['dir_working'], subdir)
            if not os.path.isdir(subdir_path):
                os.mkdir(subdir_path)
    except Exception as e:
        log.critical("Cannot use the working dir {0}: {1}".format(parset_dict['dir_working'], e))
        sys.exit(1)
    if use_log_file:
        set_log_file(os.path.join(parset_dict['dir_working'], 'logs', 'rapthor.log'))
    log.info("=========================================================\n")
    log.info("Working directory is {}".format(parset_dict['dir_working']))

    # Get the input MS files
    ms_search_list = parset_dict['input_ms'].strip('[]').split(',')
    ms_search_list = [ms.strip() for ms in ms_search_list]
    ms_files = []
    for search_str in ms_search_list:
        ms_files += glob.glob(os.path.join(search_str))
    parset_dict['mss'] = sorted(ms_files)
    if len(parset_dict['mss']) == 0:
        log.error('No input MS files were not found!')
        sys.exit(1)
    log.info("Working on {} input MS file(s)".format(len(parset_dict['mss'])))

    # Make sure the initial skymodel is present
    if 'input_skymodel' not in parset_dict:
        log.error('No input sky model file given. Exiting...')
        sys.exit(1)
    elif not os.path.exists(parset_dict['input_skymodel']):
        log.error('Input sky model file "{}" not found. Exiting...'.format(parset_dict['input_skymodel']))
        sys.exit(1)

    # Check for invalid sections
    given_sections = list(parset._sections.keys())
    allowed_sections = ['global', 'calibration', 'imaging', 'cluster']
    for section in given_sections:
        if section not in allowed_sections:
            log.warning('Section "{}" was given in the parset but is not a valid '
                        'section name'.format(section))

    return parset_dict


def get_global_options(parset):
    """
    Handle the global options

    Parameters
    ----------
    parset : RawConfigParser object
        Input parset

    Returns
    -------
    parset_dict : dict
        Dictionary with all global options

    """
    parset_dict = parset._sections['global'].copy()
    parset_dict.update({'calibration_specific': {}, 'imaging_specific': {}, 'cluster_specific': {}})

    # Fraction of data to use (default = 1.0). If less than one, the input data are divided
    # by time into chunks (of no less than slow_timestep_sec below) that sum to the requested
    # fraction, spaced out evenly over the full time range
    if 'data_fraction' in parset_dict:
        parset_dict['data_fraction'] = parset.getfloat('global', 'data_fraction')
    else:
        parset_dict['data_fraction'] = 1.0

    # Regroup input sky model (default = True)
    if 'regroup_input_skymodel' in parset_dict:
        parset_dict['regroup_input_skymodel'] = parset.getboolean('global', 'regroup_input_skymodel')
    else:
        parset_dict['regroup_input_skymodel'] = True

    # Apparent-flux input sky model (default = None)
    if 'apparent_skymodel' not in parset_dict:
        parset_dict['apparent_skymodel'] = None

    # Filename of h5parm file containing solutions for the patches in the
    # input sky model
    if 'input_h5parm' not in parset_dict:
        parset_dict['input_h5parm'] = None
    if 'solset' not in parset_dict:
        parset_dict['solset'] = None
    if 'tec_soltab' not in parset_dict:
        parset_dict['tec_soltab'] = None
    if 'scalarphase_soltab' not in parset_dict:
        parset_dict['scalarphase_soltab'] = None
    if 'slow_phase_soltab' not in parset_dict:
        parset_dict['slow_phase_soltab'] = None
    if 'slow_amplitude_soltab' not in parset_dict:
        parset_dict['slow_amplitude_soltab'] = None

    # Define strategy
    if 'strategy' not in parset_dict:
        parset_dict['strategy'] = 'selfcal'

    # Flagging ranges (default = no flagging). A range of times, baselines, and
    # frequencies to flag can be specified (see the DPPP documentation for
    # details of syntax) By default, the ranges are AND-ed to produce the final flags,
    # but a set expression can be specified that controls how the selections are
    # combined
    flag_list = []
    if 'flag_abstime' not in parset_dict:
        parset_dict['flag_abstime'] = None
    else:
        flag_list.append('flag_abstime')
    if 'flag_baseline' not in parset_dict:
        parset_dict['flag_baseline'] = None
    else:
        flag_list.append('flag_baseline')
    if 'flag_freqrange' not in parset_dict:
        parset_dict['flag_freqrange'] = None
    else:
        flag_list.append('flag_freqrange')
    if 'flag_expr' not in parset_dict:
        parset_dict['flag_expr'] = ' and '.join(flag_list)
    else:
        for f in flag_list:
            if f not in parset_dict['flag_expr']:
                log.error('Flag selection "{}" was specified but does not '
                          'appear in flag_expr'.format(f))
                sys.exit(1)

    # Check for invalid options
    given_options = parset.options('global')
    allowed_options = ['dir_working', 'input_ms', 'strategy',
                       'use_compression', 'flag_abstime', 'flag_baseline', 'flag_freqrange',
                       'flag_expr', 'input_skymodel', 'apparent_skymodel',
                       'regroup_input_skymodel', 'input_h5parm', 'data_fraction']
    for option in given_options:
        if option not in allowed_options:
            log.warning('Option "{}" was given in the [global] section of the '
                        'parset but is not a valid global option'.format(option))

    return parset_dict


def get_calibration_options(parset):
    """
    Handle the calibration options

    Parameters
    ----------
    parset : RawConfigParser object
        Input parset

    Returns
    -------
    parset_dict : dict
        Dictionary with all calibration options

    """
    if 'calibration' in list(parset._sections.keys()):
        parset_dict = parset._sections['calibration']
        given_options = parset.options('calibration')
    else:
        parset_dict = {}
        given_options = []

    # If one of the included sky models (see rapthor/skymodels) is within 2 * PB_FWHM of the
    # field center, include it in the calibration (default = False)
    if 'use_included_skymodels' in parset_dict:
        parset_dict['use_included_skymodels'] = parset.getboolean('calibration', 'use_included_skymodels')
    else:
        parset_dict['use_included_skymodels'] = False

    # Target flux density in Jy for grouping
    if 'patch_target_flux_jy' in parset_dict:
        parset_dict['patch_target_flux_jy'] = parset.getfloat('calibration', 'patch_target_flux_jy')
    else:
        parset_dict['patch_target_flux_jy'] = 2.5

    # Target number of patches for grouping
    if 'patch_target_number' in parset_dict:
        parset_dict['patch_target_number'] = parset.getint('calibration', 'patch_target_number')
        if parset_dict['patch_target_number'] < 1:
            parset_dict['patch_target_number'] = 1
    else:
        parset_dict['patch_target_number'] = None

    # Maximum number of cycles of the last step of selfcal to perform (default =
    # 10). The last step is looped until the number of cycles reaches this value or
    # until the improvement in dynamic range over the previous image is less than
    # 1.25%. A separate setting can also be used for the target facet only (allowing
    # one to reduce the number for non-target facets)
    if 'max_selfcal_loops' in parset_dict:
        parset_dict['max_selfcal_loops'] = parset.getint('calibration', 'max_selfcal_loops')
    else:
        parset_dict['max_selfcal_loops'] = 10

    # Minimum uv distance in lambda for calibration (default = 80)
    if 'solve_min_uv_lambda' in parset_dict:
        parset_dict['solve_min_uv_lambda'] = parset.getfloat('calibration', 'solve_min_uv_lambda')
    else:
        parset_dict['solve_min_uv_lambda'] = 80.0

    # Solution intervals
    if 'fast_timestep_sec' in parset_dict:
        parset_dict['fast_timestep_sec'] = parset.getfloat('calibration', 'fast_timestep_sec')
    else:
        parset_dict['fast_timestep_sec'] = 8.0
    if 'fast_freqstep_hz' in parset_dict:
        parset_dict['fast_freqstep_hz'] = parset.getfloat('calibration', 'fast_freqstep_hz')
    else:
        parset_dict['fast_freqstep_hz'] = 1e6
    if 'slow_timestep_sec' in parset_dict:
        parset_dict['slow_timestep_sec'] = parset.getfloat('calibration', 'slow_timestep_sec')
    else:
        parset_dict['slow_timestep_sec'] = 600.0
    if 'slow_freqstep_hz' in parset_dict:
        parset_dict['slow_freqstep_hz'] = parset.getfloat('calibration', 'slow_freqstep_hz')
    else:
        parset_dict['slow_freqstep_hz'] = 1e6

    # Smoothness constraint
    if 'fast_smoothnessconstraint' in parset_dict:
        parset_dict['fast_smoothnessconstraint'] = parset.getfloat('calibration', 'fast_smoothnessconstraint')
    else:
        parset_dict['fast_smoothnessconstraint'] = 6e6
    if 'slow_smoothnessconstraint' in parset_dict:
        parset_dict['slow_smoothnessconstraint'] = parset.getfloat('calibration', 'slow_smoothnessconstraint')
    else:
        parset_dict['slow_smoothnessconstraint'] = 3e6

    # Solver parameters
    if 'propagatesolutions' in parset_dict:
        parset_dict['propagatesolutions'] = parset.getboolean('calibration', 'propagatesolutions')
    else:
        parset_dict['propagatesolutions'] = True
    if 'maxiter' in parset_dict:
        parset_dict['maxiter'] = parset.getint('calibration', 'maxiter')
    else:
        parset_dict['maxiter'] = 50
    if 'stepsize' in parset_dict:
        parset_dict['stepsize'] = parset.getfloat('calibration', 'stepsize')
    else:
        parset_dict['stepsize'] = 0.02
    if 'tolerance' in parset_dict:
        parset_dict['tolerance'] = parset.getfloat('calibration', 'tolerance')
    else:
        parset_dict['tolerance'] = 1e-3

    # Use the IDG for predict during calibration (default = False)?
    if 'use_idg_predict' in parset_dict:
        parset_dict['use_idg_predict'] = parset.getboolean('calibration', 'use_idg_predict')
    else:
        parset_dict['use_idg_predict'] = False

    # Do a extra "debug" step during calibration (default = False)?
    if 'debug' in parset_dict:
        parset_dict['debug'] = parset.getboolean('calibration', 'debug')
    else:
        parset_dict['debug'] = False

    # Check for invalid options
    allowed_options = ['max_selfcal_loops', 'solve_min_uv_lambda', 'fast_timestep_sec',
                       'fast_freqstep_hz', 'slow_timestep_sec',
                       'slow_freqstep_hz', 'propagatesolutions', 'maxiter',
                       'stepsize', 'tolerance', 'patch_target_number',
                       'patch_target_flux_jy', 'fast_smoothnessconstraint',
                       'slow_smoothnessconstraint', 'use_idg_predict', 'debug']
    for option in given_options:
        if option not in allowed_options:
            log.warning('Option "{}" was given in the [calibration] section of the '
                        'parset but is not a valid calibration option'.format(option))

    return parset_dict


def get_imaging_options(parset):
    """
    Handle the imaging options

    Parameters
    ----------
    parset : RawConfigParser object
        Input parset

    Returns
    -------
    parset_dict : dict
        Dictionary with all imaging options

    """
    if 'imaging' in list(parset._sections.keys()):
        parset_dict = parset._sections['imaging']
        given_options = parset.options('imaging')
    else:
        parset_dict = {}
        given_options = []

    # Size of area to image when using a grid (default = mean FWHM of the primary beam)
    if 'grid_width_ra_deg' in parset_dict:
        parset_dict['grid_width_ra_deg'] = parset.getfloat('imaging', 'grid_width_ra_deg')
    else:
        parset_dict['grid_width_ra_deg'] = None
    if 'grid_width_dec_deg' in parset_dict:
        parset_dict['grid_width_dec_deg'] = parset.getfloat('imaging', 'grid_width_dec_deg')
    else:
        parset_dict['grid_width_dec_deg'] = None

    # Number of sectors along RA to use in imaging grid (default = 0). The number of sectors in
    # Dec will be determined automatically to ensure the whole area specified with grid_center_ra,
    # grid_center_dec, grid_width_ra_deg, and grid_width_dec_deg is imaged. Set grid_nsectors_ra = 0 to force a
    # single sector for the full area. Multiple sectors are useful for parallelizing the imaging
    # over multiple nodes of a cluster or for computers with limited memory
    if 'grid_nsectors_ra' in parset_dict:
        parset_dict['grid_nsectors_ra'] = parset.getint('imaging', 'grid_nsectors_ra')
    else:
        parset_dict['grid_nsectors_ra'] = 1

    # Center of grid to image (default = phase center of data)
    # grid_center_ra = 14h41m01.884
    # grid_center_dec = +35d30m31.52
    if 'grid_center_ra' in parset_dict:
        parset_dict['grid_center_ra'] = Angle(parset_dict['grid_center_ra']).to('deg').value
    else:
        parset_dict['grid_center_ra'] = None
    if 'grid_center_dec' in parset_dict:
        parset_dict['grid_center_dec'] = Angle(parset_dict['grid_center_dec']).to('deg').value
    else:
        parset_dict['grid_center_dec'] = None

    # Skip corner sectors of grid
    if 'skip_corner_sectors' in parset_dict:
        parset_dict['skip_corner_sectors'] = parset.getboolean('imaging', 'skip_corner_sectors')
    else:
        parset_dict['skip_corner_sectors'] = False

    # Instead of a grid, imaging sectors can be defined individually by specifying
    # their centers and widths. If sectors are specified in this way, they will be
    # used instead of the sector grid. Note that the sectors should not overlap
    # sector_center_ra_list = [14h41m01.884, 14h13m23.234]
    # sector_center_dec_list = [+35d30m31.52, +37d21m56.86]
    # sector_width_ra_deg_list = [0.532, 0.127]
    # sector_width_dec_deg_list = [0.532, 0.127]
    len_list = []
    if 'sector_center_ra_list' in parset_dict:
        val_list = parset_dict['sector_center_ra_list'].strip('[]').split(',')
        if val_list[0] == '':
            val_list = []
        val_list = [Angle(v).to('deg').value for v in val_list]
        parset_dict['sector_center_ra_list'] = val_list
        len_list.append(len(val_list))
    else:
        parset_dict['sector_center_ra_list'] = []
    if 'sector_center_dec_list' in parset_dict:
        val_list = parset_dict['sector_center_dec_list'].strip('[]').split(',')
        if val_list[0] == '':
            val_list = []
        val_list = [Angle(v).to('deg').value for v in val_list]
        parset_dict['sector_center_dec_list'] = val_list
        len_list.append(len(val_list))
    else:
        parset_dict['sector_center_dec_list'] = []
    if 'sector_width_ra_deg_list' in parset_dict:
        val_list = parset_dict['sector_width_ra_deg_list'].strip('[]').split(',')
        if val_list[0] == '':
            val_list = []
        val_list = [float(v) for v in val_list]
        parset_dict['sector_width_ra_deg_list'] = val_list
        len_list.append(len(val_list))
    else:
        parset_dict['sector_width_ra_deg_list'] = []
    if 'sector_width_dec_deg_list' in parset_dict:
        val_list = parset_dict['sector_width_dec_deg_list'].strip('[]').split(',')
        if val_list[0] == '':
            val_list = []
        val_list = [float(v) for v in val_list]
        parset_dict['sector_width_dec_deg_list'] = val_list
        len_list.append(len(val_list))
    else:
        parset_dict['sector_width_dec_deg_list'] = []
    if 'sector_do_multiscale_list' in parset_dict:
        val_list = parset_dict['sector_do_multiscale_list'].strip('[]').split(',')
        if val_list[0] == '':
            val_list = []
        bool_list = []
        for v in val_list:
            if v.lower().strip() == 'true':
                b = True
            elif v.lower().strip() == 'false':
                b = False
            elif v.lower().strip() == 'none':
                b = None
            else:
                log.error('The entry "{}" in sector_do_multiscale_list is invalid. It must '
                          'be one of True, False, or None'.format(v))
                sys.exit(1)
            bool_list.append(b)
        parset_dict['sector_do_multiscale_list'] = bool_list
        len_list.append(len(bool_list))
    else:
        parset_dict['sector_do_multiscale_list'] = []

    # Check that all the above options have the same number of entries
    if len(set(len_list)) > 1:
        log.error('The options sector_center_ra_list, sector_center_dec_list, '
                  'sector_width_ra_deg_list, sector_width_dec_deg_list, and '
                  'sector_do_multiscale_list (if specified) must all have the same number of '
                  'entires')
        sys.exit(1)

    # IDG (image domain gridder) mode to use in WSClean (default = hybrid). The mode can
    # be cpu, gpu, or hybrid.
    if 'idg_mode' not in parset_dict:
        parset_dict['idg_mode'] = 'hybrid'

    # Use screens during imaging (default = True).
    if 'use_screens' in parset_dict:
        parset_dict['use_screens'] = parset.getboolean('imaging', 'use_screens')
    else:
        parset_dict['use_screens'] = True

    # Use MPI during imaging (default = False).
    if 'use_mpi' in parset_dict:
        parset_dict['use_mpi'] = parset.getboolean('imaging', 'use_mpi')
    else:
        parset_dict['use_mpi'] = False

    # Reweight the visibility data before imaging (default = True)
    if 'reweight' in parset_dict:
        parset_dict['reweight'] = parset.getboolean('imaging', 'reweight')
    else:
        parset_dict['reweight'] = True

    # Max desired peak flux density reduction at center of the facet edges due to
    # bandwidth smearing (at the mean frequency) and time smearing (default = 0.15 =
    # 15% reduction in peak flux). Higher values result in shorter run times but
    # more smearing away from the facet centers. This value only applies to the
    # facet imaging (selfcal always uses a value of 0.15)
    if 'max_peak_smearing' in parset_dict:
        parset_dict['max_peak_smearing'] = parset.getfloat('imaging', 'max_peak_smearing')
    else:
        parset_dict['max_peak_smearing'] = 0.15

    # List of scales in pixels to use when multiscale clean is activated (default =
    # auto). Note that multiscale clean is activated for a direction only when the
    # calibrator or a source in the facet is determined to be larger than 4 arcmin,
    # the facet contains the target (specified below with target_ra and target_dec),
    # or mscale_selfcal_do / mscale_facet_do is set for the direction in the
    # directions file
    if 'multiscale_scales_pixel' in parset_dict:
        val_list = parset_dict['multiscale_scales_pixel'].strip('[]').split(',')
        str_list = ','.join([v.strip() for v in val_list])
        parset_dict['multiscale_scales_pixel'] = str_list
    else:
        parset_dict['multiscale_scales_pixel'] = None

    # Selfcal imaging parameters: pixel size in arcsec (default = 1.25), Briggs
    # robust parameter (default = -0.5) and minimum uv distance in lambda
    # (default = 80). These settings apply both to selfcal images and to the
    # full facet image used to make the improved facet model that is subtracted
    # from the data
    if 'cellsize_arcsec' in parset_dict:
        parset_dict['cellsize_arcsec'] = parset.getfloat('imaging', 'cellsize_arcsec')
    else:
        parset_dict['cellsize_arcsec'] = 1.25
    if 'robust' in parset_dict:
        parset_dict['robust'] = parset.getfloat('imaging', 'robust')
    else:
        parset_dict['robust'] = -0.5
    if 'min_uv_lambda' in parset_dict:
        parset_dict['min_uv_lambda'] = parset.getfloat('imaging', 'min_uv_lambda')
    else:
        parset_dict['min_uv_lambda'] = 80.0
    if 'max_uv_lambda' in parset_dict:
        parset_dict['max_uv_lambda'] = parset.getfloat('imaging', 'max_uv_lambda')
    else:
        parset_dict['max_uv_lambda'] = 1e6
    if 'taper_arcsec' in parset_dict:
        parset_dict['taper_arcsec'] = parset.getfloat('imaging', 'taper_arcsec')
    else:
        parset_dict['taper_arcsec'] = 0.0

    # A target can be specified to ensure that it falls entirely within a single
    # facet. The values should be those of a circular region that encloses the
    # source and not those of the target itself. Lastly, the target can be placed in
    # a facet of its own. In this case, it will not go through selfcal but will
    # instead use the selfcal solutions of the nearest facet for which selfcal was
    # done
    if 'target_ra' not in parset_dict:
        parset_dict['target_ra'] = None
    if 'target_dec' not in parset_dict:
        parset_dict['target_dec'] = None
    if 'target_radius_arcmin' in parset_dict:
        parset_dict['target_radius_arcmin'] = parset.getfloat('directions',
            'target_radius_arcmin')
    else:
        parset_dict['target_radius_arcmin'] = None

    # Padding factor for WSClean images (default = 1.2)
    if 'wsclean_image_padding' in parset_dict:
        parset_dict['wsclean_image_padding'] = parset.getfloat('imaging', 'wsclean_image_padding')
    else:
        parset_dict['wsclean_image_padding'] = 1.2
    if parset_dict['wsclean_image_padding'] < 1.0:
        parset_dict['wsclean_image_padding'] = 1.0

    # Check for invalid options
    allowed_options = ['max_peak_smearing', 'cellsize_arcsec', 'robust', 'reweight',
                       'multiscale_scales_pixel', 'grid_center_ra', 'grid_center_dec',
                       'grid_width_ra_deg', 'grid_width_dec_deg', 'grid_nsectors_ra',
                       'wsclean_image_padding', 'min_uv_lambda', 'max_uv_lambda',
                       'robust', 'padding', 'sector_center_ra_list', 'sector_center_dec_list',
                       'sector_width_ra_deg_list', 'sector_width_dec_deg_list',
                       'idg_mode', 'sector_do_multiscale_list', 'target_ra', 'use_mpi',
                       'target_dec', 'target_radius_arcmin', 'use_screens',
                       'skip_corner_sectors']
    for option in given_options:
        if option not in allowed_options:
            log.warning('Option "{}" was given in the [imaging] section of the '
                        'parset but is not a valid imaging option'.format(option))

    return parset_dict


def get_cluster_options(parset):
    """
    Handle the compute cluster options

    Parameters
    ----------
    parset : RawConfigParser object
        Input parset

    Returns
    -------
    parset_dict : dict
        Dictionary with all cluster options

    """
    if 'cluster' in list(parset._sections.keys()):
        parset_dict = parset._sections['cluster']
        given_options = parset.options('cluster')
    else:
        parset_dict = {}
        given_options = []

    # Paths to the LOFAR software
    if 'lofarroot' not in parset_dict:
        if 'LOFARROOT' in os.environ:
            parset_dict['lofarroot'] = os.environ['LOFARROOT']
        else:
            log.critical("The LOFAR root directory cannot be determined. Please "
                         "specify it in the [cluster] section of the parset as lofarroot")
            sys.exit(1)
    if 'lofarpythonpath' not in parset_dict:
        if parset_dict['lofarroot'] in os.environ['PYTHONPATH']:
            pypaths = os.environ['PYTHONPATH'].split(':')
            for pypath in pypaths:
                if parset_dict['lofarroot'] in pypath:
                    parset_dict['lofarpythonpath'] = pypath
                    break
        else:
            log.critical("The LOFAR Python root directory cannot be determined. "
                         "Please specify it in the [cluster] section of the parset as "
                         "lofarpythonpath")
            sys.exit(1)

    # Paths to required executables
    parset_dict = find_executables(parset_dict)

    # Number of CPUs per node to be used.
    if 'ncpu' in parset_dict:
        parset_dict['ncpu'] = parset.getint('cluster', 'ncpu')
    else:
        import multiprocessing
        parset_dict['ncpu'] = multiprocessing.cpu_count()

    # Cluster type (default = singleMachine). Use cluster_type = pbs to use PBS / torque
    # reserved nodes and cluster_type = slurm to use SLURM reserved ones
    if 'batch_system' not in parset_dict:
        parset_dict['batch_system'] = 'singleMachine'
    if 'max_nodes' in parset_dict:
        parset_dict['max_nodes'] = parset.getint('cluster', 'max_nodes')
    else:
        parset_dict['max_nodes'] = 12

    # Full path to a local disk on the nodes for I/O-intensive processing. The path
    # must be the same for all nodes
    if 'dir_local' not in parset_dict:
        parset_dict['dir_local'] = None
    else:
        parset_dict['dir_local'] = parset_dict['dir_local'].rstrip('/')

    # Check for invalid options
    allowed_options = ['ncpu', 'fmem', 'cluster_type', 'dir_local', 'lofarroot',
                       'lofarpythonpath', 'batch_system', 'max_nodes']
    for option in given_options:
        if option not in allowed_options:
            log.warning('Option "{}" was given in the [cluster] section of the '
                        'parset but is not a valid cluster option'.format(option))

    return parset_dict

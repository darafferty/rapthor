"""
Module that holds all parset-related functions
"""
import sys
import os
import glob
import logging
import configparser
from rapthor._logging import set_log_file
from astropy.coordinates import Angle
import multiprocessing

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
                       'solutions', 'plots']:
            subdir_path = os.path.join(parset_dict['dir_working'], subdir)
            if not os.path.isdir(subdir_path):
                os.mkdir(subdir_path)
    except Exception as e:
        log.critical("Cannot use the working dir {0}: {1}".format(parset_dict['dir_working'], e))
        sys.exit(1)
    if use_log_file:
        set_log_file(os.path.join(parset_dict['dir_working'], 'logs', 'rapthor.log'))
    log.info("=========================================================\n")
    log.info("CWLRunner is %s", parset_dict['cluster_specific']['cwl_runner'])
    log.info("Working directory is {}".format(parset_dict['dir_working']))

    # Get the input MS files
    ms_search_list = parset_dict['input_ms'].strip('[]').split(',')
    ms_search_list = [ms.strip() for ms in ms_search_list]
    ms_files = []
    for search_str in ms_search_list:
        ms_files += glob.glob(os.path.join(search_str))
    parset_dict['mss'] = sorted(ms_files)
    if len(parset_dict['mss']) == 0:
        log.error('No input MS files were found!')
        sys.exit(1)
    log.info("Working on {} input MS file(s)".format(len(parset_dict['mss'])))

    # Make sure the initial skymodel is present
    if 'input_skymodel' not in parset_dict:
        if parset_dict['download_initial_skymodel']:
            log.info('No input sky model file given and download requested. Will automatically download skymodel.')
            parset_dict.update({'input_skymodel': os.path.join(parset_dict['dir_working'], 'skymodels', 'initial_skymodel.txt')})
            if 'apparent_skymodel' in parset_dict:
                log.info('Ignoring apparent_skymodel, because skymodel download has been requested.')
                parset_dict['apparent_skymodel'] = None
        else:
            log.error('No input sky model file given and no download requested. Exiting...')
            raise RuntimeError('No input sky model file given and no download requested.')
    elif ('input_skymodel' in parset_dict) and parset_dict['download_initial_skymodel']:
        if not parset_dict['download_overwrite_skymodel']:
            # If download is requested, ignore the given skymodel.
            log.info('Skymodel download requested, but user-provided skymodel is present. Disabling download and using skymodel provided by the user.')
            parset_dict['download_initial_skymodel'] = False
        else:
            log.info('User-provided skymodel is present, but download_overwrite_skymodel is True. Overwriting user-supplied skymodel with downloaded one.')
            parset_dict['download_initial_skymodel'] = True
    elif not os.path.exists(parset_dict['input_skymodel']):
        log.error('Input sky model file "{}" not found. Exiting...'.format(parset_dict['input_skymodel']))
        raise FileNotFoundError('Input sky model file "{}" not found. Exiting...'.format(parset_dict['input_skymodel']))

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
    # TODO: Repalce all "if 'some_key' in parset_dict:" with "parset_dict.setdefault(...)"

    parset_dict = parset._sections['global'].copy()
    parset_dict.update({'calibration_specific': {}, 'imaging_specific': {}, 'cluster_specific': {}})

    # Fraction of data to use (default = 0.2). If less than one, the input data are divided
    # by time into chunks (of no less than slow_timestep_separate_sec below) that sum to the requested
    # fraction, spaced out evenly over the full time range
    if 'selfcal_data_fraction' in parset_dict:
        parset_dict['selfcal_data_fraction'] = parset.getfloat('global', 'selfcal_data_fraction')
    else:
        parset_dict['selfcal_data_fraction'] = 0.2
    if parset_dict['selfcal_data_fraction'] <= 0.0:
        log.error('The selfcal_data_fraction parameter is <= 0. It must be > 0 and <= 1')
        sys.exit(1)
    if parset_dict['selfcal_data_fraction'] > 1.0:
        log.error('The selfcal_data_fraction parameter is > 1. It must be > 0 and <= 1')
        sys.exit(1)
    if 'final_data_fraction' in parset_dict:
        parset_dict['final_data_fraction'] = parset.getfloat('global', 'final_data_fraction')
    else:
        parset_dict['final_data_fraction'] = parset_dict['selfcal_data_fraction']
    if parset_dict['final_data_fraction'] <= 0.0:
        log.error('The final_data_fraction parameter is <= 0. It must be > 0 and <= 1')
        sys.exit(1)
    if parset_dict['final_data_fraction'] > 1.0:
        log.error('The final_data_fraction parameter is > 1. It must be > 0 and <= 1')
        sys.exit(1)
    if parset_dict['final_data_fraction'] < parset_dict['selfcal_data_fraction']:
        log.warning('The final_data_fraction parameter is less than selfcal_data_fraction.')

    # Regroup input sky model (default = True)
    if 'regroup_input_skymodel' in parset_dict:
        parset_dict['regroup_input_skymodel'] = parset.getboolean('global', 'regroup_input_skymodel')
    else:
        parset_dict['regroup_input_skymodel'] = True

    # Apparent-flux input sky model (default = None)
    if 'apparent_skymodel' not in parset_dict:
        parset_dict['apparent_skymodel'] = None

    # Auto-download a sky model (default = True)?
    if 'download_initial_skymodel' not in parset_dict:
        parset_dict['download_initial_skymodel'] = True
    else:
        parset_dict['download_initial_skymodel'] = parset.getboolean('global', 'download_initial_skymodel')

    if 'download_initial_skymodel_radius' not in parset_dict:
        parset_dict['download_initial_skymodel_radius'] = 5.0
    else:
        parset_dict['download_initial_skymodel_radius'] = parset.getfloat('global', 'download_initial_skymodel_radius')

    if 'download_initial_skymodel_server' not in parset_dict:
        parset_dict['download_initial_skymodel_server'] = 'TGSS'

    if 'download_overwrite_skymodel' in parset_dict:
        parset_dict['download_overwrite_skymodel'] = parset.getboolean('global', 'download_overwrite_skymodel')
    else:
        parset_dict['download_overwrite_skymodel'] = False

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
                       'flag_expr', 'download_initial_skymodel', 'download_initial_skymodel_radius', 'download_initial_skymodel_server', 'download_overwrite_skymodel',
                       'input_skymodel', 'apparent_skymodel',
                       'regroup_input_skymodel', 'input_h5parm', 'selfcal_data_fraction',
                       'final_data_fraction']
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

    # Minimum uv distance in lambda for calibration (default = 350)
    if 'solve_min_uv_lambda' in parset_dict:
        parset_dict['solve_min_uv_lambda'] = parset.getfloat('calibration', 'solve_min_uv_lambda')
    else:
        parset_dict['solve_min_uv_lambda'] = 2000.0

    # Calculate the beam correction once per calibration patch (default = False)? If
    # False, the beam correction is calculated separately for each source in the patch.
    # Setting this to True can speed up calibration and prediction, but can also
    # reduce the quality of the calibration
    if 'onebeamperpatch' in parset_dict:
        parset_dict['onebeamperpatch'] = parset.getboolean('calibration', 'onebeamperpatch')
    else:
        parset_dict['onebeamperpatch'] = False

    # Solution intervals
    if 'fast_timestep_sec' in parset_dict:
        parset_dict['fast_timestep_sec'] = parset.getfloat('calibration', 'fast_timestep_sec')
    else:
        parset_dict['fast_timestep_sec'] = 8.0
    if 'fast_freqstep_hz' in parset_dict:
        parset_dict['fast_freqstep_hz'] = parset.getfloat('calibration', 'fast_freqstep_hz')
    else:
        parset_dict['fast_freqstep_hz'] = 1e6
    if 'slow_timestep_joint_sec' in parset_dict:
        parset_dict['slow_timestep_joint_sec'] = parset.getfloat('calibration', 'slow_timestep_joint_sec')
    else:
        parset_dict['slow_timestep_joint_sec'] = 0.0
    if 'slow_timestep_separate_sec' in parset_dict:
        parset_dict['slow_timestep_separate_sec'] = parset.getfloat('calibration', 'slow_timestep_separate_sec')
    else:
        parset_dict['slow_timestep_separate_sec'] = 600.0
    if parset_dict['slow_timestep_separate_sec'] < parset_dict['slow_timestep_joint_sec']:
        log.warning('The slow_timestep_separate_sec cannot be less than the slow_timestep_joint_sec.'
                    'Setting slow_timestep_separate_sec = slow_timestep_joint_sec')
        parset_dict['slow_timestep_separate_sec'] = parset_dict['slow_timestep_joint_sec']
    if 'slow_freqstep_hz' in parset_dict:
        parset_dict['slow_freqstep_hz'] = parset.getfloat('calibration', 'slow_freqstep_hz')
    else:
        parset_dict['slow_freqstep_hz'] = 1e6

    # Smoothness constraint
    if 'fast_smoothnessconstraint' in parset_dict:
        parset_dict['fast_smoothnessconstraint'] = parset.getfloat('calibration', 'fast_smoothnessconstraint')
    else:
        parset_dict['fast_smoothnessconstraint'] = 3e6
    if 'fast_smoothnessreffrequency' in parset_dict:
        parset_dict['fast_smoothnessreffrequency'] = parset.getfloat('calibration', 'fast_smoothnessreffrequency')
    else:
        parset_dict['fast_smoothnessreffrequency'] = None  # set later depending on whether data is HBA or LBA
    if 'fast_smoothnessrefdistance' in parset_dict:
        parset_dict['fast_smoothnessrefdistance'] = parset.getfloat('calibration', 'fast_smoothnessrefdistance')
    else:
        parset_dict['fast_smoothnessrefdistance'] = 0.0
    if 'slow_smoothnessconstraint_joint' in parset_dict:
        parset_dict['slow_smoothnessconstraint_joint'] = parset.getfloat('calibration', 'slow_smoothnessconstraint_joint')
    else:
        parset_dict['slow_smoothnessconstraint_joint'] = 3e6
    if 'slow_smoothnessconstraint_separate' in parset_dict:
        parset_dict['slow_smoothnessconstraint_separate'] = parset.getfloat('calibration', 'slow_smoothnessconstraint_separate')
    else:
        parset_dict['slow_smoothnessconstraint_separate'] = 3e6

    # Solver parameters
    if 'llssolver' not in parset_dict:
        parset_dict['llssolver'] = 'qr'
    if 'propagatesolutions' in parset_dict:
        parset_dict['propagatesolutions'] = parset.getboolean('calibration', 'propagatesolutions')
    else:
        parset_dict['propagatesolutions'] = True
    if 'solveralgorithm' not in parset_dict:
        parset_dict['solveralgorithm'] = 'hybrid'
    if 'maxiter' in parset_dict:
        parset_dict['maxiter'] = parset.getint('calibration', 'maxiter')
    else:
        parset_dict['maxiter'] = 150
    if 'stepsize' in parset_dict:
        parset_dict['stepsize'] = parset.getfloat('calibration', 'stepsize')
    else:
        parset_dict['stepsize'] = 0.02
    if 'tolerance' in parset_dict:
        parset_dict['tolerance'] = parset.getfloat('calibration', 'tolerance')
    else:
        parset_dict['tolerance'] = 5e-3

    # Parallel predict over baselines
    if 'parallelbaselines' in parset_dict:
        parset_dict['parallelbaselines'] = parset.getboolean('calibration', 'parallelbaselines')
    else:
        parset_dict['parallelbaselines'] = False

    # LBFGS solver parameters
    if 'solverlbfgs_dof' in parset_dict:
        parset_dict['solverlbfgs_dof'] = parset.getfloat('calibration', 'solverlbfgs_dof')
    else:
        parset_dict['solverlbfgs_dof'] = 200.0
    if 'solverlbfgs_iter' in parset_dict:
        parset_dict['solverlbfgs_iter'] = parset.getint('calibration', 'solverlbfgs_iter')
    else:
        parset_dict['solverlbfgs_iter'] = 4
    if 'solverlbfgs_minibatches' in parset_dict:
        parset_dict['solverlbfgs_minibatches'] = parset.getint('calibration', 'solverlbfgs_minibatches')
    else:
        parset_dict['solverlbfgs_minibatches'] = 1

    # Check for invalid options
    allowed_options = ['solve_min_uv_lambda', 'fast_timestep_sec',
                       'fast_freqstep_hz', 'slow_timestep_joint_sec',
                       'slow_timestep_separate_sec', 'onebeamperpatch',
                       'slow_freqstep_hz', 'propagatesolutions', 'maxiter', 'stepsize',
                       'tolerance', 'llssolver', 'fast_smoothnessconstraint',
                       'fast_smoothnessreffrequency', 'fast_smoothnessrefdistance',
                       'slow_smoothnessconstraint_joint',
                       'slow_smoothnessconstraint_separate', 'parallelbaselines',
                       'solveralgorithm', 'solverlbfgs_dof', 'solverlbfgs_iter',
                       'solverlbfgs_minibatches']
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
        parset_dict['grid_nsectors_ra'] = 0

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

    # Check that all the above options have the same number of entries
    if len(set(len_list)) > 1:
        log.error('The options sector_center_ra_list, sector_center_dec_list, '
                  'sector_width_ra_deg_list, and sector_width_dec_deg_list '
                  'must all have the same number of entries')
        sys.exit(1)

    # IDG (image domain gridder) mode to use in WSClean (default = cpu). The mode can
    # be cpu, gpu, or hybrid.
    if 'idg_mode' not in parset_dict:
        parset_dict['idg_mode'] = 'cpu'
    if parset_dict['idg_mode'] not in ['cpu', 'gpu', 'hybrid']:
        log.error('The option idg_mode must be one of "cpu", "gpu", or "hybrid"')
        sys.exit(1)

    # Method to use to apply direction-dependent effects during imaging: "none",
    # "facets", or "screens". If "none", the solutions closest to the image centers
    # will be used. If "facets", Voronoi faceting is used. If "screens", smooth 2-D
    # are used; the type of screen to use can be specified with screen_type:
    # "tessellated" (simple, smoothed tessellated screens) or "kl" (Karhunen-Lo`eve
    # screens) (default = kl)
    if 'dde_method' not in parset_dict:
        parset_dict['dde_method'] = 'facets'
    if parset_dict['dde_method'] not in ['none', 'screens', 'facets']:
        log.error('The option dde_method must be one of "none", "screens", or "facets"')
        sys.exit(1)
    if 'screen_type' not in parset_dict:
        parset_dict['screen_type'] = 'kl'
    if parset_dict['screen_type'] not in ['kl', 'tessellated']:
        log.error('The option screen_type must be one of "kl", or "tessellated"')
        sys.exit(1)

    # Fraction of the total memory (per node) to use for WSClean jobs (default = 0.9)
    if 'mem_fraction' in parset_dict:
        parset_dict['mem_fraction'] = parset.getfloat('imaging', 'mem_fraction')
    else:
        parset_dict['mem_fraction'] = 0.9

    # Use MPI during imaging (default = False).
    if 'use_mpi' in parset_dict:
        parset_dict['use_mpi'] = parset.getboolean('imaging', 'use_mpi')
    else:
        parset_dict['use_mpi'] = False

    # Reweight the visibility data before imaging (default = True)
    if 'reweight' in parset_dict:
        parset_dict['reweight'] = parset.getboolean('imaging', 'reweight')
    else:
        parset_dict['reweight'] = False

    # Max desired peak flux density reduction at center of the facet edges due to
    # bandwidth smearing (at the mean frequency) and time smearing (default = 0.15 =
    # 15% reduction in peak flux). Higher values result in shorter run times but
    # more smearing away from the facet centers. This value only applies to the
    # facet imaging (selfcal always uses a value of 0.15)
    if 'max_peak_smearing' in parset_dict:
        parset_dict['max_peak_smearing'] = parset.getfloat('imaging', 'max_peak_smearing')
    else:
        parset_dict['max_peak_smearing'] = 0.15

    # Imaging parameters: pixel size in arcsec (default = 1.25, suitable for HBA data), Briggs
    # robust parameter (default = -0.5), min and max uv distance in lambda (default = 0, none),
    # taper in arcsec (default = none), and whether multiscale clean should be used (default =
    # True)
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
        parset_dict['min_uv_lambda'] = 0.0
    if 'max_uv_lambda' in parset_dict:
        parset_dict['max_uv_lambda'] = parset.getfloat('imaging', 'max_uv_lambda')
    else:
        parset_dict['max_uv_lambda'] = 1e6
    if 'taper_arcsec' in parset_dict:
        parset_dict['taper_arcsec'] = parset.getfloat('imaging', 'taper_arcsec')
    else:
        parset_dict['taper_arcsec'] = 0.0
    if 'do_multiscale_clean' in parset_dict:
        parset_dict['do_multiscale_clean'] = parset.getboolean('imaging', 'do_multiscale_clean')
    else:
        parset_dict['do_multiscale_clean'] = True

    # Check for invalid options
    allowed_options = ['max_peak_smearing', 'cellsize_arcsec', 'robust', 'reweight',
                       'grid_center_ra', 'grid_center_dec',
                       'grid_width_ra_deg', 'grid_width_dec_deg', 'grid_nsectors_ra',
                       'min_uv_lambda', 'max_uv_lambda', 'mem_fraction', 'screen_type',
                       'robust', 'sector_center_ra_list', 'sector_center_dec_list',
                       'sector_width_ra_deg_list', 'sector_width_dec_deg_list',
                       'idg_mode', 'do_multiscale_clean', 'use_mpi',
                       'dde_method', 'skip_corner_sectors']
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

    # The number of processors and amount of memory per task to request from
    # SLURM can be specified with the cpus_per_task (default = 0 = all) and
    # mem_per_node_gb options (default = 190). By setting the cpus_per_task
    # value to the number of processors per node, one can ensure that each task
    # gets the entire node to itself, which is the recommended way of running
    # Rapthor
    if 'cpus_per_task' in parset_dict:
        parset_dict['cpus_per_task'] = parset.getint('cluster', 'cpus_per_task')
    else:
        parset_dict['cpus_per_task'] = 0
    if parset_dict['cpus_per_task'] == 0:
        parset_dict['cpus_per_task'] = multiprocessing.cpu_count()
    if 'mem_per_node_gb' in parset_dict:
        parset_dict['mem_per_node_gb'] = parset.getint('cluster', 'mem_per_node_gb')
    else:
        parset_dict['mem_per_node_gb'] = 0

    # Cluster type (default = single_machine). Use batch_system = slurm to use SLURM
    if 'batch_system' not in parset_dict:
        parset_dict['batch_system'] = 'single_machine'
    if 'max_nodes' in parset_dict:
        parset_dict['max_nodes'] = parset.getint('cluster', 'max_nodes')
    else:
        if parset_dict['batch_system'] == 'single_machine':
            parset_dict['max_nodes'] = 1
        else:
            parset_dict['max_nodes'] = 12

    # Maximum number of cores and threads per task to use on each node (default = 0 = all).
    if 'max_cores' in parset_dict:
        parset_dict['max_cores'] = parset.getint('cluster', 'max_cores')
    else:
        if parset_dict['batch_system'] == 'slurm':
            # If SLURM is used, set max_cores to be cpus_per_task
            parset_dict['max_cores'] = parset_dict['cpus_per_task']
        else:
            # Otherwise, get the cpu count of the current machine
            parset_dict['max_cores'] = multiprocessing.cpu_count()
    if parset_dict['max_cores'] == 0:
        parset_dict['max_cores'] = multiprocessing.cpu_count()
    if 'max_threads' in parset_dict:
        parset_dict['max_threads'] = parset.getint('cluster', 'max_threads')
    else:
        parset_dict['max_threads'] = 0
    if parset_dict['max_threads'] == 0:
        parset_dict['max_threads'] = multiprocessing.cpu_count()

    # Number of threads to use by WSClean during deconvolution and parallel gridding
    # (default = 0 = 2/5 of max_threads). Higher values will speed up imaging at the
    # expense of higher memory usage
    if 'deconvolution_threads' in parset_dict:
        parset_dict['deconvolution_threads'] = parset.getint('cluster', 'deconvolution_threads')
    else:
        parset_dict['deconvolution_threads'] = 0
    if parset_dict['deconvolution_threads'] == 0:
        parset_dict['deconvolution_threads'] = max(1, int(parset_dict['max_threads'] * 2 / 5))
    if 'parallel_gridding_threads' in parset_dict:
        parset_dict['parallel_gridding_threads'] = parset.getint('cluster', 'parallel_gridding_threads')
    else:
        parset_dict['parallel_gridding_threads'] = 0
    if parset_dict['parallel_gridding_threads'] == 0:
        parset_dict['parallel_gridding_threads'] = max(1, int(parset_dict['max_threads'] * 2 / 5))

    # Full path to a local disk on the nodes for I/O-intensive processing. The path
    # must be the same for all nodes
    if 'dir_local' not in parset_dict:
        parset_dict['dir_local'] = None
    else:
        parset_dict['dir_local'] = parset_dict['dir_local']

    # Run the pipelines inside a container (default = False)? If True, the pipeline
    # for each operation (such as calibrate or image) will be run inside a container.
    # The type of container can also be specified (one of docker, udocker, or
    # singularity; default = docker)
    if 'use_container' in parset_dict:
        parset_dict['use_container'] = parset.getboolean('cluster', 'use_container')
    else:
        parset_dict['use_container'] = False
    if 'container_type' not in parset_dict:
        parset_dict['container_type'] = 'docker'

    # Define CWL runner
    if 'cwl_runner' not in parset_dict:
        parset_dict['cwl_runner'] = 'toil'
    cwl_runner = parset_dict['cwl_runner']
    supported_cwl_runners = ('cwltool', 'toil')
    if cwl_runner not in supported_cwl_runners:
        log.critical("CWL runner '%s' is not supported; select one of: %s",
                     cwl_runner, ', '.join(supported_cwl_runners))
        sys.exit(1)

    # Check if debugging is enabled
    if 'debug_workflow' in parset_dict:
        parset_dict['debug_workflow'] = parset.getboolean('cluster', 'debug_workflow')
    else:
        parset_dict['debug_workflow'] = False

    # Check for invalid options
    allowed_options = ['cpus_per_task', 'batch_system', 'max_nodes', 'max_cores',
                       'max_threads', 'deconvolution_threads', 'parallel_gridding_threads', 'dir_local',
                       'mem_per_node_gb', 'use_container', 'container_type',
                       'cwl_runner', 'debug_workflow']
    for option in given_options:
        if option not in allowed_options:
            log.warning('Option "{}" was given in the [cluster] section of the '
                        'parset but is not a valid cluster option'.format(option))

    return parset_dict

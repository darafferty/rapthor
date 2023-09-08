"""
Module that holds all parset-related functions
"""
import os
import sys
import glob
import ast
import logging
import configparser
import astropy.coordinates
from rapthor._logging import set_log_file
import multiprocessing


if (sys.version_info.major, sys.version_info.minor) < (3, 9):
    import importlib_resources as resources
else:
    import importlib.resources as resources


log = logging.getLogger('rapthor:parset')


# def lexical_cast(string_value: str) -> any:
#     """
#     Try to cast `string_value` to a valid Python type, using `ast.literal_eval`.
#     This will only work for simple expressions. For more complex lists, try
#     to be smart ourselves. If conversion fails, return the unaltered input string.
#     """
#     try:
#         return ast.literal_eval(string_value)
#     except Exception:
#         # Try to convert a more complex list by converting each element ourselves.
#         if string_value.startswith("[") and string_value.endswith("]"):
#             values = [val.strip() for val in string_value.strip("[]").split(",")]
#             return [lexical_cast(val) for val in values]
#         # Return the unaltered string
#         return string_value

def lexical_cast(string_value: str) -> any:
    """
    Try to cast `string_value` to a valid Python type, using `ast.literal_eval`. This
    will only work for simple expressions. If that fails, try if `string_value` can be
    interpreted as an `astropy.coordinates.Angle`; if so, convert to degrees. If that
    fails, try if `string_value` represents a list of more complex items, and try to
    `lexical_cast` each item separately. If that also fails, return the input string.
    """
    try:
        return ast.literal_eval(string_value)
    except Exception:
        try:
            return astropy.coordinates.Angle(string_value).to("deg").value
        except Exception:
            # Try to convert a more complex list by converting each item separately
            if string_value.startswith("[") and string_value.endswith("]"):
                values = [val.strip() for val in string_value.strip("[]").split(",")]
                return [lexical_cast(val) for val in values]
    # Return the unaltered string
    return string_value


class Parset:
    """
    This class can read a Rapthor parset file and convert its contents to a `dict` that
    can be used elsewhere.
    """
    DEFAULT_PARSET = resources.files('rapthor') / 'settings' / 'defaults.parset'

    def __init__(self, parset_file=None):
        """
        Initializer.
        Read the default settings; define allowed and required sections and options
        based on these defaults. They will be used later to check for missing or
        invalid entries in the user-supplied parset file.

        Parameters
        ----------
        parset_file: str, optional
            Name of the parset-file to read
        """
        # Load default settings
        self.__parser = configparser.ConfigParser(interpolation=None)
        self.__parser.read_file(open(Parset.DEFAULT_PARSET))

        # Define allowed and required options here as a `dict`, where key is the name
        # of the section, and value is a `set` of key names for that section. Allowed
        # options are based on the contents of `defaults.parset`, required options are
        # hard-coded below. Also define sets of allowed and required sections.
        self.allowed_options = {
            sect: set(self.__parser.options(sect)) for sect in self.__parser.sections()
        }
        self.allowed_sections = set(self.allowed_options)
        self.required_options = {
            'global': {'dir_working', 'input_ms'},
            'calibration': {'use_included_skymodels'},  # FOR TESTING ONLY
            # 'burp': {'foo', 'bar'},  # FOR TESTING ONLY
        }
        self.required_sections = set(self.required_options)

        # Sanity check. Ensure that all required sections and options are also allowed.
        assert self.required_sections <= self.allowed_sections, \
            "%s <= %s" % (self.required_sections, self.allowed_sections)
        for section in self.required_options:
            assert self.required_options[section] <= self.allowed_options[section], \
            "%s <= %s" % (self.required_options[section], self.allowed_options[section])

        print("allowed_options =", self.allowed_options)
        print("allowed_sections =", self.allowed_sections)
        print("required_options =", self.required_options)
        print("required_sections =", self.required_sections)

        self.settings = Parset.__config_as_dict(self.__parser)

        if parset_file:
            self.settings = self.read_file(parset_file)

    @staticmethod
    def __config_as_dict(parser):
        """
        Return the current configuration as dictionary. The key is the section name,
        and the value is a dictionary of the options in the given section. 
        Option values are cast to their proper type, if possible.

        Parameters
        ----------
        parser: configparser.ConfigParser
            Configuration parser used to read a parset-file.
            
        Returns
        -------
        settings: dict
            Dictionary of the configuration: key is the section name, value is a 
            dictionary of the options in the given section.
        """
        settings = dict()
        for section in parser.sections():
            settings[section] = dict(
                (key, lexical_cast(value)) for key, value in parser.items(section)
            )
        return settings

    def __sanitize(self):
        """
        Check the configuration, as stored in `self.__parser`, for missing required
        sections and options; raise an exception if one or more are missing.
        Check for invalid, possibly misspelled, sections and options; print a warning,
        and remove these from the configuration.

        Raises
        ------
        ValueError
            If one or more required sections or options are missing
        """
        given_sections = set(self.__parser.sections())
        given_options = dict()
        for section in given_sections:
            given_options[section] = set(
                opt for opt in self.__parser.options(section)
                if self.__parser.get(section, opt) not in ("None", "")
            )
        missing_sections = self.required_sections - given_sections
        missing_options = {
            sect: self.required_options[sect] - given_options[sect]
            for sect in self.required_sections
        }
        invalid_sections = given_sections - self.allowed_sections
        invalid_options = {
            sect: given_options[sect] - self.allowed_options[sect]
            for sect in self.allowed_sections
        }
        
        print("---> given_sections =", given_sections)
        print("---> given_options =", given_options)
        print("---> missing_sections =", missing_sections)
        print("---> missing_options =", missing_options)            
        print("---> invalid_sections =", invalid_sections)
        print("---> invalid_options =", invalid_options)            

        # Check for missing required options.
        # NOTE: This will raise on the first section with missing required options.
        # Hence, if there are multiple sections with missing required options, not all
        # of them are reported. Currently only the [global] section has required
        # options, so this is not a big issue.
        for section, options in missing_options.items():
            if options:
                raise ValueError(
                    "Missing required option(s) in section [{}]: {}".format(
                        section, 
                        ", ".join("'{}'".format(opt) for opt in options)
                    )
                )

        # Check for invalid sections
        for section in invalid_sections:
            self.__parser.remove_section(section)
            log.warning("Section [%s] is invalid", section)
            
        # Check for invalid options
        for section in invalid_options:
            for option in invalid_options[section]:
                self.__parser.remove_option(section, option)
                log.warning("Option '%s' in section [%s] is invalid", option, section)
        
                
    def __check_and_adjust(self, settings):
        """
        Check for specific constraints on the option values in `settings`.
        Adjust values where needed.

        Parameters
        ----------
        settings: dict
            A dictionary of settings, where the key is the section name,
            and the value is a dictionary of the options in the given section.

        Raises
        ------
        ValueError
            If one or more option values do not meet with the constraints.
        """
        # Global options
        options = settings["global"]
        selfcal_data_fraction = options["selfcal_data_fraction"]
        if not 0.0 < selfcal_data_fraction <= 1.0:
            raise ValueError(
                f"The selfcal_data_fraction parameter is {selfcal_data_fraction}; "
                f"it must be > 0 and <= 1"
            )
        final_data_fraction = options["final_data_fraction"]
        if not 0.0 < final_data_fraction <= 1.0:
            raise ValueError(
                f"The final_data_fraction parameter is {final_data_fraction}. "
                f"It must be > 0 and <= 1"
            )
        if final_data_fraction < selfcal_data_fraction:
            log.warning(
                f"The final_data_fraction ({final_data_fraction}) "
                f"is less than selfcal_data_fraction ({selfcal_data_fraction})"
            )
        flag_list = [
            key for key in ("flag_abstime", "flag_baseline", "flag_freqrange")
            if options[key]
        ]
        if not options["flag_expr"]:
            options["flag_expr"] = " and ".join(flag_list)
        else:
            for flag in flag_list:
                if flag not in options['flag_expr']:
                    raise ValueError(
                        f"Flag selection '{flag}' was specified but does not "
                        f"appear in 'flag_expr'"
                    )

        # Calibration options
        options = settings["calibration"]
        if options["slow_timestep_separate_sec"] < options["slow_timestep_joint_sec"]:
            log.warning(
                "The slow_timestep_separate_sec cannot be less than the "
                "slow_timestep_joint_sec. Setting slow_timestep_separate_sec = "
                "slow_timestep_joint_sec"
            )
            options["slow_timestep_separate_sec"] = options["slow_timestep_joint_sec"]
            
        # Imaging options
        options = settings["imaging"]

        for opt, valid_values in {
            "idg_mode": ("cpu", "gpu", "hybrid"),
            "dde_method": ("none", "screens", "facets"),
            "screen_type": ("kl", "tesselated"),
        }.items():
            if options[opt] not in valid_values:
                raise ValueError(
                    "The option '{}' must be one of {}".format(
                        opt, ", ".join("'{}'".format(val) for val in valid_values)
                    )
                )

        if not(
            len(options["sector_center_ra_list"]) ==
            len(options["sector_center_dec_list"]) ==
            len(options["sector_width_ra_deg_list"]) ==
            len(options["sector_width_dec_deg_list"])
        ):
            raise ValueError(
                "The options 'sector_center_ra_list', 'sector_center_dec_list', "
                "'sector_width_ra_deg_list', and 'sector_width_dec_deg_list' "
                "must all have the same number of entries"
            )

        if len(options["dd_psf_grid"]) != 2:
            raise ValueError(
                "The option 'dd_psf_grid' must be a list of length 2 (e.g. '[3, 3]')"
            )
            
        # Cluster options
        options = settings["cluster"]

        for opt, valid_values in {
            "batch_system": ("single_machine", "slurm"),
            "cwl_runner": ("cwltool", "toil"),
        }.items():
            if options[opt] not in valid_values:
                raise ValueError(
                    "The option '{}' must be one of {}".format(
                        opt, ", ".join("'{}'".format(val) for val in valid_values)
                    )
                )

        cpu_count = multiprocessing.cpu_count()
        if not options["cpus_per_task"]:
            options["cpus_per_task"] = cpu_count

        single_machine = options["batch_system"] == "single_machine"
        cpus_per_task = options["cpus_per_task"]
        if not options["max_nodes"]:
            options["max_nodes"] = 1 if single_machine else 12
        if not options["max_cores"]:
            options["max_cores"] = cpu_count if single_machine else cpus_per_task
        if not options["max_threads"]:
            options["max_threads"] = cpu_count

        max_threads = options["max_threads"]
        if not options["deconvolution_threads"]:
            options["deconvolution_threads"] = max(1, max_threads * 2 // 5)
        if not options["parallel_gridding_threads"]:
            options["parallel_gridding_threads"] = max(1, max_threads * 2 // 5)

    # def __update_settings(self, settings):
    #     """
    #     Update the locally cached settings (`self.settings`) with the contents of
    #     `settings`.

    #     Parameters
    #     ----------
    #     settings: dict
    #         A dictionary of settings, where the key is the section name,
    #         and the value is a dictionary of the options in the given section.        
    #     """
    #     for section in settings:
    #         self.settings[section].update(settings[section])

    # def __validate(self, settings):
    #     """
    #     Validate the provided `settings`. Validation consists of two step: 
    #     - sanitization: check for missing required and for invalid sections/options;
    #     - constraint checking: check for additional constraints on some option values.

    #     Parameters
    #     ----------
    #     settings: dict
    #         A dictionary of settings, where the key is the section name,
    #         and the value is a dictionary of the options in the given section.

    #     Returns
    #     -------
    #     settings: dict
    #         Validated settings.

    #     """
    #     # settings = 
    #     self.__sanitize_settings(settings)
    #     self.__check_and_adjust(settings)
    #     return settings


    def read_file(self, parset_file):
        """
        Read the contents of `parset_file`. The format of this file is often referred to
        as parset format in the context of Rapthor, but is actually in the well-known
        Windows INI-format. A check is made for missing required sections and options. 
        Given options values are checked against possible constraints.

        Parameters
        ----------
        parset_file: str
            Name of the parset file
            
        Returns
        -------
        settings: dict
            A dictionary of settings, where the key is the section name,
            and the value is a dictionary of the options in the given section.
            Option values are cast to their proper type, if possible.

        Raises
        ------
        FileNotFoundError
            If `parset_file` does not exist
        ValueError
            If `parset_file` cannot be parsed correctly, or if one or more required
            sections or options are missing, or if it contains invalid option values.
        """
        log.info("Reading parset file: %s", parset_file)
        # settings = Parset.__config_as_dict(self.__parser)
        # print("settings_0 =", settings)
        try:
            if not self.__parser.read(parset_file):
                raise FileNotFoundError(f"Missing parset file ({parset_file}).")
        except configparser.ParsingError as err:
            raise ValueError(
                f"Parset file '{parset_file}' could not be parsed correctly.\n{err}"
            )

        self.__sanitize()

        settings = Parset.__config_as_dict(self.__parser)
        # print("settings_1 =", settings)
        self.__check_and_adjust(settings)
        # print("settings_2 =", settings)
        return settings

    def as_parset_dict(self):
        """
        Return the current settings as parset-dict. The parset-dict differs from the
        internal settings in the sense that all the key/value pairs defined in the
        [global] section are put at the top-level instead of under 'global'. All other
        section names (like [imaging]), get a post-fix '_specific'.
        """
        parset = self.settings["global"]
        for section in self.settings: 
            if section != "global": 
                parset[section + "_specific"] = self.settings[section]
        return parset
    

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

    parset_dict = Parset(parset_file).as_parset_dict()

    # Set up working directory. All output will be placed in this directory
    try:
        if not os.path.isdir(parset_dict['dir_working']):
            os.mkdir(parset_dict['dir_working'])
        for subdir in ['logs', 'pipelines', 'regions', 'skymodels', 'images',
                       'solutions', 'plots']:
            subdir_path = os.path.join(parset_dict['dir_working'], subdir)
            if not os.path.isdir(subdir_path):
                os.mkdir(subdir_path)
    except Exception as e:
        raise RuntimeError("Cannot use the working dir {0}: {1}".format(parset_dict['dir_working'], e))
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
        raise FileNotFoundError('No input MS files were found (searched for files '
                                'matching: {}).'.format(', '.join('"{0}"'.format(search_str)
                                                                  for search_str in ms_search_list)))
    log.info("Working on {} input MS file(s)".format(len(parset_dict['mss'])))

    # Make sure the initial skymodel is present
    if not parset_dict['input_skymodel']:
        if parset_dict['download_initial_skymodel']:
            log.info('No input sky model file given and download requested. Will automatically download skymodel.')
            parset_dict.update({'input_skymodel': os.path.join(parset_dict['dir_working'], 'skymodels', 'initial_skymodel.txt')})
            if parset_dict['apparent_skymodel']:
                log.info('Ignoring apparent_skymodel, because skymodel download has been requested.')
                parset_dict['apparent_skymodel'] = None
        else:
            log.error('No input sky model file given and no download requested. Exiting...')
            raise RuntimeError('No input sky model file given and no download requested.')
    elif (parset_dict['input_skymodel']) and parset_dict['download_initial_skymodel']:
        if not parset_dict['download_overwrite_skymodel']:
            # If download is requested, ignore the given skymodel.
            log.info('Skymodel download requested, but user-provided skymodel is present. Disabling download and using skymodel provided by the user.')
            parset_dict['download_initial_skymodel'] = False
        else:
            log.info('User-provided skymodel is present, but download_overwrite_skymodel is True. Overwriting user-supplied skymodel with downloaded one.')
            parset_dict['download_initial_skymodel'] = True
    elif not os.path.exists(parset_dict['input_skymodel']):
        raise FileNotFoundError('Input sky model file "{}" not found.'.format(parset_dict['input_skymodel']))

    return parset_dict


# def handle_global_options(parset_dict):
#     """
#     """
#     print("===> parset_dict =", parset_dict)

def get_global_options(settings):
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
    parset_dict = settings

    if not 0.0 < parset_dict['selfcal_data_fraction'] <= 1.0:
        raise ValueError(
            f"The selfcal_data_fraction parameter is {parset_dict['selfcal_data_fraction']}. "
            f"It must be > 0 and <= 1"
        )
    if not 0.0 < parset_dict['final_data_fraction'] <= 1.0:
        raise ValueError(
            f"The final_data_fraction parameter is {parset_dict['final_data_fraction']}. "
            f"It must be > 0 and <= 1"
        )
    if parset_dict['final_data_fraction'] < parset_dict['selfcal_data_fraction']:
        log.warning(
            f"The final_data_fraction parameter ({parset_dict['final_data_fraction']}) "
            f"is less than selfcal_data_fraction ({parset_dict['selfcal_data_fraction']})"
        )

    flag_list = [key for key in ('flag_abstime', 'flag_baseline', 'flag_freqrange') if parset_dict[key]]
    if not parset_dict['flag_expr']:
        parset_dict['flag_expr'] = ' and '.join(flag_list)
    else:
        for flag in flag_list:
            if flag not in parset_dict['flag_expr']:
                raise ValueError(f"Flag selection '{flag}' was specified but does not "
                                 f"appear in flag_expr")

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
    # raise Exception("AAARGH: get_calibration_options !!!!")
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

    # Use SAGECalpredict
    if 'sagecalpredict' in parset_dict:
        parset_dict['sagecalpredict'] = parset.getboolean('calibration', 'sagecalpredict')
    else:
        parset_dict['sagecalpredict'] = False

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
    allowed_options = ['fast_timestep_sec', 'fast_freqstep_hz', 'slow_timestep_joint_sec',
                       'slow_timestep_separate_sec', 'onebeamperpatch',
                       'slow_freqstep_hz', 'propagatesolutions', 'maxiter', 'stepsize',
                       'tolerance', 'llssolver', 'fast_smoothnessconstraint',
                       'fast_smoothnessreffrequency', 'fast_smoothnessrefdistance',
                       'slow_smoothnessconstraint_joint',
                       'slow_smoothnessconstraint_separate', 'parallelbaselines',
                       'solveralgorithm', 'solverlbfgs_dof', 'solverlbfgs_iter',
                       'solverlbfgs_minibatches','sagecalpredict']
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
    # raise Exception("AAARGH: get_imaging_options !!!!")
    if 'imaging' in list(parset._sections.keys()):
        parset_dict = parset._sections['imaging']
        given_options = parset.options('imaging')
    else:
        parset_dict = {}
        given_options = []

    # Size of area to image when using a grid (default = 1.7 * mean FWHM of the primary beam)
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
    len_list = []
    if 'sector_center_ra_list' in parset_dict:
        val_list = parset_dict['sector_center_ra_list'].strip('[]').split(',')
        val_list = [Angle(v).to('deg').value for v in val_list]
        parset_dict['sector_center_ra_list'] = val_list
        len_list.append(len(val_list))
    else:
        parset_dict['sector_center_ra_list'] = []
    if 'sector_center_dec_list' in parset_dict:
        val_list = parset_dict['sector_center_dec_list'].strip('[]').split(',')
        val_list = [Angle(v).to('deg').value for v in val_list]
        parset_dict['sector_center_dec_list'] = val_list
        len_list.append(len(val_list))
    else:
        parset_dict['sector_center_dec_list'] = []
    if 'sector_width_ra_deg_list' in parset_dict:
        val_list = parset_dict['sector_width_ra_deg_list'].strip('[]').split(',')
        val_list = [float(v) for v in val_list]
        parset_dict['sector_width_ra_deg_list'] = val_list
        len_list.append(len(val_list))
    else:
        parset_dict['sector_width_ra_deg_list'] = []
    if 'sector_width_dec_deg_list' in parset_dict:
        val_list = parset_dict['sector_width_dec_deg_list'].strip('[]').split(',')
        val_list = [float(v) for v in val_list]
        parset_dict['sector_width_dec_deg_list'] = val_list
        len_list.append(len(val_list))
    else:
        parset_dict['sector_width_dec_deg_list'] = []

    # Check that all the above options have the same number of entries
    if len(set(len_list)) > 1:
        raise ValueError('The options sector_center_ra_list, sector_center_dec_list, '
                         'sector_width_ra_deg_list, and sector_width_dec_deg_list '
                         'must all have the same number of entries')

    # IDG (image domain gridder) mode to use in WSClean (default = cpu). The mode can
    # be cpu, gpu, or hybrid.
    if 'idg_mode' not in parset_dict:
        parset_dict['idg_mode'] = 'cpu'
    if parset_dict['idg_mode'] not in ['cpu', 'gpu', 'hybrid']:
        raise ValueError('The option idg_mode must be one of "cpu", "gpu", or "hybrid"')

    # Method to use to apply direction-dependent effects during imaging: "none",
    # "facets", or "screens". If "none", the solutions closest to the image centers
    # will be used. If "facets", Voronoi faceting is used. If "screens", smooth 2-D
    # are used; the type of screen to use can be specified with screen_type:
    # "tessellated" (simple, smoothed tessellated screens) or "kl" (Karhunen-Lo`eve
    # screens) (default = kl)
    if 'dde_method' not in parset_dict:
        parset_dict['dde_method'] = 'facets'
    if parset_dict['dde_method'] not in ['none', 'screens', 'facets']:
        raise ValueError('The option dde_method must be one of "none", "screens", or "facets"')
    if 'screen_type' not in parset_dict:
        parset_dict['screen_type'] = 'kl'
    if parset_dict['screen_type'] not in ['kl', 'tessellated']:
        raise ValueError('The option screen_type must be one of "kl", or "tessellated"')

    # Maximum memory in GB (per node) to use for WSClean jobs (default = 0 = 100%)
    if 'mem_gb' in parset_dict:
        parset_dict['mem_gb'] = parset.getfloat('imaging', 'mem_gb')
    else:
        parset_dict['mem_gb'] = 0

    # Apply separate XX and YY corrections during imaging (default = True). If False,
    # scalar solutions (the average of the XX and YY solutions) are applied instead
    if 'apply_diagonal_solutions' in parset_dict:
        parset_dict['apply_diagonal_solutions'] = parset.getboolean('imaging', 'apply_diagonal_solutions')
    else:
        parset_dict['apply_diagonal_solutions'] = True

    # The number of direction-dependent PSFs which should be fit horizontally and
    # vertically in the image (default = [1, 1] = direction-independent PSF).
    if 'dd_psf_grid' in parset_dict:
        val_list = parset_dict['dd_psf_grid'].strip('[]').split(',')
        parset_dict['dd_psf_grid'] = [int(v) for v in val_list]
    else:
        parset_dict['dd_psf_grid'] = [1, 1]
    if len(parset_dict['dd_psf_grid']) != 2:
        raise ValueError('The option dd_psf_grid must be a list of length 2 (e.g. "[3, 3]")')

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
                       'grid_center_ra', 'grid_center_dec', 'apply_diagonal_solutions',
                       'grid_width_ra_deg', 'grid_width_dec_deg', 'grid_nsectors_ra',
                       'min_uv_lambda', 'max_uv_lambda', 'mem_gb', 'screen_type',
                       'robust', 'sector_center_ra_list', 'sector_center_dec_list',
                       'sector_width_ra_deg_list', 'sector_width_dec_deg_list',
                       'idg_mode', 'do_multiscale_clean', 'use_mpi',
                       'dde_method', 'skip_corner_sectors', 'dd_psf_grid']
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
    raise Exception("AAARGH: get_cluster_options !!!!")
    if 'cluster' in list(parset._sections.keys()):
        parset_dict = parset._sections['cluster']
        given_options = parset.options('cluster')
    else:
        parset_dict = {}
        given_options = []
    # raise Exception ("AAAARGH")
    # parset_dict = dict(parset.items('cluster'))
    # given_options = parset.options('cluster')

    print ("++ cluster ++: parset_dict =", parset_dict)

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

    # Run the workflows inside a container (default = False)? If True, the CWL workflow
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
        raise ValueError("CWL runner '%s' is not supported; select one of: %s",
                         cwl_runner, ', '.join(supported_cwl_runners))

    # Set Toil's coordination directory (only used when Toil is used as the CWL runner;
    # default = selected automatically by Toil). Note that this directory does not
    # need to be on a shared file system but must be located one that is 100%
    # POSIX-compatible (which many shared file systems are not)
    if 'dir_coordination' not in parset_dict:
        parset_dict['dir_coordination'] = None
    else:
        parset_dict['dir_coordination'] = parset_dict['dir_coordination']

    # Check if debugging is enabled
    if 'debug_workflow' in parset_dict:
        parset_dict['debug_workflow'] = parset.getboolean('cluster', 'debug_workflow')
    else:
        parset_dict['debug_workflow'] = False

    # Check for invalid options
    allowed_options = ['cpus_per_task', 'batch_system', 'max_nodes', 'max_cores',
                       'max_threads', 'deconvolution_threads', 'parallel_gridding_threads', 'dir_local',
                       'mem_per_node_gb', 'use_container', 'container_type',
                       'cwl_runner', 'dir_coordination', 'debug_workflow']
    for option in given_options:
        if option not in allowed_options:
            log.warning('Option "{}" was given in the [cluster] section of the '
                        'parset but is not a valid cluster option'.format(option))

    return parset_dict

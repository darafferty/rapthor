"""
Module that holds all parset-related functions
"""
import ast
import configparser
import glob
import logging
import os
from importlib import resources

import astropy.coordinates
import rapthor.lib.miscellaneous as misc
from rapthor._logging import set_log_file
from rapthor._version import __version__

log = logging.getLogger("rapthor:parset")


def lexical_cast(string_value):
    """
    Try to cast `string_value` to a valid Python type, using `ast.literal_eval`. This
    will only work for simple expressions. If that fails, try if `string_value` can be
    interpreted as an `astropy.coordinates.Angle`; if so, convert to degrees. If that
    fails, try if `string_value` represents a list of more complex items, and try to
    `lexical_cast` each item separately. If that also fails, return the input string.
    """
    if not isinstance(string_value, str):
        raise TypeError("lexical_cast requires a string argument")
    try:
        # Try to convert to simple Python types, like `int`, `float`, `list`, etc.
        return ast.literal_eval(string_value)
    except (ValueError, TypeError, SyntaxError):
        try:
            # Try to convert to an angle, and return value in degrees
            return astropy.coordinates.Angle(string_value).to("deg").value
        except (ValueError, TypeError):
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

    Notes
    -----
    If you want to add a new option, make sure that this option is added to the file
    `defaults.parset`. Any option used by Rapthor _must_ be defined in that file with
    its default value. If it is not in that file, a warning will be issued when it is
    used in a user-supplied parset file, and its value will be ignored. Similarly for a
    new section.
    If you want to add extra constraint checks for an option, or want to adjust its
    value, you need to add these checks to the method  `__check_and_adjust`.
    If you want to add a new _required_ option, an extra entry must be added to
    the attribute `self.required_options` in the method `__init__`. Similarly for a new
    _required_ section.
    """

    DEFAULT_PARSET = resources.files("rapthor") / "settings" / "defaults.parset"

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
            "global": {"dir_working", "input_ms"},
            # 'calibration': {'use_included_skymodels'},  # FOR TESTING ONLY
            # 'burp': {'foo', 'bar'},  # FOR TESTING ONLY
        }
        self.required_sections = set(self.required_options)

        # Deprecated options are hard-coded below. Each deprecated option can have
        # zero or more suggestions for alternative options.
        self.deprecated_options = {
            "cluster": {
                "dir_local": {"local_scratch_dir"}
            }
        }

        # Sanity check. Ensure that all required sections and options are also allowed.
        assert self.required_sections <= self.allowed_sections, "%s <= %s" % (
            self.required_sections,
            self.allowed_sections,
        )
        for section in self.required_options:
            assert (
                self.required_options[section] <= self.allowed_options[section]
            ), "%s <= %s" % (
                self.required_options[section],
                self.allowed_options[section],
            )

        self.settings = Parset.config_as_dict(self.__parser)

        if parset_file:
            self.settings = self.read_file(parset_file)

    @staticmethod
    def config_as_dict(parser):
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
                opt
                for opt in self.__parser.options(section)
                if self.__parser.get(section, opt) not in ("None", "")
            )
        # Currently, `missing_sections` is unused, but we may need it in the future
        missing_sections = self.required_sections - given_sections  # noqa: F841
        missing_options = {
            sect: self.required_options[sect] - given_options[sect]
            for sect in self.required_sections
        }
        invalid_sections = given_sections - self.allowed_sections
        invalid_options = {
            sect: given_options[sect] - self.allowed_options[sect]
            for sect in self.allowed_sections
        }
        deprecated_options = {
            sect: set(self.deprecated_options[sect]) & given_options[sect]
            for sect in self.deprecated_options
        }

        # Check for missing required options.
        # NOTE: This will raise on the first section with missing required options.
        # Hence, if there are multiple sections with missing required options, not all
        # of them are reported. Currently only the [global] section has required
        # options, so this is not a big issue.
        for section, options in missing_options.items():
            if options:
                raise ValueError(
                    "Missing required option(s) in section [{}]: {}".format(
                        section, ", ".join("'{}'".format(opt) for opt in options)
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

        # Check for deprecated options
        for section in deprecated_options:
            for option in deprecated_options[section]:
                alternatives = self.deprecated_options[section][option]
                message = f"Option '{option}' in section [{section}] is deprecated"
                if alternatives:
                    message += "; use %s instead" % (
                        ", or ".join("'{}'".format(opt) for opt in alternatives)
                    )
                log.warning(message)

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
        if options["generate_initial_skymodel"] and options["download_initial_skymodel"]:
            raise ValueError(
                "Both 'generate_initial_skymodel' and 'download_initial_skymodel' are "
                "activated. Only one of these options can be active."
            )
        if (
            options["regroup_input_skymodel"]
            and options["input_skymodel"]
            and options["input_h5parm"]
            and not options["facet_layout"]
        ):
            raise ValueError(
                "Regrouping of the input sky model was activated, but regrouping "
                "cannot be done when input solutions are provided unless a facet "
                "layout file is also provided. This restriction ensures that the "
                "directions defined for the solutions match the patches defined in "
                "the sky model"
            )

        # Calibration options
        options = settings["calibration"]

        for opt, valid_values in {
            "fast_datause": ("single", "dual", "full"),
            "medium_datause": ("single", "dual", "full"),
            "slow_datause": ("dual", "full"),
            "solveralgorithm": ("hybrid", "lbfgs", "directioniterative", "directionsolve"),
        }.items():
            if options[opt] not in valid_values:
                raise ValueError(
                    "The option '{}' must be one of {}".format(
                        opt, ", ".join("'{}'".format(val) for val in valid_values)
                    )
                )

        dd_smoothness_factor = options["dd_smoothness_factor"]
        if dd_smoothness_factor < 1:
            raise ValueError(
                f"The dd_smoothness_factor parameter is {dd_smoothness_factor}; "
                f"it must be >= 1"
            )
        dd_interval_factor = options["dd_interval_factor"]
        solveralgorithm = options["solveralgorithm"]
        if dd_interval_factor < 1:
            raise ValueError(
                f"The dd_interval_factor parameter is {dd_interval_factor}; "
                f"it must be >= 1"
            )
        elif dd_interval_factor > 1 and solveralgorithm != "directioniterative":
            log.warning(
                f"Switching from the '{solveralgorithm}' solver to the "
                "'directioniterative' solver, since dd_interval_factor > 1."
            )
            options["solveralgorithm"] = "directioniterative"
        if dd_interval_factor > 1:
            # TODO: direction-dependent solution intervals cannot yet be used with
            # multi-calibration; once they can be, the restriction on their use
            # should be removed
            log.warning(
                "Switching off direction-dependent intervals, since they are "
                "not yet supported."
            )
            options["dd_interval_factor"] = 1
        if (
            (options["fast_datause"] != "full" or options["medium_datause"] != "full" or options["slow_datause"] != "full") and
            options["solveralgorithm"] != "directioniterative"
        ):
            log.warning(
                f"Switching from the '{solveralgorithm}' solver to the "
                "'directioniterative' solver, since single or dual visibilities "
                "solving is activated."
            )
            options["solveralgorithm"] = "directioniterative"

        if (
            options['use_image_based_predict'] and
            (
                options['bda_timebase'] > 0 or
                options['bda_frequencybase'] > 0
            )
        ):
            log.warning(
                "Switching off BDA during solving, since image-based predict is "
                "activated."
            )
            options['bda_timebase'] = 0
            options['bda_frequencybase'] = 0

        # Imaging options
        options = settings["imaging"]
        
        for opt, valid_values in {
            "idg_mode": ("cpu", "gpu", "hybrid"),
            "dde_method": ("single", "full"),
            "pol_combine_method": ("link", "join"),
        }.items():
            if options[opt] not in valid_values:
                raise ValueError(
                    "The option '{}' must be one of {}".format(
                        opt, ", ".join("'{}'".format(val) for val in valid_values)
                    )
                )

        if not (
            len(options["sector_center_ra_list"])
            == len(options["sector_center_dec_list"])
            == len(options["sector_width_ra_deg_list"])
            == len(options["sector_width_dec_deg_list"])
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

        if (
            options["correct_time_frequency_smearing"] and
            options["bda_timebase"] > 0
        ):
            log.warning(
                "Switching off BDA during imaging, since correction for time "
                "and frequency smearing is activated."
            )
            options["bda_timebase"] = 0
            # options["bda_frequencybase"] = 0  # TODO: also disable frequency BDA once implemented

        if (
            settings["imaging"]["correct_time_frequency_smearing"] !=
            settings["calibration"]["correct_time_frequency_smearing"]
        ):
            log.warning(
                "Correction for time and frequency smearing is enabled "
                "in either calibration or imaging, but not in both. "
                "Generally, the correction should be enabled (or disabled) "
                "together in both sections."
            )

        options["image_cube_stokes_list"] = [
            pol.upper() for pol in options["image_cube_stokes_list"]
        ]
        if any([pol not in "IQUV" for pol in options["image_cube_stokes_list"]]):
            raise ValueError(
                "The option 'image_cube_stokes_list' specifies one or more invalid "
                "Stokes parameters. Allowed Stokes parameters are 'I', 'Q', 'U', or "
                "'V'."
            )

        if options["save_image_cube"] and options["image_cube_stokes_list"] == []:
            log.warning(
                "The option 'save_image_cube' is enabled, but 'image_cube_stokes_list' "
                "is empty. Setting 'image_cube_stokes_list' to '[I]'."
            )
            options["image_cube_stokes_list"] = ["I"]
        if (
            options["save_image_cube"]
            and not options["make_quv_images"]
            and options["image_cube_stokes_list"] != ["I"]
        ):
            raise ValueError(
                "The option 'image_cube_stokes_list' specifies that a cube for a "
                "Stokes parameter other than I should be saved, but non-Stokes-I "
                "images will not be made (since 'make_quv_images' is not enabled). "
                "Please enable 'make_quv_images' or set 'image_cube_stokes_list' to "
                "'[I]'."
            )

        # Cluster options
        options = settings["cluster"]

        for opt, valid_values in {
            "batch_system": ("single_machine", "slurm", "slurm_static"),
            "cwl_runner": ("cwltool", "streamflow", "toil"),
        }.items():
            if options[opt] not in valid_values:
                raise ValueError(
                    "The option '{}' must be one of {}".format(
                        opt, ", ".join("'{}'".format(val) for val in valid_values)
                    )
                )

        cpu_count = misc.nproc()
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
            options["deconvolution_threads"] = max(1, min(14, max_threads * 2 // 5))
        if not options["parallel_gridding_threads"]:
            options["parallel_gridding_threads"] = max(1, min(6, max_threads * 2 // 5))

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
        try:
            if not self.__parser.read(parset_file):
                raise FileNotFoundError(f"Missing parset file ({parset_file}).")
        except configparser.ParsingError as err:
            raise ValueError(
                f"Parset file '{parset_file}' could not be parsed correctly.\n{err}"
            )

        self.__sanitize()
        settings = Parset.config_as_dict(self.__parser)
        self.__check_and_adjust(settings)

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


def parset_read(parset_file, use_log_file=True):
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

    Raises
    ------
        RuntimeError
            If the working directory cannot be created.
        FileNotFoundError
            If no input MS files can be found; or
            if the sky model file cannot be found and download is not requested.
    """

    parset_dict = Parset(parset_file).as_parset_dict()

    # Set up working directory. All output will be placed in this directory
    try:
        if not os.path.isdir(parset_dict["dir_working"]):
            os.mkdir(parset_dict["dir_working"])
        for subdir in (
            "logs",
            "pipelines",
            "regions",
            "skymodels",
            "images",
            "solutions",
            "plots",
        ):
            subdir_path = os.path.join(parset_dict["dir_working"], subdir)
            if not os.path.isdir(subdir_path):
                os.mkdir(subdir_path)
    except Exception as e:
        raise RuntimeError(
            "Cannot use the working dir {0}: {1}".format(parset_dict["dir_working"], e)
        )
    if use_log_file:
        set_log_file(os.path.join(parset_dict["dir_working"], "logs", "rapthor.log"))
    log.info("=========================================================")
    log.info("Rapthor version %s", __version__)
    log.info("CWLRunner is %s", parset_dict["cluster_specific"]["cwl_runner"])
    log.info("Working directory is {}".format(parset_dict["dir_working"]))

    # Get the input MS files; it can either be a string, or a list of strings
    input_ms = parset_dict["input_ms"]
    ms_search_list = [input_ms] if isinstance(input_ms, str) else input_ms
    ms_files = []
    for search_str in ms_search_list:
        ms_files += glob.glob(os.path.join(search_str))
    parset_dict["mss"] = sorted(ms_files)
    if len(parset_dict["mss"]) == 0:
        raise FileNotFoundError(
            "No input MS files were found (searched for files "
            "matching: {}).".format(
                ", ".join('"{0}"'.format(search_str) for search_str in ms_search_list)
            )
        )
    suffix = 's' if len(parset_dict["mss"]) > 1 else ''
    log.info("Working on {0} input MS file{1}".format(len(parset_dict["mss"]), suffix))

    check_skymodel_settings(parset_dict)
    log.info("=========================================================")

    return parset_dict


def check_skymodel_settings(parset_dict):
    """
    En‌sure·‌the·‌initial·‌sky·‌model·‌is·‌present·‌or,·‌if·‌not,·‌that·‌generation·‌or
    download·‌is·‌requested

    Parameters
    ----------
    parset_dict : dict
        Dictionary containing parset parameters

    Raises
    ------
    FileNotFoundError
        If the input sky model file is not found.
    """
    
    if parset_dict["input_skymodel"]:
        if not os.path.exists(parset_dict["input_skymodel"]):
            raise FileNotFoundError(
                f'Input sky model file "{parset_dict["input_skymodel"]}" not found.'
            )
        if parset_dict["generate_initial_skymodel"]:
            # If sky model is given but generation requested, disable generation and use
            # the given skymodel.
            log.warning(
                "Sky model generation requested, but user-provided sky model is present. "
                "Disabling generation and using sky model provided by the user."
            )
            parset_dict["generate_initial_skymodel"] = False
        elif parset_dict["download_initial_skymodel"]:
            # If sky model is given but download requested, use the given skymodel and
            # disable download.
            log.warning(
                "Sky model download requested, but user-provided sky model is present. "
                "Disabling download and using sky model provided by the user."
            )
            parset_dict["download_initial_skymodel"] = False

    elif parset_dict["generate_initial_skymodel"]:
        log.info(
            "No input sky model file given and generation requested. "
            "Will automatically generate sky model from input data."
        )
        if parset_dict["apparent_skymodel"]:
            log.warning(
                "The input apparent sky model will not be used "
                "because sky model generation has been requested."
            )
    elif parset_dict["download_initial_skymodel"]:
        log.info(
            "No input sky model file given and download requested. "
            "Will automatically download sky model."
        )
        if parset_dict["apparent_skymodel"]:
            log.warning(
                "The input apparent sky model will not be used "
                "because sky model download has been requested."
            )
    else:
        log.warning(
            "No input sky model file given and neither generation nor download of "
            "sky model requested. If no calibration is to be done, this warning can "
            "be ignored."
        )
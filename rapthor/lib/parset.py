"""
Module that holds all parset-related functions
"""
import ast
import configparser
import glob
import logging
import multiprocessing
import os
import sys

import astropy.coordinates
from rapthor._logging import set_log_file

if (sys.version_info.major, sys.version_info.minor) < (3, 9):
    import importlib_resources as resources
else:
    import importlib.resources as resources


log = logging.getLogger("rapthor:parset")


def lexical_cast(string_value):
    """
    Try to cast `string_value` to a valid Python type, using `ast.literal_eval`. This
    will only work for simple expressions. If that fails, try if `string_value` can be
    interpreted as an `astropy.coordinates.Angle`; if so, convert to degrees. If that
    fails, try if `string_value` represents a list of more complex items, and try to
    `lexical_cast` each item separately. If that also fails, return the input string.
    """
    try:
        # Try to convert to simple Python types, like `int`, `float`, `list`, etc.
        return ast.literal_eval(string_value)
    except Exception:
        try:
            # Try to convert to an angle, and return value in degrees
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
                opt
                for opt in self.__parser.options(section)
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
            key
            for key in ("flag_abstime", "flag_baseline", "flag_freqrange")
            if options[key]
        ]
        if not options["flag_expr"]:
            options["flag_expr"] = " and ".join(flag_list)
        else:
            for flag in flag_list:
                if flag not in options["flag_expr"]:
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
        settings = Parset.__config_as_dict(self.__parser)
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
            If the working directory cannot be created; or
            if no input sky model file is given and download is not requested.
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
    log.info("=========================================================\n")
    log.info("CWLRunner is %s", parset_dict["cluster_specific"]["cwl_runner"])
    log.info("Working directory is {}".format(parset_dict["dir_working"]))

    # Get the input MS files
    ms_search_list = parset_dict["input_ms"].strip("[]").split(",")
    ms_search_list = [ms.strip() for ms in ms_search_list]
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
    log.info("Working on {} input MS file(s)".format(len(parset_dict["mss"])))

    # Make sure the initial skymodel is present
    if not parset_dict["input_skymodel"]:
        if parset_dict["download_initial_skymodel"]:
            log.info(
                "No input sky model file given and download requested. "
                "Will automatically download skymodel."
            )
            parset_dict.update(
                {
                    "input_skymodel": os.path.join(
                        parset_dict["dir_working"], "skymodels", "initial_skymodel.txt"
                    )
                }
            )
            if parset_dict["apparent_skymodel"]:
                log.info(
                    "Ignoring apparent_skymodel, "
                    "because skymodel download has been requested."
                )
                parset_dict["apparent_skymodel"] = None
        else:
            log.error(
                "No input sky model file given and no download requested. Exiting..."
            )
            raise RuntimeError(
                "No input sky model file given and no download requested."
            )
    elif (parset_dict["input_skymodel"]) and parset_dict["download_initial_skymodel"]:
        if not parset_dict["download_overwrite_skymodel"]:
            # If download is requested, ignore the given skymodel.
            log.info(
                "Skymodel download requested, but user-provided skymodel is present. "
                "Disabling download and using skymodel provided by the user."
            )
            parset_dict["download_initial_skymodel"] = False
        else:
            log.info(
                "User-provided skymodel is present, but download_overwrite_skymodel "
                "is True. Overwriting user-supplied skymodel with downloaded one."
            )
            parset_dict["download_initial_skymodel"] = True
    elif not os.path.exists(parset_dict["input_skymodel"]):
        raise FileNotFoundError(
            'Input sky model file "{}" not found.'.format(parset_dict["input_skymodel"])
        )

    return parset_dict

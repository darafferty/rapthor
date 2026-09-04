import configparser
import re
from pathlib import Path

REGEX_LOGGED_CLI_PARAMETER = re.compile(
    r"""
    \s*                     # leading whitespace
    (?P<outer_quote>['"])?  # key-value pair may start with quote
    (?: # key group
        (?P<key>[\w.]+)     # key name
        =                   # key name is followed by equal sign
    )?                      # key name is optional eg. positional parameters
    (?: # value group
        (?: # quoted value group
            (?P<value_quote>['"])  # value starts with quote
            (?P<value_quoted>
                (?: # only match if the next character is not the closing quote
                    (?!(?P=value_quote)).
                )+ # one or more characters that are not the closing quote
            )
            (?P=value_quote)  # match the exact same quote character again
        )
        |
        (?P<value> # unquoted value group
            (?:(?!\\\n).)*? # match zero or more characters up to the line continuation
        )
    )
    (?(outer_quote)         # if we matched an outer quote earlier
        (?P=outer_quote)    # match the exact same closing quote character again
    |                       # if no quote, match nothing here
    )
    (?:
        (?:
            \s              # key-value pairs are separated by a space
            (?:\\\n)?         # and possibly a line continuation
        )
        |$                  # or the end of the line
    )
    """,
    # flags: allow comments and whitespace in pattern, dot matches newline
    re.VERBOSE | re.DOTALL,
)


def get_working_dir_from_parset(parset_path):
    """Return dir_working from a parset file."""
    parset = configparser.ConfigParser()
    parset.read(parset_path)
    return parset["global"]["dir_working"]


def find_step_logs(log_dir, step_name):
    """Return CWL job logs for a specific step name."""
    return sorted(Path(log_dir).glob(f"CWLJob_*pipeline_parset.cwl.*{step_name}_*.log"))


def update_parset_path(parset_path, param_dict):
    """Helper function to update parset parameters and return a new path."""
    parset = configparser.ConfigParser()
    parset.read(parset_path)
    missing_params = set(param_dict.keys())

    for section in parset.sections():
        for key, value in param_dict.items():
            if key in parset[section]:
                parset[section][key] = value
                missing_params.discard(key)

    updated_parset_path = parset_path.parent / "updated.parset"

    if missing_params:
        raise ValueError(f"Parameters {missing_params} not found in parset.")

    with updated_parset_path.open("w") as fp:
        parset.write(fp)
    return updated_parset_path


def get_wsclean_output_mtimes(image_pipeline_dir):
    """Return a mapping of WSClean output product filenames to their modification timestamps"""
    products = {}
    for pattern in [
        "*-MFS-image.fits",
        "*-MFS-image-pb.fits",
        "*-MFS-residual.fits",
        "*-MFS-dirty.fits",
    ]:
        for path in Path(image_pipeline_dir).glob(pattern):
            products[path.name] = path.stat().st_mtime_ns
    return products


def make_failing_filter_skymodel(fake_bin_dir):
    """Create a PATH-injected wrapper for filter_skymodel.py."""
    fake_script = fake_bin_dir / "filter_skymodel.py"
    fake_script.write_text("#!/usr/bin/env python3\nraise SystemExit(1)")
    fake_script.chmod(0o755)
    return fake_script


def parse_cmd_args_from_logs(log_path, cmd):
    """Parse the command arguments from a CWL job log file.

    Returns a dict of ``{key: value}`` strings for every ``key=value`` argument
    passed to the command.  Arguments without a value (e.g. ``msout=``) are stored with
    an empty string.

    Raises ``ValueError`` if no DP3 command is found in the log.
    """
    text = Path(log_path).read_text()
    # Extract everything after "$ DP3 " up to the first non-continuation line
    match = re.search(rf"{cmd}\s+((?:.*\\\n)*.*)", text)
    if not match:
        raise ValueError(f"No {cmd} command found in {log_path}")

    # Extract key-value pairs using the regular expression
    param_string = match[1].removeprefix("\\\n")
    if "=" in param_string:
        return {
            key: val
            for match in REGEX_LOGGED_CLI_PARAMETER.finditer(param_string)
            if (key := match["key"]) and (val := match["value"] or match["value_quoted"] or "")
        }

    # Avoid parsing and return the full string
    return param_string.replace("\\\n", " ")


def parse_dp3_args_from_log(log_path):
    """Parse the DP3 command arguments from a CWL job log file.

    Returns a dict of ``{key: value}`` strings for every ``key=value`` argument
    passed to DP3.  Arguments without a value (e.g. ``msout=``) are stored with
    an empty string.

    Raises ``ValueError`` if no DP3 command is found in the log.
    """
    return parse_cmd_args_from_logs(log_path, "DP3")

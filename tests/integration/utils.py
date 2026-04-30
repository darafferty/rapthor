import configparser
import re
from pathlib import Path


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


def parse_dp3_args_from_log(log_path):
    """Parse the DP3 command arguments from a CWL job log file.

    Returns a dict of ``{key: value}`` strings for every ``key=value`` argument
    passed to DP3.  Arguments without a value (e.g. ``msout=``) are stored with
    an empty string.

    Raises ``ValueError`` if no DP3 command is found in the log.
    """
    text = Path(log_path).read_text()

    # Extract everything after "$ DP3 " up to the first non-continuation line
    match = re.search(r"\$ DP3\s+((?:.*\\\n)*.*)", text)
    if not match:
        raise ValueError(f"No DP3 command found in {log_path}")

    # Join continuation lines, drop backslashes, split into tokens
    tokens = match.group(1).replace("\\\n", " ").split()

    return {k: v for k, _, v in (t.partition("=") for t in tokens if "=" in t)}

import configparser
from pathlib import Path

def get_working_dir_from_parset(parset_path):
    """Return dir_working from a parset file."""
    parset = configparser.ConfigParser()
    parset.read(parset_path)
    return parset["global"]["dir_working"]


def find_step_logs(log_dir, step_name):
    """Return CWL job logs for a specific step name."""
    return sorted(Path(log_dir).glob(f"CWLJob_subpipeline_parset.cwl.{step_name}_*.log"))


def set_parset_option(parset_path, section, key, value):
    """Set or add a single option in a specific parset section."""
    parset = configparser.ConfigParser()
    parset.read(parset_path)
    parset[section][key] = value
    with parset_path.open("w") as fp:
        parset.write(fp)
    return parset_path


#!/usr/bin/env python3
#
# Script to convert a Rapthor parset to a JSON file.
# Currently, we do not use JSON files for specifying Rapthor settings,
# but that might change in the future.
#
# Usage: parset2json.py [<parset-file>] [<json-file>]
#
# If <json-file> is not given, use the basename of <parset-file>, and
# replacing its extension with `.json`.
# If <parset-file> is not given, read the default Rapthor parset file:
# `rapthor/settings/defaults.parset`

import configparser
import json
import os
import sys
from rapthor.lib.parset import Parset


def main(src, dest):
    """
    Convert Rapthor parset file to a JSON file

    Parameters
    ----------
    src : str
        Name of the input parset file
    dest : str
        Name of the output JSON file
    """
    parser = configparser.ConfigParser(interpolation=None)
    if not parser.read(src):
        raise RuntimeError(f"Failed to read parset '{src}'")
    settings = Parset.config_as_dict(parser)
    with open(dest, "w") as f:
        json.dump(settings, f, indent=4)


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else Parset.DEFAULT_PARSET
    dest = (
        sys.argv[2]
        if len(sys.argv) > 2
        else os.path.splitext(os.path.basename(src))[0] + ".json"
    )
    main(src, dest)

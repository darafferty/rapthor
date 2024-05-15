#!/usr/bin/env python3

import configparser
import json
import os
import sys
from rapthor.lib.parset import Parset, lexical_cast


def main(src, dest):
    parser = configparser.ConfigParser(interpolation=None)
    if not parser.read(src):
        raise RuntimeError(f"Failed to read parset '{src}'")
    settings = dict()
    for section in parser.sections():
        settings[section] = dict(
            (key, lexical_cast(value)) for key, value in parser.items(section)
        )
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

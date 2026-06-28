"""
Tests for the make_region_file script and region helper.
"""

import subprocess
import sys
from pathlib import Path

from rapthor.execution.regions import make_ds9_region_from_skymodel

RESOURCE_DIR = Path(__file__).resolve().parents[1] / "resources"


def test_make_region_file_cli_matches_function(tmp_path):
    """The CLI wrapper and helper produce the same DS9 region file."""
    skymodel = RESOURCE_DIR / "test_true_sky.txt"
    function_region = tmp_path / "function.reg"
    cli_region = tmp_path / "cli.reg"

    make_ds9_region_from_skymodel(
        str(skymodel),
        258.0,
        57.5,
        3.0,
        2.0,
        str(function_region),
        enclose_names=False,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "rapthor.scripts.make_region_file",
            str(skymodel),
            "258.0",
            "57.5",
            "3.0",
            "2.0",
            str(cli_region),
            "--enclose_names=False",
        ],
        check=True,
    )

    region_text = function_region.read_text()
    assert cli_region.read_text() == region_text
    assert region_text.startswith("# Region file format: DS9 version 4.0")
    assert "polygon(" in region_text
    assert "point(" in region_text
    assert "Patch_patch_1" in region_text

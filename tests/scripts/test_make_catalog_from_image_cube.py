"""
Tests for the make_catalog_from_image_cube script.
"""

import os
import runpy
import sys

import pytest

import rapthor.execution.image.cubes as cube_module
from rapthor.execution.image.cubes import make_catalog_from_image_cube


class FakeBdsfImage:
    def __init__(self):
        self.catalog_calls = []

    def write_catalog(self, **kwargs):
        self.catalog_calls.append(kwargs)
        with open(kwargs["outfile"], "w") as catalog_file:
            catalog_file.write("catalog")


def test_main_parses_cube_metadata_and_writes_catalog(tmp_path, monkeypatch):
    cube_image = tmp_path / "cube.fits"
    cube_beams = tmp_path / "cube_beams.txt"
    cube_frequencies = tmp_path / "cube_frequencies.txt"
    output_catalog = tmp_path / "catalog.fits"
    cube_image.write_text("cube")
    cube_beams.write_text("(0.03, 0.015, 35.0), (0.02, 0.01, 45.0)")
    cube_frequencies.write_text("140000000.0, 150000000.0")
    fake_image = FakeBdsfImage()
    process_calls = []

    def fake_process_image(*args, **kwargs):
        process_calls.append((args, kwargs))
        return fake_image

    monkeypatch.setattr(cube_module.bdsf, "process_image", fake_process_image)

    make_catalog_from_image_cube(
        str(cube_image),
        str(cube_beams),
        str(cube_frequencies),
        str(output_catalog),
        threshisl=4.0,
        threshpix=6.0,
        rmsbox="(60, 20)",
        rmsbox_bright="(30, 10)",
        adaptive_thresh=50.0,
        ncores=3,
    )

    assert output_catalog.read_text() == "catalog"
    assert os.environ["TMPDIR"]
    assert process_calls == [
        (
            (str(cube_image),),
            {
                "mean_map": "zero",
                "rms_box": (60, 20),
                "thresh_pix": 6.0,
                "thresh_isl": 4.0,
                "thresh": "hard",
                "adaptive_rms_box": True,
                "adaptive_thresh": 50.0,
                "rms_box_bright": (30, 10),
                "atrous_do": False,
                "rms_map": True,
                "quiet": True,
                "spectralindex_do": True,
                "beam_spectrum": ((0.03, 0.015, 35.0), (0.02, 0.01, 45.0)),
                "frequency_sp": (140000000.0, 150000000.0),
                "ncores": 3,
                "outdir": ".",
            },
        )
    ]
    assert fake_image.catalog_calls == [
        {
            "outfile": str(output_catalog),
            "format": "fits",
            "catalog_type": "srl",
            "incl_chan": True,
            "clobber": True,
        }
    ]


def test_main_rejects_empty_beam_file(tmp_path):
    cube_image = tmp_path / "cube.fits"
    cube_beams = tmp_path / "cube_beams.txt"
    cube_frequencies = tmp_path / "cube_frequencies.txt"
    output_catalog = tmp_path / "catalog.fits"
    cube_image.write_text("cube")
    cube_beams.write_text("")
    cube_frequencies.write_text("140000000.0")

    with pytest.raises(RuntimeError, match="No beam parameters found"):
        make_catalog_from_image_cube(
            str(cube_image), str(cube_beams), str(cube_frequencies), str(output_catalog)
        )


def test_main_rejects_empty_frequency_file(tmp_path):
    cube_image = tmp_path / "cube.fits"
    cube_beams = tmp_path / "cube_beams.txt"
    cube_frequencies = tmp_path / "cube_frequencies.txt"
    output_catalog = tmp_path / "catalog.fits"
    cube_image.write_text("cube")
    cube_beams.write_text("(0.03, 0.015, 35.0)")
    cube_frequencies.write_text("")

    with pytest.raises(RuntimeError, match="No frequencies found"):
        make_catalog_from_image_cube(
            str(cube_image), str(cube_beams), str(cube_frequencies), str(output_catalog)
        )


def test_make_catalog_from_image_cube_cli_matches_function(tmp_path, monkeypatch):
    cube_image = tmp_path / "cube.fits"
    cube_beams = tmp_path / "cube_beams.txt"
    cube_frequencies = tmp_path / "cube_frequencies.txt"
    function_catalog = tmp_path / "function_catalog.fits"
    cli_catalog = tmp_path / "cli_catalog.fits"
    cube_image.write_text("cube")
    cube_beams.write_text("(0.03, 0.015, 35.0), (0.02, 0.01, 45.0)")
    cube_frequencies.write_text("140000000.0, 150000000.0")
    process_calls = []

    def fake_process_image(*args, **kwargs):
        process_calls.append((args, kwargs))
        return FakeBdsfImage()

    monkeypatch.setattr(cube_module.bdsf, "process_image", fake_process_image)

    make_catalog_from_image_cube(
        str(cube_image),
        str(cube_beams),
        str(cube_frequencies),
        str(function_catalog),
        threshisl=4.0,
        threshpix=6.0,
        rmsbox=(60, 20),
        rmsbox_bright=(30, 10),
        adaptive_thresh=50.0,
        ncores=3,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "make_catalog_from_image_cube.py",
            str(cube_image),
            str(cube_beams),
            str(cube_frequencies),
            str(cli_catalog),
            "--threshisl=4.0",
            "--threshpix=6.0",
            "--rmsbox=(60, 20)",
            "--rmsbox_bright=(30, 10)",
            "--adaptive_thresh=50.0",
            "--ncores=3",
        ],
    )

    runpy.run_module("rapthor.scripts.make_catalog_from_image_cube", run_name="__main__")

    assert cli_catalog.read_text() == function_catalog.read_text() == "catalog"
    assert len(process_calls) == 2
    assert process_calls[1][1] == process_calls[0][1]

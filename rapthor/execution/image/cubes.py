"""Image-cube helpers used by image execution scripts."""

import ast
import os
import tempfile
from typing import Optional, Sequence, Tuple, Union

import bdsf

from rapthor.lib.fitsimage import FITSCube

RmsBox = Union[str, Tuple[float, float], Tuple[int, int]]

DEFAULT_CUBE_CATALOG_THRESHISL = 3.0
DEFAULT_CUBE_CATALOG_THRESHPIX = 5.0
DEFAULT_CUBE_CATALOG_RMSBOX: RmsBox = (150, 50)
DEFAULT_CUBE_CATALOG_RMSBOX_BRIGHT: RmsBox = (35, 7)
DEFAULT_CUBE_CATALOG_ADAPTIVE_THRESH = 75.0
DEFAULT_CUBE_CATALOG_NCORES = 8


def make_image_cube(
    input_image_filenames: Sequence[str],
    output_image_filename: str,
    output_beams_filename: Optional[str] = None,
    output_frequencies_filename: Optional[str] = None,
) -> None:
    """
    Build a frequency-ordered FITS image cube and write beam/frequency metadata.

    When metadata filenames are omitted, they are derived from
    ``output_image_filename`` by appending ``_beams.txt`` and
    ``_frequencies.txt``.
    """
    image = FITSCube(list(input_image_filenames))
    image.write(output_image_filename)

    if output_beams_filename is None:
        output_beams_filename = f"{output_image_filename}_beams.txt"
    image.write_beams(output_beams_filename)

    if output_frequencies_filename is None:
        output_frequencies_filename = f"{output_image_filename}_frequencies.txt"
    image.write_frequencies(output_frequencies_filename)


def make_catalog_from_image_cube(
    cube_image: str,
    cube_beams: str,
    cube_frequencies: str,
    output_catalog: str,
    threshisl: float = DEFAULT_CUBE_CATALOG_THRESHISL,
    threshpix: float = DEFAULT_CUBE_CATALOG_THRESHPIX,
    rmsbox: Optional[RmsBox] = DEFAULT_CUBE_CATALOG_RMSBOX,
    rmsbox_bright: RmsBox = DEFAULT_CUBE_CATALOG_RMSBOX_BRIGHT,
    adaptive_thresh: float = DEFAULT_CUBE_CATALOG_ADAPTIVE_THRESH,
    ncores: int = DEFAULT_CUBE_CATALOG_NCORES,
) -> None:
    """
    Detect sources in an image cube and write a FITS source catalog.

    The beam and frequency metadata are read from the sidecar text files written
    by ``make_image_cube`` and passed to PyBDSF as spectral metadata.
    """
    rmsbox = _parse_optional_tuple(rmsbox)
    rmsbox_bright = _parse_tuple(rmsbox_bright)

    # Keep PyBDSF multiprocessing socket paths short in deep work directories.
    os.environ["TMPDIR"] = tempfile.gettempdir()

    beams = _read_literal_metadata(cube_beams, "beam parameters")
    frequencies = _read_literal_metadata(cube_frequencies, "frequencies")

    image = bdsf.process_image(
        cube_image,
        mean_map="zero",
        rms_box=rmsbox,
        thresh_pix=threshpix,
        thresh_isl=threshisl,
        thresh="hard",
        adaptive_rms_box=True,
        adaptive_thresh=adaptive_thresh,
        rms_box_bright=rmsbox_bright,
        atrous_do=False,
        rms_map=True,
        quiet=True,
        spectralindex_do=True,
        beam_spectrum=beams,
        frequency_sp=frequencies,
        ncores=ncores,
        outdir=".",
    )
    image.write_catalog(
        outfile=output_catalog,
        format="fits",
        catalog_type="srl",
        incl_chan=True,
        clobber=True,
    )


def _parse_optional_tuple(value: Optional[RmsBox]):
    """Parse a tuple-like option while preserving ``None``."""
    if value is None:
        return None
    return _parse_tuple(value)


def _parse_tuple(value: RmsBox):
    """Parse tuple options accepted from CLI strings or Python callers."""
    if isinstance(value, str):
        return ast.literal_eval(value)
    return value


def _read_literal_metadata(filename: str, label: str):
    """Read a one-line literal metadata file and reject empty files clearly."""
    with open(filename) as metadata_file:
        lines = metadata_file.readlines()
    if not lines:
        raise RuntimeError(f"No {label} found in {filename}")
    return ast.literal_eval(lines[0])

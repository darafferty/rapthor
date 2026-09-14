import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from rapthor.execution.image.astrometry import (
    astrometry_corrected_image_path,
    correct_astrometry_image,
    make_astrometry_corrected_image_record,
    validate_astrometry_corrections,
)


@pytest.mark.parametrize("suffix", [".fits", ".fits.fz", "", ".fz"])
def test_astrometry_corrected_image_path_adds_astrometry_infix(suffix):
    root = "/work/sector_1-MFS-I-image-pb"
    expected = f"{root}-ast.fits" + (".fz" if suffix.endswith(".fz") else "")
    assert astrometry_corrected_image_path(root + suffix).as_posix() == expected


def test_correct_astrometry_image_copies_input_when_no_offsets_are_available(tmp_path):
    input_image = tmp_path / "sector_1-MFS-I-image-pb.fits"
    input_image.write_text("pb image")

    output_image = correct_astrometry_image(input_image, region_file=None, corrections_file=None)

    assert output_image == tmp_path / "sector_1-MFS-I-image-pb-ast.fits"
    assert output_image.read_text() == "pb image"


@pytest.mark.parametrize("offsets", [{}, {"facet_name": []}])
def test_correct_astrometry_image_treats_empty_offsets_as_no_correction(tmp_path, offsets):
    input_image = tmp_path / "sector_1-MFS-I-image-pb.fits"
    offsets_file = tmp_path / "sector_1.astrometry_offsets.json"
    input_image.write_text("pb image")
    offsets_file.write_text(json.dumps(offsets))

    output_image = correct_astrometry_image(
        input_image, region_file=None, corrections_file=offsets_file
    )

    assert output_image.read_text() == "pb image"


def test_validate_astrometry_corrections_rejects_missing_keys():
    corrections = {
        "facet_name": ["field"],
        "meanRAOffsetDeg": [0.1],
        "meanDecOffsetDeg": [0.1],
        "stdRAOffsetDeg": [0.01],
    }

    with pytest.raises(ValueError, match="stdDecOffsetDeg"):
        validate_astrometry_corrections(corrections)


def test_validate_astrometry_corrections_rejects_mismatched_lengths():
    corrections = {
        "facet_name": ["field", "other"],
        "meanRAOffsetDeg": [0.1],
        "meanDecOffsetDeg": [0.1],
        "stdRAOffsetDeg": [0.01],
        "stdDecOffsetDeg": [0.01],
    }

    with pytest.raises(ValueError, match="equal length"):
        validate_astrometry_corrections(corrections)


def test_validate_astrometry_corrections_accepts_diagnostics_schema():
    corrections = {
        "facet_name": ["field"],
        "meanRAOffsetDeg": [0.1],
        "meanDecOffsetDeg": [0.2],
        "stdRAOffsetDeg": [0.01],
        "stdDecOffsetDeg": [0.02],
        "meanClippedRAOffsetDeg": [0.1],
        "stdClippedRAOffsetDeg": [0.01],
        "meanClippedDecOffsetDeg": [0.2],
        "stdClippedDecOffsetDeg": [0.02],
    }

    assert validate_astrometry_corrections(json.loads(json.dumps(corrections))) == corrections


@pytest.fixture
def astrometry_inputs(tmp_path):
    """Write small FITS images, DS9 facets, and diagnostic offset files."""

    def write_inputs(*, ndim=2, compressed=False, facets=("field",), offsets=None):
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ["RA---SIN", "DEC--SIN"]
        wcs.wcs.crval = [10.0, 20.0]
        wcs.wcs.crpix = [11.0, 11.0]
        # Unequal scales catch swapped axes as well as incorrect shift signs.
        wcs.wcs.cdelt = [-0.01, 0.02]
        header = wcs.to_header()
        header["RESTFREQ"] = 150e6
        header["BUNIT"] = "Jy/beam"
        data = np.zeros((21, 21))
        data[10, 10] = 1.0
        data = data.reshape((1,) * (ndim - 2) + data.shape)
        input_image = tmp_path / ("image.fits.fz" if compressed else "image.fits")
        if compressed:
            # Lossless compression keeps pixel assertions independent of quantization.
            fits.CompImageHDU(
                data=data, header=header, compression_type="GZIP_1", quantize_level=0
            ).writeto(input_image)
        else:
            fits.writeto(input_image, data, header)

        # Fractional pixel boundaries avoid round-off ambiguity at mask edges.
        vertices = wcs.all_pix2world(
            [[4.25, 4.25], [15.25, 4.25], [15.25, 15.25], [4.25, 15.25]], 0
        )
        coordinates = ",".join(str(value) for value in vertices.ravel())
        region_file = tmp_path / "facets.reg"
        region_file.write_text(
            "fk5\n"
            + "".join(
                f"polygon({coordinates}) # text = {{{name}}}\npoint(10.0,20.0)\n" for name in facets
            )
        )
        if offsets is None:
            # The expected pixel shift is +1 in x (RA), -2 in y (Dec).
            offsets = [("field", 0.01, 0.04, 0.001, 0.002)]
        corrections = dict(
            zip(
                [
                    "facet_name",
                    "meanRAOffsetDeg",
                    "meanDecOffsetDeg",
                    "stdRAOffsetDeg",
                    "stdDecOffsetDeg",
                ],
                map(list, zip(*offsets)),
            )
        )
        corrections_file = tmp_path / "offsets.json"
        corrections_file.write_text(json.dumps(corrections))
        return input_image, region_file, corrections_file

    return write_inputs


@pytest.mark.parametrize("ndim", [2, 4])
def test_correct_astrometry_image_shifts_pixels_and_preserves_header(astrometry_inputs, ndim):
    input_image, region_file, corrections_file = astrometry_inputs(
        ndim=ndim,
        # Only the second diagnostics entry matches the region's facet name.
        offsets=[("other", 0, 0, 0.001, 0.002), ("field", 0.01, 0.04, 0.001, 0.002)],
    )
    original = input_image.read_bytes()

    output_image = correct_astrometry_image(input_image, region_file, corrections_file)

    expected = np.zeros((21, 21))
    expected[8, 11] = 1.0
    data, header = fits.getdata(output_image, header=True)
    assert data.shape == (1,) * (ndim - 2) + (21, 21)
    np.testing.assert_allclose(data.squeeze(), expected, atol=1e-12)
    for key in ["CTYPE1", "CTYPE2", "CRPIX1", "CRPIX2", "CDELT1", "CDELT2", "BUNIT", "RESTFREQ"]:
        assert header[key] == fits.getheader(input_image)[key]
    assert input_image.read_bytes() == original


@pytest.mark.parametrize("uncertainty", [1.0, 2.0])
def test_correct_astrometry_image_skips_offsets_at_or_below_error(
    astrometry_inputs, caplog, uncertainty
):
    inputs = astrometry_inputs(
        offsets=[("field", 0.01, 0.04, 0.01 * uncertainty, 0.04 * uncertainty)]
    )

    with caplog.at_level("INFO", logger="rapthor:image:astrometry"):
        output_image = correct_astrometry_image(*inputs)

    np.testing.assert_array_equal(fits.getdata(output_image), fits.getdata(inputs[0]))
    assert "Skipping correction for facet field" in caplog.text


def test_correct_astrometry_image_leaves_unmatched_facet_unshifted(astrometry_inputs, caplog):
    inputs = astrometry_inputs(facets=("unmatched",))

    output_image = correct_astrometry_image(*inputs)

    np.testing.assert_array_equal(fits.getdata(output_image), fits.getdata(inputs[0]))
    assert "Astrometry offsets for facet unmatched were not found" in caplog.text


def test_correct_astrometry_image_averages_overlapping_facets(astrometry_inputs):
    inputs = astrometry_inputs(
        facets=("shifted", "stationary"),
        # Deliberately reverse the diagnostics order relative to the region file.
        offsets=[("stationary", 0, 0, 0.001, 0.002), ("shifted", 0.01, 0.04, 0.001, 0.002)],
    )
    with fits.open(inputs[0], mode="update") as hdus:
        hdus[0].data[0, 0] = 5.0  # Outside both facets: must be masked out.

    output_image = correct_astrometry_image(*inputs)

    expected = np.zeros((21, 21))
    expected[10, 10] = 0.5
    expected[8, 11] = 0.5
    np.testing.assert_allclose(fits.getdata(output_image), expected, atol=1e-12)


def test_correct_astrometry_image_without_regions_warns_and_copies(astrometry_inputs, caplog):
    input_image, _, corrections_file = astrometry_inputs()

    output_image = correct_astrometry_image(input_image, None, corrections_file)

    assert output_image.read_bytes() == input_image.read_bytes()
    assert "no facet region file was provided" in caplog.text


@pytest.mark.parametrize("overwrite", [False, True])
def test_correct_astrometry_image_copy_honors_overwrite(tmp_path, overwrite):
    input_image = tmp_path / "input.fits"
    input_image.write_bytes(b"new image")
    output_image = tmp_path / "output.fits"
    output_image.write_bytes(b"existing image")

    result = correct_astrometry_image(
        input_image, None, None, output_image=output_image, overwrite=overwrite
    )

    assert result == output_image
    assert result.read_bytes() == (b"new image" if overwrite else b"existing image")


@pytest.mark.parametrize("output_name", ["copy.fits", "copy.fits.fz"])
def test_correct_astrometry_image_copy_preserves_compression_and_creates_parent(
    tmp_path, output_name
):
    input_image = tmp_path / "input.fits.fz"
    input_image.write_bytes(b"compressed image")

    result = correct_astrometry_image(
        input_image, None, None, output_image=tmp_path / "products" / output_name
    )

    assert result == tmp_path / "products" / "copy.fits.fz"
    assert result.read_bytes() == input_image.read_bytes()


def test_correct_astrometry_image_correction_honors_overwrite(astrometry_inputs, tmp_path):
    inputs = astrometry_inputs()
    output_image = tmp_path / "corrected.fits"
    output_image.write_bytes(b"existing image")

    with pytest.raises(OSError, match="already exists"):
        correct_astrometry_image(*inputs, output_image=output_image, overwrite=False)
    assert output_image.read_bytes() == b"existing image"

    assert correct_astrometry_image(*inputs, output_image=output_image) == output_image
    assert fits.getdata(output_image)[8, 11] == pytest.approx(1.0)


@pytest.mark.parametrize("output_name", ["corrected.fits", "corrected.fits.fz"])
def test_correct_astrometry_image_compresses_corrected_output(
    astrometry_inputs, tmp_path, monkeypatch, output_name
):
    inputs = astrometry_inputs(compressed=True)
    commands = []

    def fpack(command, *, check):
        commands.append((command, check))
        uncompressed = Path(command[1])
        data, header = fits.getdata(uncompressed, header=True)
        assert data[8, 11] == pytest.approx(1.0)
        fits.CompImageHDU(
            data=data, header=header, compression_type="GZIP_1", quantize_level=0
        ).writeto(f"{uncompressed}.fz")

    monkeypatch.setattr("rapthor.execution.image.astrometry.subprocess.run", fpack)

    output_image = correct_astrometry_image(*inputs, output_image=tmp_path / output_name)

    assert output_image == tmp_path / "corrected.fits.fz"
    assert commands == [(["fpack", str(tmp_path / "corrected.fits")], True)]
    assert not (tmp_path / "corrected.fits").exists()
    assert fits.getdata(output_image)[8, 11] == pytest.approx(1.0)


def test_correct_astrometry_image_keeps_intermediate_when_compression_fails(
    astrometry_inputs, monkeypatch, capsys
):
    inputs = astrometry_inputs(compressed=True)
    error = subprocess.CalledProcessError(1, ["fpack"])

    def fail_fpack(*args, **kwargs):
        raise error

    monkeypatch.setattr("rapthor.execution.image.astrometry.subprocess.run", fail_fpack)

    with pytest.raises(subprocess.CalledProcessError) as exc:
        correct_astrometry_image(*inputs)

    assert exc.value is error
    assert str(error) in capsys.readouterr().err
    output_image = astrometry_corrected_image_path(inputs[0])
    assert not output_image.exists()
    assert fits.getdata(output_image.with_suffix(""))[8, 11] == pytest.approx(1.0)


@pytest.mark.parametrize("with_offsets", [False, True])
def test_make_astrometry_corrected_image_record(astrometry_inputs, with_offsets):
    input_image, region_file, corrections_file = astrometry_inputs()

    record = make_astrometry_corrected_image_record(
        {"class": "File", "path": str(input_image)},
        {"class": "File", "path": str(region_file)} if with_offsets else None,
        {"class": "File", "path": str(corrections_file)} if with_offsets else None,
    )

    assert record == {"class": "File", "path": str(astrometry_corrected_image_path(input_image))}
    peak = (8, 11) if with_offsets else (10, 10)
    assert fits.getdata(record["path"])[peak] == pytest.approx(1.0)

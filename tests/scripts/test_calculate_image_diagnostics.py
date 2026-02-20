"""
Test suite for rapthor.scripts.calculate_image_diagnostics.
"""

from rapthor.lib import fitsimage
from rapthor.scripts.calculate_image_diagnostics import (
    check_astrometry,
    check_photometry,
)
import pytest
from astropy.table import Table
import astropy.units as u
import lsmtool.skymodel


def test_check_astrometry_zero_sources(
    observation,
    input_catalog_fits,
    image_fits,
    facet_region_ds9,
    sky_model_path,
    tmp_path,
    monkeypatch,
    capsys,
):
    """Test the check_astrometry function."""
    # Mock Table.read for FITS files to return a catalog with length zero
    monkeypatch.setattr("astropy.table.Table.read", lambda *args, **kwargs: [])

    expected_result = {}
    actual_result = check_astrometry(
        observation,
        input_catalog_fits,
        image_fits,
        facet_region_ds9,
        min_number=1,
        output_root=tmp_path / "astrometry_check",
        comparison_skymodel=sky_model_path,
    )
    captured_print = capsys.readouterr()
    assert "No sources found" in captured_print.out
    assert "Skipping the astrometry check..." in captured_print.out
    assert actual_result == expected_result


@pytest.mark.parametrize(
    "min_number,num_sources",
    [
        (1, 0),  # Test with zero sources
        (2, 1),  # Test with one source
        (10, 9),  # Test with fewer sources than min_number
    ],
)
def test_check_astrometry_sources_below_minimum_number(
    observation,
    input_catalog_fits,
    image_fits,
    facet_region_ds9,
    sky_model_path,
    tmp_path,
    monkeypatch,
    capsys,
    min_number,
    num_sources,
):
    """Test the check_astrometry function."""
    # Create mock Table for FITS files to return a catalog
    DC_Maj_max = 10 / 3600  # Hardcoded value in check_astrometry()
    RA_DEC_max = 2 / 3600  # Hardcoded value in check_astrometry()
    # Ensure all sources are within the maximum allowed values
    mock_data = {
        "DC_Maj": [DC_Maj_max - 0.0001] * num_sources,
        "E_RA": [RA_DEC_max - 0.0001] * num_sources,
        "E_DEC": [RA_DEC_max - 0.0001] * num_sources,
    }
    monkeypatch.setattr(
        "astropy.table.Table.read", lambda *args, **kwargs: Table(mock_data)
    )
    expected_result = {}
    actual_result = check_astrometry(
        observation,
        input_catalog_fits,
        image_fits,
        facet_region_ds9,
        min_number=min_number,
        output_root=tmp_path / "astrometry_check",
        comparison_skymodel=sky_model_path,
    )
    captured_print = capsys.readouterr()
    if num_sources == 0:
        assert "No sources found" in captured_print.out
    else:
        assert f"Fewer than {min_number} sources found" in captured_print.out
    assert "Skipping the astrometry check..." in captured_print.out
    assert actual_result == expected_result


def test_check_photometry_zero_sources(
    observation, input_catalog_fits, sky_model_path, monkeypatch, capsys
):
    """Test the check_astrometry function."""
    # Mock Table.read for FITS files to return a catalog with length zero
    monkeypatch.setattr("astropy.table.Table.read", lambda *args, **kwargs: [])

    expected_result = {}
    actual_result = check_photometry(
        observation,
        input_catalog_fits,
        freq=150e6,  # Example frequency in Hz
        comparison_skymodel=sky_model_path,
        min_number=1,
    )
    captured_print = capsys.readouterr()
    assert "No sources found" in captured_print.out
    assert "Skipping the photometry check..." in captured_print.out
    assert actual_result == expected_result


@pytest.mark.parametrize(
    "min_number,num_sources",
    [
        (2, 1),  # Test with one source
        (10, 9),  # Test with fewer sources than min_number
    ],
)
def test_check_photometry_below_min_number_sources(
    observation,
    input_catalog_fits,
    sky_model_path,
    monkeypatch,
    capsys,
    min_number,
    num_sources,
):
    """Test the check_photometry function."""
    monkeypatch.setattr("astropy.table.Table.read", lambda *args, **kwargs: [])
    mock_data = {
        "DC_Maj": [10 / 3600] * num_sources,
        "RA": [2 / 3600 * u.degree] * num_sources,
        "DEC": [2 / 3600 * u.degree] * num_sources,
    }
    monkeypatch.setattr(
        "astropy.table.Table.read", lambda *args, **kwargs: Table(mock_data)
    )
    freq = 150e6  # Example frequency in Hz
    expected_result = {}
    actual_result = check_photometry(
        observation,
        input_catalog_fits,
        freq,
        comparison_skymodel=sky_model_path,
        min_number=min_number,
    )
    captured_print = capsys.readouterr()
    assert f"Fewer than {min_number} sources found" in captured_print.out
    assert "Skipping the photometry check..." in captured_print.out
    assert actual_result == expected_result


@pytest.mark.disable_socket
def test_check_photometry_with_comparison_skymodel_does_not_access_internet(
    observation, input_catalog_fits, sky_model_path, monkeypatch, mocker, capsys
):
    """Test the check_photometry function."""
    num_sources = 10  # Test with more sources than min_number
    mock_data = {
        "Source_id": [f"Source_{i}" for i in range(num_sources)],
        "DC_Maj": [10 / 3600 - 0.0001] * num_sources,
        "E_RA": [2 / 3600 - 0.0001] * num_sources,
        "E_DEC": [2 / 3600 - 0.0001] * num_sources,
        "RA": [2 / 3600 * u.degree] * num_sources,
        "DEC": [2 / 3600 * u.degree] * num_sources,
        "Total_flux": [1.0] * num_sources,
    }
    mock_compare_result = {
        "meanRatio": 1,
        "stdRatio": 1,
        "meanClippedRatio": 1,
        "stdClippedRatio": 1,
        "meanRAOffsetDeg": 1,
        "stdRAOffsetDeg": 1,
        "meanClippedRAOffsetDeg": 1,
        "stdClippedRAOffsetDeg": 1,
        "meanDecOffsetDeg": 1,
        "stdDecOffsetDeg": 1,
        "meanClippedDecOffsetDeg": 1,
        "stdClippedDecOffsetDeg": 1,
    }
    mocker.patch.object(
        lsmtool.skymodel.SkyModel, "compare", return_value=mock_compare_result
    )
    mocker.patch.object(lsmtool.skymodel.SkyModel, "group")
    mocker.patch(
        "rapthor.scripts.calculate_image_diagnostics.os.rename", autospec=True
    )
    mocker.patch(
        "rapthor.scripts.calculate_image_diagnostics.os.path.exists",
        return_value=False,
    )
    monkeypatch.setattr(
        "astropy.table.Table.read", lambda *args, **kwargs: Table(mock_data)
    )
    diagnostics_dict = check_photometry(
        observation,
        input_catalog_fits,
        freq=15,
        comparison_skymodel=sky_model_path,
        min_number=1,
    )
    captured_print = capsys.readouterr()
    assert "No sources found" not in captured_print.out
    assert "Skipping the photometry check..." not in captured_print.out
    assert diagnostics_dict != {}


@pytest.mark.disable_socket
def test_check_astrometry_with_comparison_skymodel_does_not_access_internet(
    observation,
    input_catalog_fits,
    image_fits,
    facet_region_ds9,
    sky_model_path,
    tmp_path,
    monkeypatch,
    mocker,
    capsys,
):
    """Test the check_astrometry function."""
    num_sources = 10  # Test with more sources than min_number
    mock_data = {
        "Source_id": [f"Source_{i}" for i in range(num_sources)],
        "DC_Maj": [10 / 3600 - 0.0001] * num_sources,
        "E_RA": [2 / 3600 - 0.0001] * num_sources,
        "E_DEC": [2 / 3600 - 0.0001] * num_sources,
        "RA": [2 / 3600 * u.degree] * num_sources,
        "DEC": [2 / 3600 * u.degree] * num_sources,
        "Isl_Total_flux": [1.0] * num_sources,
    }
    fitsimage.FITSImage = mocker.MagicMock()
    mock_image = fitsimage.FITSImage(image_fits)
    mock_image.freq.return_value = 150e6  # Mock frequency in Hz

    mocker.patch.object(lsmtool.skymodel.SkyModel, "group")
    mock_compare_result = {
        "meanRatio": 1,
        "stdRatio": 1,
        "meanClippedRatio": 1,
        "stdClippedRatio": 1,
        "meanRAOffsetDeg": 1,
        "stdRAOffsetDeg": 1,
        "meanClippedRAOffsetDeg": 1,
        "stdClippedRAOffsetDeg": 1,
        "meanDecOffsetDeg": 1,
        "stdDecOffsetDeg": 1,
        "meanClippedDecOffsetDeg": 1,
        "stdClippedDecOffsetDeg": 1,
    }
    mocker.patch.object(
        lsmtool.skymodel.SkyModel, "compare", return_value=mock_compare_result
    )
    mocker.patch(
        "rapthor.lib.facet.filter_skymodel",
        side_effect=lambda polygon, sm, wcs: sm,
    )
    monkeypatch.setattr(
        "astropy.table.Table.read", lambda *args, **kwargs: Table(mock_data)
    )

    diagnostics_dict = check_astrometry(
        observation,
        input_catalog_fits,
        mock_image,
        facet_region_ds9,
        min_number=1,
        output_root=str(tmp_path / "astrometry_check"),
        comparison_skymodel=sky_model_path,
    )
    captured_print = capsys.readouterr()
    assert "No sources found" not in captured_print.out
    assert "Skipping the astrometry check..." not in captured_print.out
    assert diagnostics_dict != {}
    assert "meanClippedRAOffsetDeg" in diagnostics_dict
    assert "meanClippedDecOffsetDeg" in diagnostics_dict
    assert "stdClippedRAOffsetDeg" in diagnostics_dict
    assert "stdClippedDecOffsetDeg" in diagnostics_dict

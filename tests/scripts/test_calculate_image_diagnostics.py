"""
Test suite for rapthor.scripts.calculate_image_diagnostics.
"""

import logging

import astropy.units as u
import lsmtool.skymodel
import pytest
from astropy.table import Table

from rapthor.lib import fitsimage
from rapthor.scripts.calculate_image_diagnostics import check_astrometry, check_photometry, parse_args

# ---------------------------------------------------------------------------- #


@pytest.fixture(scope="session", params=[0, 1, 9])
def mock_minimal_table(request):
    num_sources = request.param
    # Create mock Table for FITS files to return a catalog
    dec_maj_max = 10 / 3600  # Hardcoded value in check_astrometry()
    ra_dec_max = 2 / 3600  # Hardcoded value in check_astrometry()
    # Ensure all sources are within the maximum allowed values
    mock_data = {
        "DC_Maj": [dec_maj_max - 0.0001] * num_sources,
        "E_RA": [ra_dec_max - 0.0001] * num_sources,
        "E_DEC": [ra_dec_max - 0.0001] * num_sources,
        "RA": [2 / 3600 * u.degree] * num_sources,
        "DEC": [2 / 3600 * u.degree] * num_sources,
    }

    # Test with fewer sources than min_number
    return Table(mock_data)


@pytest.fixture(params=[10])
def mock_full_photometry_table(request):
    # Create a mock Table for photometry with the same data as mock_full_table
    mock_data = mock_full_table_data(num_sources=request.param, total_flux_keyword="Total_flux")
    return Table(mock_data)


@pytest.fixture(params=[10])
def mock_full_astrometry_table(request):
    # Create a mock Table for astrometry with the same data as mock_full_table
    mock_data = mock_full_table_data(num_sources=request.param, total_flux_keyword="Isl_Total_flux")
    return Table(mock_data)


def mock_full_table_data(num_sources, total_flux_keyword):
    # Test with more sources than min_number
    return {
        "Source_id": [f"Source_{i}" for i in range(num_sources)],
        "DC_Maj": [10 / 3600 - 0.0001] * num_sources,
        "E_RA": [2 / 3600 - 0.0001] * num_sources,
        "E_DEC": [2 / 3600 - 0.0001] * num_sources,
        "RA": [2 / 3600 * u.degree] * num_sources,
        "DEC": [2 / 3600 * u.degree] * num_sources,
        total_flux_keyword: [1.0] * num_sources,
    }


# ---------------------------------------------------------------------------- #
# Test: check_astrometry


def test_check_astrometry_zero_sources(
    observation,
    input_catalog_fits,
    image_fits,
    facet_region_ds9,
    sky_model_path,
    tmp_path,
    caplog,
    monkeypatch,
):
    """
    Test the check_astrometry function when the input skymodel contains zero
    sources.
    """

    monkeypatch.setattr("astropy.table.Table.read", lambda *args, **kwargs: [])

    with caplog.at_level(logging.INFO):
        actual_result = check_astrometry(
            observation,
            input_catalog_fits,
            image_fits,
            facet_region_ds9,
            min_number=1,
            output_root=tmp_path / "astrometry_check",
            comparison_skymodel=sky_model_path,
        )
        assert "No sources found" in caplog.text
        assert "Skipping the astrometry check..." in caplog.text

    assert actual_result == {}


def test_check_astrometry_sources_below_minimum_number(
    observation,
    input_catalog_fits,
    image_fits,
    facet_region_ds9,
    sky_model_path,
    tmp_path,
    caplog,
    mocker,
    monkeypatch,
    mock_minimal_table,
):
    """
    Test the check_astrometry function when the input skymodel contains fewer
    sources than the required minimum.
    """

    # Mock Table.read for FITS files to return a catalog with length zero
    monkeypatch.setattr("astropy.table.Table.read", lambda *args, **kwargs: mock_minimal_table)

    num_sources = len(mock_minimal_table)
    min_number = num_sources + 1

    fitsimage.FITSImage = mocker.MagicMock()
    mock_image = fitsimage.FITSImage(image_fits)
    mock_image.freq.return_value = 150e6  # Mock frequency in Hz

    with caplog.at_level(logging.INFO):
        actual_result = check_astrometry(
            observation,
            input_catalog_fits,
            mock_image,
            facet_region_ds9,
            min_number=min_number,
            output_root=tmp_path / "astrometry_check",
            comparison_skymodel=sky_model_path,
        )

        msg = "No sources found" if num_sources == 0 else f"Fewer than {min_number} sources found"
        assert msg in caplog.text
        assert "Skipping the astrometry check..." in caplog.text

    assert actual_result == {}


# ---------------------------------------------------------------------------- #
# Test: check_photometry


def test_check_photometry_zero_sources(
    observation, input_catalog_fits, sky_model_path, caplog, monkeypatch
):
    """
    Test the check_photometry function when the skymodel contains zero sources.
    """

    # Mock Table.read for FITS files to return a catalog with length zero
    monkeypatch.setattr("astropy.table.Table.read", lambda *args, **kwargs: [])

    with caplog.at_level(logging.INFO):
        actual_result = check_photometry(
            observation,
            input_catalog_fits,
            freq=1.5e8,  # Example frequency in Hz
            comparison_skymodel=sky_model_path,
            min_number=1,
        )
        assert "No sources found" in caplog.text
        assert "Skipping the photometry check..." in caplog.text

    assert actual_result == {}


def test_check_photometry_below_min_number_sources(
    observation,
    input_catalog_fits,
    sky_model_path,
    caplog,
    monkeypatch,
    mock_minimal_table,
):
    """
    Test the check_photometry function when the skymodel contains fewer sources
    than the required minimum.
    """

    # Mock Table.read for FITS files to return a catalog with length zero
    monkeypatch.setattr("astropy.table.Table.read", lambda *args, **kwargs: mock_minimal_table)
    num_sources = len(mock_minimal_table)
    min_number = num_sources + 1

    with caplog.at_level(logging.INFO):
        actual_result = check_photometry(
            observation,
            input_catalog_fits,
            freq=1.5e8,  # Example frequency in Hz
            comparison_skymodel=sky_model_path,
            min_number=min_number,
        )
        msg = "No sources found" if num_sources == 0 else f"Fewer than {min_number} sources found"
        assert msg in caplog.text
        assert "Skipping the photometry check..." in caplog.text

    assert actual_result == {}


@pytest.mark.disable_socket
def test_check_photometry_with_comparison_skymodel_does_not_access_internet(
    observation,
    input_catalog_fits,
    sky_model_path,
    mocker,
    caplog,
    monkeypatch,
    mock_full_photometry_table,
):
    """
    Test that the  check_photometry function does not access the internet
    when a comparison skymodel is provided.
    """

    # Mock Table.read for FITS files to return a catalog with length zero
    monkeypatch.setattr(
        "astropy.table.Table.read",
        lambda *args, **kwargs: mock_full_photometry_table,
    )

    mocker.patch(
        "rapthor.scripts.calculate_image_diagnostics.compare_photometry_survey",
        return_value={
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
        },
    )
    mocker.patch(
        "rapthor.scripts.calculate_image_diagnostics._rename_plots",
        autospec=True,
    )

    with caplog.at_level(logging.INFO):
        diagnostics_dict = check_photometry(
            observation,
            input_catalog_fits,
            freq=15,
            comparison_skymodel=sky_model_path,
            min_number=1,
        )
        assert "No sources found" not in caplog.text
        assert "Skipping the photometry check..." not in caplog.text

    assert diagnostics_dict != {}


@pytest.mark.disable_socket
def test_check_photometry_without_comparison_surveys_does_not_access_internet(
    observation,
    input_catalog_fits,
    sky_model_path,
    mocker,
    caplog,
    monkeypatch,
    mock_full_photometry_table,
):
    """
    Test that the  check_photometry function does not access the internet
    when comparison surveys are not provided.
    """

    # Mock Table.read for FITS files to return a catalog with length zero
    monkeypatch.setattr(
        "astropy.table.Table.read",
        lambda *args, **kwargs: mock_full_photometry_table,
    )

    with caplog.at_level(logging.INFO):
        diagnostics_dict = check_photometry(
            observation,
            input_catalog_fits,
            freq=15,
            comparison_skymodel=None,
            comparison_surveys=(),
            min_number=1,
        )
        assert "No sources found" not in caplog.text
        assert "The backup survey catalog" not in caplog.text
        assert (
            "A comparison sky model is not available and a list of comparison surveys was not supplied. Skipping photometry check..."
            in caplog.text
        )
    assert diagnostics_dict == {}


@pytest.mark.disable_socket
def test_check_astrometry_with_comparison_skymodel_does_not_access_internet(
    observation,
    input_catalog_fits,
    image_fits,
    facet_region_ds9,
    sky_model_path,
    tmp_path,
    mocker,
    caplog,
    monkeypatch,
    mock_full_astrometry_table,
):
    """
    Test that the  check_astrometry function does not access the internet
    when a comparison skymodel is provided.
    """
    # Mock Table.read for FITS files to return a catalog with length zero
    monkeypatch.setattr(
        "astropy.table.Table.read",
        lambda *args, **kwargs: mock_full_astrometry_table,
    )

    fitsimage.FITSImage = mocker.MagicMock()
    mock_image = fitsimage.FITSImage(image_fits)
    mock_image.freq.return_value = 150e6  # Mock frequency in Hz

    mocker.patch.object(lsmtool.skymodel.SkyModel, "group")
    mocker.patch.object(
        lsmtool.skymodel.SkyModel,
        "compare",
        return_value={
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
        },
    )
    mocker.patch(
        "rapthor.lib.facet.filter_skymodel",
        side_effect=lambda polygon, sm, wcs: sm,
    )

    with caplog.at_level(logging.INFO):
        diagnostics_dict = check_astrometry(
            observation,
            input_catalog_fits,
            mock_image,
            facet_region_ds9,
            min_number=1,
            output_root=str(tmp_path / "astrometry_check"),
            comparison_skymodel=sky_model_path,
        )
        assert "No sources found" not in caplog.text
        assert "Skipping the astrometry check..." not in caplog.text

    assert "meanClippedRAOffsetDeg" in diagnostics_dict
    assert "meanClippedDecOffsetDeg" in diagnostics_dict
    assert "stdClippedRAOffsetDeg" in diagnostics_dict
    assert "stdClippedDecOffsetDeg" in diagnostics_dict


@pytest.mark.disable_socket
def test_check_astrometry_with_no_internet_access_does_not_access_internet(
    observation,
    input_catalog_fits,
    image_fits,
    facet_region_ds9,
    tmp_path,
    mocker,
    caplog,
    monkeypatch,
    mock_full_astrometry_table,
):
    """
    Test that the  check_astrometry function does not access the internet
    when internet access is not permitted.
    """
    # Mock Table.read for FITS files to return a catalog with length zero
    monkeypatch.setattr(
        "astropy.table.Table.read",
        lambda *args, **kwargs: mock_full_astrometry_table,
    )

    fitsimage.FITSImage = mocker.MagicMock()
    mock_image = fitsimage.FITSImage(image_fits)
    mock_image.freq.return_value = 150e6  # Mock frequency in Hz

    mocker.patch.object(lsmtool.skymodel.SkyModel, "group")
    mocker.patch.object(
        lsmtool.skymodel.SkyModel,
        "compare",
        return_value={
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
        },
    )
    mocker.patch(
        "rapthor.lib.facet.filter_skymodel",
        side_effect=lambda polygon, sm, wcs: sm,
    )

    with caplog.at_level(logging.INFO):
        diagnostics_dict = check_astrometry(
            observation,
            input_catalog_fits,
            mock_image,
            facet_region_ds9,
            min_number=1,
            output_root=str(tmp_path / "astrometry_check"),
            comparison_skymodel=None,
            allow_internet_access=False,
        )
        assert "No sources found" not in caplog.text
        assert "internet access is not permitted" in caplog.text

    assert diagnostics_dict == {}


# ---------------------------------------------------------------------------- #
# Test: parse_args

@pytest.mark.parametrize("allow_internet_access", [True, False])
def test_calculate_image_diagnostics_parse_args(monkeypatch, allow_internet_access):
    """
    Test that parse_args() is called with the correct default arguments when the CLI is invoked without the --allow_internet_access flag.
    """
    monkeypatch.setattr(
        "sys.argv", ["calculate_image_diagnostics",
                     "flat_noise_image",
                     "flat_noise_rms_image", 
                     "true_sky_image",
                     "true_sky_rms_image",
                     "input_catalog",
                     "obs_ms",
                     "obs_starttime",
                     "obs_ntimes",
                     "diagnostics_file",
                     "output_root"]
                     + (["--allow_internet_access"] if allow_internet_access else []),
    )
    args = parse_args()
    assert args.flat_noise_image == "flat_noise_image"
    assert args.flat_noise_rms_image == "flat_noise_rms_image"
    assert args.true_sky_image == "true_sky_image"
    assert args.true_sky_rms_image == "true_sky_rms_image"
    assert args.input_catalog == "input_catalog"
    assert args.obs_ms == "obs_ms"
    assert args.obs_starttime == "obs_starttime"
    assert args.obs_ntimes == "obs_ntimes"
    assert args.diagnostics_file == "diagnostics_file"
    assert args.output_root == "output_root"
    
    assert args.photometry_comparison_skymodel is None
    assert args.photometry_comparison_surveys == ["TGSS", "LOTSS"]
    assert args.photometry_backup_survey == "NVSS"
    assert args.astrometry_comparison_skymodel is None
    assert args.min_number == 5

    assert args.allow_internet_access is allow_internet_access
    
    
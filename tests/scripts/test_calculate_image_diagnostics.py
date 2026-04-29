"""
Test suite for rapthor.scripts.calculate_image_diagnostics.
"""

import logging
from zipfile import Path

import astropy.units as u
import lsmtool
import lsmtool.skymodel
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.table import Table

from rapthor.lib import fitsimage
from rapthor.scripts.calculate_image_diagnostics import (
    _rename_plots,
    check_astrometry,
    check_photometry,
    compare_photometry_survey,
    filter_skymodel_for_photometry,
    fits_to_makesourcedb,
    parse_args,
)

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


@pytest.fixture
def mock_full_photometry_table(observation, mock_comparison_skymodel_table):
    # Use selected sources, but anchor coordinates to the observation center so
    # they always pass the FWHM beam cut in check_photometry().
    table = mock_comparison_skymodel_table.copy()
    num_sources = len(table)
    offsets_deg = np.linspace(-2e-4, 2e-4, num_sources)
    table["RA"] = (observation.ra + offsets_deg) * u.degree
    table["DEC"] = (observation.dec + offsets_deg) * u.degree
    table["DC_Maj"] = np.full(num_sources, 1 / 3600)
    return table


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


@pytest.fixture()
def mock_comparison_skymodel_table():
    # Subset copied from tests/resources/test_apparent_sky.txt
    selected_data = [
        ("s0c0", "-6:48:16.324", "56:56:59.178", 0.004379054102049962),
        ("s0c1", "-6:48:16.476", "56:56:59.175", 0.0013177344234482774),
        ("s0c2", "-6:48:16.784", "56:57:04.168", 0.00023802236965550248),
        ("s0c761", "-6:50:28.524", "57:26:08.192", 0.00015119098359826986),
        ("s0c762", "-6:50:28.371", "57:26:09.455", 0.0014859517024192579),
        ("s0c763", "-6:50:28.526", "57:26:09.442", 0.0027168816209913564),
        ("s0c764", "-6:49:11.444", "57:26:42.234", 0.00107158521926894),
        ("s0c765", "-6:49:11.6", "57:26:43.477", 0.0002788397060369219),
        ("s0c766", "-6:46:30.998", "57:26:56.315", 0.005508009846638286),
        ("s0c767", "-6:46:30.997", "57:26:57.565", 0.0009893880953724564),
    ]
    source_ids = [name for name, _, _, _ in selected_data]
    coords = SkyCoord(
        ra=[ra for _, ra, _, _ in selected_data],
        dec=[dec for _, _, dec, _ in selected_data],
        unit=(u.hourangle, u.deg),
    )
    flux_values = [flux for _, _, _, flux in selected_data]

    return Table(
        {
            "Source_id": source_ids,
            "DC_Maj": [10 / 3600 - 0.0001] * len(source_ids),
            "E_RA": [2 / 3600 - 0.0001] * len(source_ids),
            "E_DEC": [2 / 3600 - 0.0001] * len(source_ids),
            "RA": coords.ra.deg * u.degree,
            "DEC": coords.dec.deg * u.degree,
            "Total_flux": flux_values,
        }
    )


@pytest.fixture()
def grouped_comparison_skymodel(mock_comparison_skymodel_table):
    return fits_to_makesourcedb(mock_comparison_skymodel_table, 150e6, flux_colname="Total_flux")


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
    mocker,
):
    """
    Test the check_astrometry function when the input skymodel contains zero
    sources.
    """

    monkeypatch.setattr("astropy.table.Table.read", lambda *args, **kwargs: [])
    mocker.patch.object(lsmtool.skymodel.SkyModel, "group")

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
    mocker.patch.object(lsmtool.skymodel.SkyModel, "group")

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
        "lsmtool.facet.filter_skymodel",
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
        "lsmtool.facet.filter_skymodel",
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
        "sys.argv",
        [
            "calculate_image_diagnostics",
            "flat_noise_image",
            "flat_noise_rms_image",
            "true_sky_image",
            "true_sky_rms_image",
            "input_catalog",
            "obs_ms",
            "obs_starttime",
            "obs_ntimes",
            "diagnostics_file",
            "output_root",
        ]
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


# ---------------------------------------------------------------------------- #
# Test: fits_to_makesourcedb


def test_fits_to_makesourcedb(mock_full_astrometry_table):
    """
    Test that fits_to_makesourcedb correctly converts a PyBDSF catalog to a
    makesourcedb sky model with correct format and content.
    """
    reference_freq = 150e6
    skymodel = fits_to_makesourcedb(mock_full_astrometry_table, reference_freq)

    assert skymodel is not None
    assert len(skymodel) == len(mock_full_astrometry_table)
    assert skymodel.getColNames() == ["Name", "Type", "Ra", "Dec", "I", "ReferenceFrequency"]
    assert np.array_equal(skymodel.getColValues("Name"), mock_full_astrometry_table["Source_id"])
    assert np.array_equal(
        skymodel.getColValues("Type"), ["POINT"] * len(mock_full_astrometry_table)
    )
    assert np.array_equal(skymodel.getColValues("I"), mock_full_astrometry_table["Isl_Total_flux"])
    assert np.array_equal(
        skymodel.getColValues("ReferenceFrequency"),
        [reference_freq] * len(mock_full_astrometry_table),
    )
    assert np.allclose(skymodel.getColValues("Ra"), mock_full_astrometry_table["RA"])
    assert np.allclose(skymodel.getColValues("Dec"), mock_full_astrometry_table["DEC"])


def test_fits_to_makesourcedb_single_source():
    """
    Test that fits_to_makesourcedb correctly handles a catalog with a single
    source.
    """

    single_source = Table(
        {
            "Source_id": ["TestSource"],
            "RA": [2.5],
            "DEC": [45.0],
            "Total_flux": [1.5],
        }
    )

    reference_freq = 150e6
    skymodel = fits_to_makesourcedb(single_source, reference_freq, flux_colname="Total_flux")

    assert len(skymodel) == 1
    assert skymodel.getColValues("Name")[0] == "TestSource"
    assert np.isclose(skymodel.getColValues("Ra")[0], 2.5)
    assert np.isclose(skymodel.getColValues("Dec")[0], 45.0)


# ---------------------------------------------------------------------------- #
# Test: _rename_plots


def test_rename_plots(tmp_path):
    """
    Test that _rename_plots correctly renames plot files according to the specified output root.
    """
    # Temporarily create some dummy plot files to test the renaming
    for filename in [
        "flux_ratio_vs_distance.pdf",  # Should be renamed to flux_ratio_vs_distance_TEST.pdf
        "flux_ratio_vs_flux.pdf",  # Should be renamed to flux_ratio_vs_flux_TEST.pdf
        "flux_ratio_sky.pdf",  # Should be renamed to flux_ratio_sky_TEST.pdf
        "positional_offsets_sky.pdf",  # Should be removed by _rename_plots
    ]:
        (tmp_path / filename).touch()

    expected_files = [
        "flux_ratio_vs_distance_TEST.pdf",
        "flux_ratio_vs_flux_TEST.pdf",
        "flux_ratio_sky_TEST.pdf",
    ]
    _rename_plots("TEST", tmp_path)
    for expected_file in expected_files:
        assert (tmp_path / expected_file).exists(), f"{expected_file} was not found in {tmp_path}"

    assert not (tmp_path / "positional_offsets_sky.pdf").exists(), (
        "positional_offsets_sky.pdf should have been removed"
    )


def test_compare_skymodel(
    grouped_comparison_skymodel, mock_comparison_skymodel_table, monkeypatch, tmp_path
):
    """
    Test that compare_photometry_survey correctly generates diagnostic plots and returns expected statistics.
    """
    catalog = mock_comparison_skymodel_table
    survey = "TEST_SURVEY"
    comparison_skymodel = grouped_comparison_skymodel
    reference_skymodel = fits_to_makesourcedb(catalog, 150e6, flux_colname="Total_flux")
    monkeypatch.chdir(tmp_path)
    result = reference_skymodel.compare(
        comparison_skymodel,
        radius="5 arcsec",
        excludeMultiple=True,
        make_plots=True,
        name1="Input Catalog",
        name2=survey,
    )
    expected_keys = ["meanRatio", "stdRatio", "meanClippedRatio", "stdClippedRatio"]
    expected_plot_files = [
        "flux_ratio_vs_distance.pdf",
        "flux_ratio_vs_flux.pdf",
        "flux_ratio_sky.pdf",
        "positional_offsets_sky.pdf",
    ]
    assert all(key in result for key in expected_keys), (
        "Not all expected keys are present in the result"
    )
    assert all((tmp_path / plot).exists() for plot in expected_plot_files), (
        "Not all expected plot files were generated"
    )


def test_compare_photometry_survey(
    grouped_comparison_skymodel, mock_comparison_skymodel_table, monkeypatch, tmp_path
):
    """
    Test that compare_photometry_survey correctly compares photometry between the input catalog
    and the specified survey, and returns expected statistics.
    """
    survey = "TEST_SURVEY"
    monkeypatch.chdir(tmp_path)
    result = compare_photometry_survey(
        mock_comparison_skymodel_table,
        survey,
        grouped_comparison_skymodel,
        freq=150e6,
    )
    expected_keys = [
        "meanRatio_TEST_SURVEY",
        "stdRatio_TEST_SURVEY",
        "meanClippedRatio_TEST_SURVEY",
        "stdClippedRatio_TEST_SURVEY",
    ]
    expected_plot_files = [
        "flux_ratio_vs_distance_TEST_SURVEY.pdf",
        "flux_ratio_vs_flux_TEST_SURVEY.pdf",
        "flux_ratio_sky_TEST_SURVEY.pdf",
    ]
    assert all(key in result for key in expected_keys), (
        "Not all expected keys are present in the result"
    )
    assert all((tmp_path / plot).exists() for plot in expected_plot_files), (
        "Not all expected plot files were generated"
    )


def test_compare_photometry_survey_no_matches(
    grouped_comparison_skymodel, mock_comparison_skymodel_table, monkeypatch, tmp_path, caplog
):
    """
    Test that compare_photometry_survey correctly handles the case where there are no matches between the input catalog and the survey.
    """
    survey = "TEST_SURVEY"
    # Modify the comparison skymodel to have no sources within the matching radius
    monkeypatch.setattr(
        "lsmtool.skymodel.SkyModel.compare",
        lambda self, *args, **kwargs: None,
    )
    monkeypatch.chdir(tmp_path)
    with caplog.at_level(logging.INFO):
        result = compare_photometry_survey(
            mock_comparison_skymodel_table,
            survey,
            grouped_comparison_skymodel,
            freq=150e6,
        )
    assert result == {}, (
        "Expected an empty result when there are no matches between the input catalog and the survey"
    )
    assert (
        "The photometry check with the TEST_SURVEY catalog could not be done due to insufficient matches. Skipping this survey..."
        in caplog.text
    )


def test_check_photometry_expected_plots(
    observation,
    input_catalog_fits,
    mock_full_photometry_table,
    selected_sky_model_path,
    monkeypatch,
    tmp_path,
):
    """
    Test that check_photometry generates the expected diagnostic plots for a given survey.
    """
    survey = "TEST_SURVEY"
    original_table_read = Table.read

    def patched_table_read(*args, **kwargs):
        if args and str(args[0]) == str(input_catalog_fits):
            return mock_full_photometry_table
        return original_table_read(*args, **kwargs)

    monkeypatch.setattr(
        "rapthor.scripts.calculate_image_diagnostics.Table.read",
        patched_table_read,
    )
    comparison_skymodel = fits_to_makesourcedb(
        mock_full_photometry_table,
        150e6,
        flux_colname="Total_flux",
    )
    monkeypatch.setattr(
        "rapthor.scripts.calculate_image_diagnostics.load_photometry_surveys",
        lambda *args, **kwargs: {"USER_SUPPLIED": comparison_skymodel},
    )
    monkeypatch.chdir(tmp_path)
    result = check_photometry(
        obs=observation,
        input_catalog=input_catalog_fits,
        freq=150e6,
        min_number=1,
        comparison_skymodel=str(selected_sky_model_path),
        comparison_surveys=[survey],
    )
    expected_plot_files = [
        "flux_ratio_vs_distance_USER_SUPPLIED.pdf",
        "flux_ratio_vs_flux_USER_SUPPLIED.pdf",
        "flux_ratio_sky_USER_SUPPLIED.pdf",
    ]
    assert all((tmp_path / plot).exists() for plot in expected_plot_files), (
        "Not all expected plot files were generated"
    )
    assert result != {}, (
        "Expected non-empty result from check_photometry when there are matches between the input catalog and the survey"
    )


def test_filter_skymodel_for_photometry_keeps_expected_sources(
    mock_full_photometry_table, observation
):
    """
    Test that the filter_skymodel function correctly filters sources.
    """
    max_major_axis = 10 / 3600  # 10 arcseconds in degrees (default)
    freq = 150e6
    filtered_catalog = filter_skymodel_for_photometry(
        mock_full_photometry_table, observation, freq, max_major_axis
    )
    assert len(filtered_catalog) == len(mock_full_photometry_table), (
        "Expected all sources to pass the filtering criteria since they are all within the specified limits"
    )


def test_filter_skymodel_for_photometry_filters_out_sources_major_axis(
    mock_full_photometry_table, observation
):
    """
    Test that the filter_skymodel function correctly filters out sources that do not meet the criteria.
    """
    max_major_axis = (
        0.001 / 3600
    )  # 0.001 arcseconds in degrees, which is smaller than the DC_Maj of the sources
    freq = 150e6
    filtered_catalog = filter_skymodel_for_photometry(
        mock_full_photometry_table, observation, freq, max_major_axis=max_major_axis
    )
    assert len(filtered_catalog) == 0, (
        "Expected all sources to be filtered out since they do not meet the specified criteria"
    )

"""
Test cases for the normalize_flux_scale script in the rapthor package.
"""

import os
from pathlib import Path
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
import pytest
import rapthor.scripts.normalize_flux_scale
from rapthor.scripts.normalize_flux_scale import (
    create_normalization_h5parm,
    filter_sources,
    find_normalizations,
    fit_sed,
    get_field_phase_center,
    get_output_frequencies,
    read_source_catalog,
    main,
    _validate_source_catalog,
    _download_survey_data,
    _get_survey_coords,
    _get_survey_metadata,
    _cross_match_sources,
    _get_data_from_skymodel,
    _sort_metadata_by_frequency,
    _get_source_data
)


@pytest.fixture
def mock_survey_catalog(mocker, true_sky_model):
    """Mock lsmtool.load to return a valid non-empty survey sky model."""
    return mocker.patch(
        "rapthor.scripts.normalize_flux_scale.lsmtool.load",
        return_value=true_sky_model,
    )


@pytest.fixture
def mock_survey_catalog_with_no_sources(mocker):
    """Mock lsmtool.load to return an empty survey sky model."""
    return mocker.patch(
        "rapthor.scripts.normalize_flux_scale.lsmtool.load",
        return_value=[],
    )


@pytest.fixture
def phase_center():
    """Return the test phase center in degrees for the lsmtool query."""
    return [24.422081, 33.159759]


@pytest.fixture
def source_catalog(source_catalog_fits):
    """Return parsed source catalog data for the default test catalog."""
    with fits.open(source_catalog_fits) as hdul:
        return hdul[1].data


@pytest.fixture
def source_coords(source_catalog):
    """Return SkyCoord objects for the source catalog."""
    return SkyCoord(
        ra=np.array(source_catalog["RA"]) * u.degree, dec=np.array(source_catalog["DEC"]) * u.degree
    )


@pytest.fixture
def survey_data(true_sky_model, tmp_path):
    """Return parsed survey catalog data from the mock sky model."""
    survey_fits = tmp_path / "survey_catalog.fits"
    true_sky_model.write(survey_fits.as_posix(), format="fits", clobber=True)
    with fits.open(survey_fits) as hdul:
        return hdul[1].data


@pytest.fixture
def survey_coords(survey_data):
    """Return SkyCoord objects for the survey catalog."""
    return SkyCoord(
        ra=np.array(survey_data["RA"]) * u.degree, dec=np.array(survey_data["DEC"]) * u.degree
    )


@pytest.fixture
def source_catalog_with_outliers(source_catalog_with_outliers_fits):
    """Return parsed source catalog data for the outlier test catalog."""
    with fits.open(source_catalog_with_outliers_fits) as hdul:
        return hdul[1].data


def test_fit_sed():
    """
    Test the fit_sed function.
    """
    # Define test parameters
    frequencies = np.arange(120e6, 160e6, 5e6)  # Hz
    ref_flux = 1.0  # Jy
    ref_frequency = 140e6  # Hz
    alpha = -0.7
    fluxes = np.array(
        [ref_flux * (frequency / ref_frequency) ** alpha for frequency in frequencies]
    )
    errors = np.array([0.05] * len(fluxes))

    # Test single-flux case
    fit_fcn = fit_sed(fluxes[:1], errors[:1], frequencies[:1])
    assert fit_fcn(1.2e8) == pytest.approx(0.0)

    # Test two-fluxes case
    fit_fcn = fit_sed(fluxes[:2], errors[:2], frequencies[:2])
    assert fit_fcn(1.2e8) == pytest.approx(fluxes[0])

    # Test many-fluxes case
    fit_fcn = fit_sed(fluxes, errors, frequencies)
    assert fit_fcn(1.2e8) == pytest.approx(fluxes[0])


@pytest.mark.parametrize(
    "survey_fluxes, rapthor_fluxes, expected_normalizations",
    [
        (np.array([1.5, 2.5, 3.5]), np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 1.0])),
        (
            np.array([1.0, 2.0, 3.0]),
            np.array([2.0, 4.0, 6.0]),
            np.array([3.404309214649, 2.730960256794, 2.400637032291]),
        ),
    ],
)
def test_find_normalizations_different_frequencies(
    survey_fluxes, rapthor_fluxes, expected_normalizations
):
    """
    Test the find_normalizations function.
    """
    # Define test parameters
    rapthor_errors = np.array([0.1, 0.2, 0.3])
    rapthor_frequencies = np.array([100e6, 200e6, 300e6])  # Frequencies in Hz
    survey_errors = np.array([0.15, 0.25, 0.35])
    survey_frequencies = np.array([150e6, 250e6, 350e6])  # Frequencies in Hz
    output_frequencies = np.array([100e6, 200e6, 300e6])  # Frequencies in Hz
    normalizations = find_normalizations(
        rapthor_fluxes,
        rapthor_errors,
        rapthor_frequencies,
        survey_fluxes,
        survey_errors,
        survey_frequencies,
        output_frequencies,
    )
    assert np.allclose(normalizations, expected_normalizations), (
        f"Expected {expected_normalizations}, got {normalizations}"
    )


@pytest.mark.parametrize(
    "survey_fluxes, rapthor_fluxes, expected_normalizations",
    [
        (np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 6.0]), np.array([2.0, 2.0, 2.0])),
        (np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 1.0])),
        (np.array([2.0, 4.0, 6.0]), np.array([1.0, 2.0, 3.0]), np.array([0.5, 0.5, 0.5])),
    ],
)
def test_find_normalizations_same_frequencies_same_errors(
    survey_fluxes, rapthor_fluxes, expected_normalizations
):
    """
    Test the find_normalizations function.
    """
    rapthor_errors = survey_errors = np.array([0.1, 0.2, 0.3])
    rapthor_frequencies = survey_frequencies = output_frequencies = np.array([100e6, 200e6, 300e6])
    normalizations = find_normalizations(
        rapthor_fluxes,
        rapthor_errors,
        rapthor_frequencies,
        survey_fluxes,
        survey_errors,
        survey_frequencies,
        output_frequencies,
    )
    assert np.allclose(normalizations, expected_normalizations), (
        f"Expected {expected_normalizations}, got {normalizations}"
    )


def test_find_normalizations_zero_fluxes_returns_nan():
    """
    Test the find_normalizations function with zero fluxes.
    """
    rapthor_zero_fluxes = np.array([0.0, 0.0, 0.0])
    rapthor_errors = np.array([0.1, 0.2, 0.3])
    rapthor_frequencies = np.array([100e6, 200e6, 300e6])  # Frequencies in Hz
    survey_fluxes = np.array([1.0, 2.0, 3.0])
    survey_errors = np.array([0.15, 0.25, 0.35])
    survey_frequencies = np.array([150e6, 250e6, 350e6])  # Frequencies in Hz
    output_frequencies = np.array([100e6, 200e6, 300e6])  # Frequencies in Hz
    normalizations = find_normalizations(
        rapthor_zero_fluxes,
        rapthor_errors,
        rapthor_frequencies,
        survey_fluxes,
        survey_errors,
        survey_frequencies,
        output_frequencies,
    )
    assert np.isnan(normalizations).all(), f"Expected [nan, nan, nan], got {normalizations}"


@pytest.mark.parametrize(
    "survey_fluxes, rapthor_fluxes, expected_normalizations",
    [
        (np.array([1.0, 2.0]), np.array([2.0, 4.0]), np.array([2.0, 2.0])),
        (np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.array([1.0, 1.0])),
        (np.array([2.0, 4.0]), np.array([1.0, 2.0]), np.array([0.5, 0.5])),
    ],
)
def test_find_normalizations_two_fluxes_returns_power_law_result(
    survey_fluxes, rapthor_fluxes, expected_normalizations
):
    """
    Test the find_normalizations function with two fluxes.
    """
    rapthor_errors = survey_errors = np.array([0.1, 0.2])
    rapthor_frequencies = survey_frequencies = output_frequencies = np.array([100e6, 200e6])
    normalizations = find_normalizations(
        rapthor_fluxes,
        rapthor_errors,
        rapthor_frequencies,
        survey_fluxes,
        survey_errors,
        survey_frequencies,
        output_frequencies,
    )
    assert np.allclose(normalizations, expected_normalizations), (
        f"Expected {expected_normalizations}, got {normalizations}"
    )


def test_find_normalizations_single_flux_returns_nan():
    """
    Test the find_normalizations function with a single flux.
    """
    rapthor_fluxes = np.array([2.0])
    rapthor_errors = np.array([0.1])
    rapthor_frequencies = np.array([100e6])  # Frequencies in Hz
    survey_fluxes = np.array([1.0])
    survey_errors = np.array([0.15])
    survey_frequencies = np.array([150e6])  # Frequencies in Hz
    output_frequencies = np.array([100e6])  # Frequencies in Hz
    normalizations = find_normalizations(
        rapthor_fluxes,
        rapthor_errors,
        rapthor_frequencies,
        survey_fluxes,
        survey_errors,
        survey_frequencies,
        output_frequencies,
    )
    assert np.isnan(normalizations).all(), f"Expected [nan], got {normalizations}"


def test_create_normalization_h5parm(test_ms, tmp_path):
    """
    Test the create_normalization_h5parm function.
    """
    # Define test parameters
    antenna_file = str(Path(test_ms) / "ANTENNA")
    field_file = str(Path(test_ms) / "FIELD")
    h5parm_file = str(tmp_path / "test_h5parm.h5")
    frequencies = np.array([100e6, 200e6, 300e6])  # Frequencies in Hz
    normalizations = np.array([1.0, 2.0, 3.0])  # Normalization values for the test
    solset_name = "test_solset"
    soltab_name = "test_soltab"
    create_normalization_h5parm(
        antenna_file,
        field_file,
        h5parm_file,
        frequencies,
        normalizations,
        solset_name,
        soltab_name,
    )
    # Check if the file is created
    assert os.path.exists(h5parm_file), f"Expected {h5parm_file} to be created."


@pytest.mark.parametrize("use_input_skymodel", [False, True])
def test_main(
    test_ms,
    source_catalog_fits,
    tmp_path,
    use_input_skymodel,
    true_sky_path,
    apparent_sky_path,
    caplog,
    mocker,
    mock_survey_catalog,
):
    """
    Test the main function of the normalize_flux_scale script.
    """
    # Spy on match_coordinates_sky to ensure it is called
    match_coordinates_sky_spy = mocker.spy(
        rapthor.scripts.normalize_flux_scale, "match_coordinates_sky"
    )
    normalize_ra_dec_spy = mocker.spy(rapthor.scripts.normalize_flux_scale, "normalize_ra_dec")
    # Mock the fit_sed function to return a function that always returns 1.0
    mocker.patch("rapthor.scripts.normalize_flux_scale.fit_sed", return_value=lambda x: 1.0)

    output_h5parm = str(tmp_path / "test_output.h5parm")
    with caplog.at_level("INFO"):
        main(
            source_catalog_fits,
            test_ms,
            output_h5parm,
            radius_cut=3.0,  # in arcseconds
            major_axis_cut=30 / 3600,  # in degrees
            neighbor_cut=30 / 3600,  # in degrees
            spurious_match_cut=30 / 3600,  # in degrees
            min_sources=5,
            weight_by_flux_err=False,  # Whether to weight by flux error
            ignore_frequency_dependence=False,  # Whether to ignore frequency dependence,
            reference_skymodels=[str(true_sky_path), str(apparent_sky_path)]
            if use_input_skymodel
            else None,
            reference_skymodels_frequencies=[142000000.0, 142100000.0]
            if use_input_skymodel
            else None,
        )
    # Check if the file is created
    assert os.path.exists(output_h5parm), f"Expected {output_h5parm} to be created."
    assert "Number of sources in source catalog: 8" in caplog.text, (
        "Expected log message about number of sources before cuts."
    )
    assert "Number of sources in source catalog: 7" in caplog.text, (
        "Expected log message about number of sources after cuts."
    )
    if not use_input_skymodel:
        for survey in ["vlssr", "wenss"]:
            assert f"Downloading {survey} catalog for this field..." in caplog.text, (
                f"Expected log message about downloading {survey} catalog."
            )
    assert match_coordinates_sky_spy.call_count == 3, (
        "Expected match_coordinates_sky to be called three times "
        "(once for each survey and once for nearest neighbour distance). "
        f"Got {match_coordinates_sky_spy.call_count} calls."
    )
    assert normalize_ra_dec_spy.call_count > 0, (
        "Expected normalize_ra_dec to be called at least once."
    )


def test_main_raises_error_if_zero_channels_in_source_catalog(
    test_ms, source_catalog_zero_channels_fits, tmp_path
):
    """
    Test that the main function raises an error if there are zero channels in the source catalog.
    """
    with pytest.raises(
        ValueError, match="No channel frequency columns were found in the input source catalog."
    ):
        main(
            source_catalog_zero_channels_fits,
            test_ms,
            str(tmp_path / "test_output.h5parm"),
            radius_cut=3.0,
            major_axis_cut=30 / 3600,
            neighbor_cut=30 / 3600,
            spurious_match_cut=30 / 3600,
            min_sources=5,
            weight_by_flux_err=False,
            ignore_frequency_dependence=False,
        )


def test_main_skips_normalization_if_too_few_sources_before_cuts(
    test_ms, source_catalog_fits, tmp_path, caplog
):
    """
    Test that the main function skips normalization if there are too few sources.
    """
    with caplog.at_level("INFO"):
        main(
            source_catalog_fits,
            test_ms,
            str(tmp_path / "test_output.h5parm"),
            radius_cut=3.0,
            major_axis_cut=30 / 3600,
            neighbor_cut=30 / 3600,
            spurious_match_cut=30 / 3600,
            min_sources=10,  # Set min_sources higher than the number of sources in the catalog (8)
            weight_by_flux_err=False,
            ignore_frequency_dependence=False,
        )
    assert "Too few sources. Flux normalization will be skipped." in caplog.text, (
        "Expected log message about too few sources."
    )


def test_main_skips_normalization_if_too_few_sources_after_cuts(
    test_ms, source_catalog_fits, tmp_path, caplog
):
    """
    Test that the main function skips normalization if there are too few sources after applying cuts.
    """
    with caplog.at_level("INFO"):
        main(
            source_catalog_fits,
            test_ms,
            str(tmp_path / "test_output.h5parm"),
            radius_cut=3.0,
            major_axis_cut=30 / 3600,
            neighbor_cut=30 / 3600,
            spurious_match_cut=30 / 3600,
            min_sources=8,  # Set min_sources to 8, since one source is outside the radius cut, only 7 will remain after cuts
            weight_by_flux_err=False,
            ignore_frequency_dependence=False,
        )
    assert "Too few sources. Flux normalization will be skipped." in caplog.text, (
        "Expected log message about too few sources after cuts."
    )


def test_main_raises_error_if_download_fails(
    test_ms, source_catalog_fits, tmp_path, caplog, mocker
):
    """
    Test that the main function raises an error if the survey catalog download fails.
    """
    # Mock the lsmtool.load function to raise a ConnectionError
    mocker.patch(
        "rapthor.scripts.normalize_flux_scale.lsmtool.load",
        side_effect=ConnectionError("Mocked connection error"),
    )

    with caplog.at_level("ERROR"):
        main(
            source_catalog_fits,
            test_ms,
            str(tmp_path / "test_output.h5parm"),
            radius_cut=3.0,
            major_axis_cut=30 / 3600,
            neighbor_cut=30 / 3600,
            spurious_match_cut=30 / 3600,
            min_sources=5,
            weight_by_flux_err=False,
            ignore_frequency_dependence=False,
        )
    assert "A problem occurred when downloading the" in caplog.text, (
        "Expected log message about download problem."
    )


def test_main_skips_normalization_if_no_sources_in_survey_catalog(
    test_ms, source_catalog_fits, tmp_path, caplog, mocker
):
    """
    Test that the main function skips normalization if no sources are found in the survey catalog.
    """
    # Mock the lsmtool.load function to return an empty skymodel
    mocker.patch("rapthor.scripts.normalize_flux_scale.lsmtool.load", return_value=[])

    with caplog.at_level("WARNING"):
        main(
            source_catalog_fits,
            test_ms,
            str(tmp_path / "test_output.h5parm"),
            radius_cut=3.0,
            major_axis_cut=30 / 3600,
            neighbor_cut=30 / 3600,
            spurious_match_cut=30 / 3600,
            min_sources=5,
            weight_by_flux_err=False,
            ignore_frequency_dependence=False,
        )
    assert "No sources found in the" in caplog.text, (
        "Expected log message about no sources in survey catalog."
    )


@pytest.mark.usefixtures("mock_survey_catalog")
def test_main_logs_warning_when_too_few_sources_with_valid_fits(
    test_ms,
    source_catalog_fits,
    tmp_path,
    caplog,
    mocker,
):
    """
    Test that the main function logs a warning when too few sources have valid SED fits.
    """
    # Mock the fit_sed function to return a function that always returns NaN
    mocker.patch("rapthor.scripts.normalize_flux_scale.fit_sed", return_value=lambda x: np.nan)

    with caplog.at_level("WARNING"):
        main(
            source_catalog_fits,
            test_ms,
            str(tmp_path / "test_output.h5parm"),
            radius_cut=3.0,
            major_axis_cut=30 / 3600,
            neighbor_cut=30 / 3600,
            spurious_match_cut=30 / 3600,
            min_sources=5,
            weight_by_flux_err=False,
            ignore_frequency_dependence=False,
        )
    assert (
        "Too few sources with successful SED fits. Flux normalization will be skipped."
        in caplog.text
    ), "Expected log message about too few sources with valid SED fits."


@pytest.mark.parametrize("weight_by_flux_err", [False, True])
@pytest.mark.usefixtures("mock_survey_catalog")
def test_main_weights_by_flux_error(
    test_ms,
    source_catalog_fits,
    tmp_path,
    caplog,
    weight_by_flux_err,
    mocker,
):
    """
    Test that the main function correctly applies weighting by flux error when the flag is set.
    """
    # Mock the normalization calculations to ensure there are valid fits for the test
    mocker.patch("rapthor.scripts.normalize_flux_scale.find_normalizations", return_value=1.0)

    with caplog.at_level("INFO"):
        main(
            source_catalog_fits,
            test_ms,
            str(tmp_path / "test_output.h5parm"),
            radius_cut=3.0,
            major_axis_cut=30 / 3600,
            neighbor_cut=30 / 3600,
            spurious_match_cut=30 / 3600,
            min_sources=5,
            weight_by_flux_err=weight_by_flux_err,  # Enable weighting by flux error
            ignore_frequency_dependence=False,
        )

    # Check for log messages indicating that weighting by flux error was applied
    if weight_by_flux_err:
        assert (
            "Calculating weights given by the inverse of the errors on the source flux densities."
            in caplog.text
        ), "Expected log message about weighting by flux error."
    else:
        assert (
            "Weights will be set to 1 (i.e., no weighting by flux density errors)." in caplog.text
        ), "Expected log message about not weighting by flux error."


@pytest.mark.usefixtures("mock_survey_catalog")
def test_main_ignores_frequency_dependence(
    test_ms,
    source_catalog_fits,
    tmp_path,
    caplog,
    mocker,
):
    """
    Test that the main function ignores frequency dependence when the flag is set.
    """
    # Mock the normalization calculations to ensure there are valid fits for the test
    mocker.patch("rapthor.scripts.normalize_flux_scale.find_normalizations", return_value=1.0)

    with caplog.at_level("INFO"):
        main(
            source_catalog_fits,
            test_ms,
            str(tmp_path / "test_output.h5parm"),
            radius_cut=3.0,
            major_axis_cut=30 / 3600,
            neighbor_cut=30 / 3600,
            spurious_match_cut=30 / 3600,
            min_sources=5,
            weight_by_flux_err=False,
            ignore_frequency_dependence=True,  # Enable ignoring frequency dependence
        )
    assert (
        "Ignoring frequency dependence of normalizations. A single correction will be applied at all frequencies."
        in caplog.text
    ), "Expected log message about ignoring frequency dependence."


def test_read_source_catalog_raises_error_if_no_channel_columns(source_catalog_zero_channels_fits):
    """
    Test that the read_source_catalog function raises an error if there are no
    channel frequency columns in the source catalog.
    """
    with pytest.raises(
        ValueError, match="No channel frequency columns were found in the input source catalog."
    ):
        read_source_catalog(source_catalog_zero_channels_fits)


def test_read_source_catalog_returns_data_and_num_channels(source_catalog_fits):
    """
    Test that the read_source_catalog function returns the source catalog data and the number of channels.
    """
    source_catalog_data, num_channels = read_source_catalog(source_catalog_fits)
    assert source_catalog_data is not None, "Expected source catalog data to be returned."
    assert num_channels == 8, f"Expected number of channels to be 8, got {num_channels}."


def test_get_output_frequencies(test_ms):
    """
    Test that the get_output_frequencies function returns the correct frequencies from the measurement set.
    """
    expected_channel_width = 24414.0625  # Hz
    expected_min_freq = 134288024.90234375  # Hz
    expected_max_freq = 134458923.33984375  # Hz
    expected_frequencies = np.arange(
        expected_min_freq - expected_channel_width,
        expected_max_freq + expected_channel_width / 2,
        expected_channel_width,
    )  # Hz
    frequencies = get_output_frequencies(test_ms)
    assert np.allclose(frequencies, expected_frequencies), (
        f"Expected frequencies {expected_frequencies}, got {frequencies}"
    )


def test_get_field_phase_center(test_ms):
    """
    Test that the get_field_phase_center function returns the correct RA and Dec of the phase center from the FIELD table of the measurement set.
    """
    expected_ra = 0.4262457236387493  # radians
    expected_dec = 0.5787469737178225  # radians
    ra, dec = get_field_phase_center(test_ms)
    assert np.isclose(ra, expected_ra), f"Expected RA {expected_ra}, got {ra}"
    assert np.isclose(dec, expected_dec), f"Expected Dec {expected_dec}, got {dec}"


def test_filter_sources(source_catalog_with_outliers):
    """
    Test that the filter_sources function correctly filters sources based on the specified criteria.

    Fixture contains 10 sources, 5 of which are valid and 5 of which are outliers.
    """
    original_coords = np.array([(data["RA"], data["DEC"]) for data in source_catalog_with_outliers])
    original_source_catalog_data = source_catalog_with_outliers.copy()
    radius_cut = 3  # degrees
    major_axis_cut = 0.01  # degrees
    neighbor_cut = 0.01  # degrees
    phase_center_ra = np.deg2rad(24.422081)
    phase_center_dec = np.deg2rad(33.159759)
    expected_coords = np.array(
        [
            # Source 0 fails radius cut, should be removed
            (23.22208, 33.15976),  # Source 1: at phase center, should be kept
            # Sources 2-4: fail radius and major axis cuts, should be removed
            # Sources 5-9: within all cuts, should be kept
            (24.422081, 33.15976),
            (24.72208, 33.15976),
            (25.022081, 33.15976),
            (25.32208, 33.15976),
            (25.622082, 33.15976),
        ]
    )
    expected_data = (
        source_catalog_with_outliers[1:2].tolist() + source_catalog_with_outliers[5:10].tolist()
    )  # Sources 1 and 5-9 should be kept
    filtered_coords, filtered_data = filter_sources(
        source_catalog_with_outliers,
        phase_center_ra,
        phase_center_dec,
        radius_cut,
        major_axis_cut,
        neighbor_cut,
    )
    # Convert filtered coordinates to numpy array for easier comparison
    filtered_coords = np.array([(coord.ra.deg, coord.dec.deg) for coord in filtered_coords])
    assert np.allclose(filtered_coords, expected_coords), (
        f"Expected filtered coordinates {expected_coords}, got {filtered_coords}"
    )
    assert filtered_data.tolist() == expected_data, (
        f"Expected filtered data {expected_data}, got {filtered_data}"
    )

    # Check original data is unchanged
    assert np.array_equal(source_catalog_with_outliers, original_source_catalog_data), (
        "Expected original source catalog data to be unchanged."
    )


def test_validate_input_source_catalog_valid(source_catalog):
    """
    Test that source catalog with enough sources is considered valid.
    """
    assert _validate_source_catalog(source_catalog, min_sources=8)
    assert not _validate_source_catalog(source_catalog, min_sources=9)


@pytest.mark.parametrize("survey", ["vlssr", "wenss"])
def test_download_get_survey_data(survey, phase_center, mock_survey_catalog):
    """
    Test downloading a survey sky model and returning the data for normalization.
    """
    survey_data = _download_survey_data(
        survey, phase_center=[np.deg2rad(phase_center[0]), np.deg2rad(phase_center[1])]
    )

    radius = 5.0  # degrees hardcoded in rapthor
    mock_survey_catalog.assert_called_once()
    call_args, call_kwargs = mock_survey_catalog.call_args
    assert call_args == (survey,)
    assert np.allclose(call_kwargs["VOPosition"], phase_center)
    assert call_kwargs["VORadius"] == radius
    assert len(survey_data) > 0, f"Expected non-empty survey data, got {len(survey_data)}"


@pytest.mark.parametrize("survey", ["vlssr", "wenss"])
def test_download_get_survey_data_download_fails(survey, mocker):
    """
    Test that an error is raised when downloading a survey sky model fails.
    """
    # Mock the lsmtool.load function to raise a ConnectionError
    mocker.patch(
        "rapthor.scripts.normalize_flux_scale.lsmtool.load",
        side_effect=ConnectionError("Mocked connection error"),
    )
    survey_data = _download_survey_data(
        survey, phase_center=[np.deg2rad(24.422081), np.deg2rad(33.159759)]
    )
    assert survey_data is None


@pytest.mark.parametrize("survey", ["vlssr", "wenss"])
@pytest.mark.usefixtures("mock_survey_catalog_with_no_sources")
def test_download_get_survey_data_no_sources(survey):
    """
    Test that an error is raised when no sources are found in the downloaded survey sky model.
    """
    survey_data = _download_survey_data(
        survey, phase_center=[np.deg2rad(24.422081), np.deg2rad(33.159759)]
    )
    assert survey_data is None


def test_get_survey_coords(survey_data):
    """
    Test that the get_survey_coords function returns the correct coordinates from the survey data.
    """
    coords = _get_survey_coords(survey_data)
    assert len(coords) == len(survey_data)
    for coord, source in zip(coords, survey_data):
        assert np.isclose(coord.ra.deg, source["RA"])
        assert np.isclose(coord.dec.deg, source["DEC"])


def test_cross_match_sources(source_catalog, source_coords, survey_data, survey_coords):
    """
    Test that the _cross_match_sources function correctly cross-matches
    sources between the source catalog and the survey catalog.
    """
    survey_fluxes = _cross_match_sources(
        source_coords, survey_coords, survey_data, spurious_match_cut=30 / 3600
    )
    assert len(survey_fluxes) == len(source_catalog)


def test_get_survey_metadata_without_reference_skymodels(caplog):
    """
    Test that the _get_survey_metadata function correctly retrieves the survey metadata.
    """
    with caplog.at_level("INFO"):
        survey_metadata = _get_survey_metadata()
    assert "Using external survey catalogs for normalization" in caplog.text, (
        "Expected log message about using external survey catalogs."
    )
    expected_metadata = {
        "vlssr": {
            "flux_correction": 1,
            "flux_err": 0.1,
            "frequency": 74e6,
        },
        "wenss": {
            "flux_correction": 0.9,
            "flux_err": 3.6e-3,
            "frequency": 327e6,
        },
    }
    # Make sure metadata is sorted by frequency (low to high)
    assert list(survey_metadata.items()) == list(expected_metadata.items())


def test_get_survey_metadata_with_reference_skymodels(caplog):
    """
    Test that the _get_survey_metadata function correctly retrieves the survey metadata when reference skymodels are provided.
    """
    with caplog.at_level("INFO"):
        survey_metadata = _get_survey_metadata(
            reference_skymodels=["skymodel1", "skymodel2"],
            reference_frequencies=[142200000.0, 142100000.0],
        )
    expected_metadata = {
        "skymodel2": {
            "flux_correction": 1.0,
            "flux_err": 0.0,
            "frequency": 142100000.0,
        },
        "skymodel1": {
            "flux_correction": 1.0,
            "flux_err": 0.0,
            "frequency": 142200000.0,
        },
    }
    assert survey_metadata == expected_metadata


@pytest.mark.parametrize(
    "reference_frequencies",
    [
        [None, 142000000.0],  # One frequency is None
        None,  # All frequencies are None
    ],
)
def test_get_survey_metadata_with_reference_skymodels_and_missing_frequencies(
    true_sky_path, reference_frequencies
):
    """
    Test that the _get_survey_metadata function raises an error when reference skymodels are provided but frequencies are missing.
    """
    with pytest.raises(
        ValueError,
        match="Frequencies corresponding to the reference sky models must be provided if reference sky models are given.",
    ):
        _get_survey_metadata(
            reference_skymodels=[str(true_sky_path), str(true_sky_path)],
            reference_frequencies=reference_frequencies,
        )


def test_get_data_from_skymodel(true_sky_model):
    """Test data is extracted from LSMTool format skymodel."""
    data = _get_data_from_skymodel(true_sky_model)

    assert "Ra" in data.columns.names
    assert "Dec" in data.columns.names
    assert "I" in data.columns.names
    assert len(data) == len(true_sky_model)
    assert np.allclose(data["Ra"], true_sky_model.getColValues("Ra", units="degree"))
    assert np.allclose(data["Dec"], true_sky_model.getColValues("Dec", units="degree"))
    assert np.allclose(data["I"], true_sky_model.getColValues("I"))


def test_sort_metadata_by_frequency():
    """Test that the _sort_metadata_by_frequency function correctly sorts the survey metadata by frequency."""
    metadata = {
        "survey1": {"frequency": 327e6, "flux_correction": 0.9, "flux_err": 3.6e-3},
        "survey2": {"frequency": 74e6, "flux_correction": 1, "flux_err": 0.1},
        "survey3": {"frequency": 150e6, "flux_correction": 0.8, "flux_err": 0.05},
    }
    sorted_metadata = _sort_metadata_by_frequency(metadata)
    expected_order = ["survey2", "survey3", "survey1"]
    assert list(sorted_metadata.keys()) == expected_order, (
        f"Expected order {expected_order}, got {list(sorted_metadata.keys())}"
    )

def test_get_source_data(source_catalog):
    """Test that the correct fluxes, errors and frequencies are returned"""
    n_chan = 3  # Example number of channels
    i = 0  # Example source index
    rapthor_fluxes, rapthor_errors, rapthor_frequencies = _get_source_data(source_catalog, n_chan, i)
    assert rapthor_fluxes.shape[0] == n_chan
    assert rapthor_errors.shape[0] == n_chan
    assert rapthor_frequencies.shape[0] == n_chan
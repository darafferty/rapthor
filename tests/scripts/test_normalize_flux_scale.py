"""
Test cases for the normalize_flux_scale script in the rapthor package.
"""

import os
from pathlib import Path
import numpy as np
import pytest
import rapthor.scripts.normalize_flux_scale
from rapthor.scripts.normalize_flux_scale import (
    create_normalization_h5parm,
    find_normalizations,
    fit_sed,
    get_output_frequencies,
    read_source_catalog,
    main,
)


@pytest.fixture
def mock_survey_catalog_with_single_source(mocker, true_sky_model):
    """Mock lsmtool.load to return a valid non-empty survey sky model."""
    return mocker.patch(
        "rapthor.scripts.normalize_flux_scale.lsmtool.load",
        return_value=true_sky_model,
    )


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
    caplog,
    mocker,
    mock_survey_catalog_with_single_source,
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
            reference_skymodel=true_sky_path if use_input_skymodel else None,
        )
    # Check if the file is created
    assert os.path.exists(output_h5parm), f"Expected {output_h5parm} to be created."
    assert "Number of sources before applying cuts: 8" in caplog.text, (
        "Expected log message about number of sources before cuts."
    )
    assert "Number of sources after applying cuts: 7" in caplog.text, (
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
    assert (
        "Too few sources remain after applying cuts. Flux normalization will be skipped."
        in caplog.text
    ), "Expected log message about too few sources after cuts."


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


def test_main_logs_warning_when_too_few_sources_with_valid_fits(
    test_ms,
    source_catalog_fits,
    tmp_path,
    caplog,
    mocker,
    mock_survey_catalog_with_single_source,
):
    """
    Test that the main function logs a warning when too few sources have valid SED fits.
    """
    # Use fixture to mock survey download with a valid, minimal skymodel.
    _ = mock_survey_catalog_with_single_source
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
def test_main_weights_by_flux_error(
    test_ms,
    source_catalog_fits,
    tmp_path,
    caplog,
    weight_by_flux_err,
    mocker,
    mock_survey_catalog_with_single_source,
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


def test_main_ignores_frequency_dependence(
    test_ms,
    source_catalog_fits,
    tmp_path,
    caplog,
    mocker,
    mock_survey_catalog_with_single_source,
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

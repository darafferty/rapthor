"""
Test cases for the normalize_flux_scale script in the rapthor package.
"""

import os
from pathlib import Path
import numpy as np
import pytest
from rapthor.scripts.normalize_flux_scale import (
    create_normalization_h5parm,
    find_normalizations,
    fit_sed,
    main,
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


def test_main(test_ms, source_catalog_fits, tmp_path):
    """
    Test the main function of the normalize_flux_scale script.
    """
    # Define test parameters
    output_h5parm = str(tmp_path / "test_output.h5parm")
    radius_cut = 3.0  # in arcseconds
    major_axis_cut = 30 / 3600  # in degrees
    neighbor_cut = 30 / 3600  # in degrees
    spurious_match_cut = 30 / 3600  # in degrees
    min_sources = 5  # Minimum number of sources to consider
    weight_by_flux_err = False  # Whether to weight by flux error
    ignore_frequency_dependence = False  # Whether to ignore frequency dependence
    main(
        source_catalog_fits,
        test_ms,
        output_h5parm,
        radius_cut=radius_cut,
        major_axis_cut=major_axis_cut,
        neighbor_cut=neighbor_cut,
        spurious_match_cut=spurious_match_cut,
        min_sources=min_sources,
        weight_by_flux_err=weight_by_flux_err,
        ignore_frequency_dependence=ignore_frequency_dependence,
    )
    # Check if the file is created
    assert os.path.exists(output_h5parm), f"Expected {output_h5parm} to be created."
    pass

"""
Test cases for the normalize_flux_scale script in the rapthor package.
"""

import os

import numpy as np
import pytest
from rapthor.scripts.normalize_flux_scale import (create_normalization_h5parm,
                                                  find_normalizations, fit_sed,
                                                  main)


def test_fit_sed():
    """
    Test the fit_sed function.
    """
    # Define test parameters
    frequencies = np.arange(120e6, 160e6, 5e6)  # Hz
    ref_flux = 1.0  # Jy
    ref_frequency = 140e6  # Hz
    alpha = -0.7
    fluxes = np.array([ref_flux * (frequency / ref_frequency) ** alpha for frequency in frequencies])
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


def test_find_normalizations():
    """
    Test the find_normalizations function.
    """
    # # Define test parameters
    # rapthor_fluxes = [1.0, 2.0, 3.0]
    # rapthor_errors = [0.1, 0.2, 0.3]
    # rapthor_frequencies = [100e6, 200e6, 300e6]  # Frequencies in Hz
    # survey_fluxes = [1.5, 2.5, 3.5]
    # survey_errors = [0.15, 0.25, 0.35]
    # survey_frequencies = [150e6, 250e6, 350e6]  # Frequencies in Hz
    # output_frequencies = [100e6, 200e6, 300e6]  # Frequencies in Hz
    # expected_normalizations = [1.5, 2.5, 3.5]  # Expected normalization values

    # normalizations = find_normalizations(
    #     rapthor_fluxes,
    #     rapthor_errors,
    #     rapthor_frequencies,
    #     survey_fluxes,
    #     survey_errors,
    #     survey_frequencies,
    #     output_frequencies,
    # )

    # assert normalizations == expected_normalizations, (
    #     f"Expected {expected_normalizations}, got {normalizations}"
    # )
    pass


def test_create_normalization_h5parm(test_ms):
    """
    Test the create_normalization_h5parm function.
    """
    # # Define test parameters
    # antenna_file = test_ms / "antenna"
    # field_file = test_ms / "field"
    # h5parm_file = "test_h5parm.h5"
    # frequencies = [100e6, 200e6, 300e6]  # Frequencies in Hz
    # normalizations = [1.0, 2.0, 3.0]  # Normalization values for the test
    # solset_name = "test_solset"
    # soltab_name = "test_soltab"

    # create_normalization_h5parm(
    #     antenna_file,
    #     field_file,
    #     h5parm_file,
    #     frequencies,
    #     normalizations,
    #     solset_name,
    #     soltab_name,
    # )
    # # Check if the file is created
    # assert os.path.exists(h5parm_file), f"Expected {h5parm_file} to be created."
    pass


def test_main(test_ms):
    """
    Test the main function of the normalize_flux_scale script.
    """
    # # Define test parameters
    # source_catalog = "test_source_catalog.fits"
    # ms_file = test_ms
    # output_h5parm = "test_output.h5parm"
    # radius_cut = 3.0  # in arcseconds
    # major_axis_cut = 30 / 3600  # in degrees
    # neighbor_cut = 30 / 3600  # in degrees
    # spurious_match_cut = 30 / 3600  # in degrees
    # min_sources = 5  # Minimum number of sources to consider
    # weight_by_flux_err = False  # Whether to weight by flux error
    # ignore_frequency_dependence = False  # Whether to ignore frequency dependence
    # main(
    #     source_catalog,
    #     ms_file,
    #     output_h5parm,
    #     radius_cut=radius_cut,
    #     major_axis_cut=major_axis_cut,
    #     neighbor_cut=neighbor_cut,
    #     spurious_match_cut=spurious_match_cut,
    #     min_sources=min_sources,
    #     weight_by_flux_err=weight_by_flux_err,
    #     ignore_frequency_dependence=ignore_frequency_dependence,
    # )
    # # Check if the file is created
    # assert os.path.exists(output_h5parm), f"Expected {output_h5parm} to be created."
    pass

"""
Test suite for rapthor.scripts.calculate_image_diagnostics.
"""

from unittest.mock import Mock
from rapthor.scripts.calculate_image_diagnostics import check_astrometry
import pytest
from astropy.table import Table


def test_check_astrometry_zero_sources(observation, input_catalog_fits, image_fits, facet_region_ds9, sky_model_path, tmp_path, monkeypatch):
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
        comparison_skymodel=sky_model_path
    )
    assert actual_result == expected_result


@pytest.mark.parametrize("min_number,num_sources", [
    (1, 0),  # Test with zero sources
    (2, 1),  # Test with one source
    (10, 9),  # Test with fewer sources than min_number
])
def test_check_astrometry_sources_below_minimum_number(observation, input_catalog_fits, image_fits, facet_region_ds9, sky_model_path, tmp_path, monkeypatch, min_number, num_sources):
    """Test the check_astrometry function."""
    # Create mock Table for FITS files to return a catalog
    DC_Maj_max = 10/3600 # Hardcoded value in check_astrometry()
    RA_DEC_max = 2/3600 # Hardcoded value in check_astrometry()
    # Ensure all sources are within the maximum allowed values
    mock_data = {
        "DC_Maj": [DC_Maj_max-0.0001] * num_sources,
        "E_RA": [RA_DEC_max-0.0001] * num_sources,
        "E_DEC": [RA_DEC_max-0.0001] * num_sources,
    }    
    monkeypatch.setattr("astropy.table.Table.read", lambda *args, **kwargs: Table(mock_data))
    expected_result = {}
    actual_result = check_astrometry(
        observation,
        input_catalog_fits,
        image_fits,
        facet_region_ds9,
        min_number=min_number,
        output_root=tmp_path / "astrometry_check",
        comparison_skymodel=sky_model_path
    )
    assert actual_result == expected_result

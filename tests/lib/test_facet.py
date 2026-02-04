"""
Test cases for the rapthor.lib.facet module.
"""

from rapthor.lib import facet
import pytest

def test_read_ds9_region_file(request):
    """
    Test reading a DS9 region file.
    """
    facets = facet.read_ds9_region_file(request.config.resource_dir / "test.reg")
    assert len(facets) == 15
    assert facets[0].name == "Patch_1"
    assert facets[0].ra == 318.2026666666666
    assert facets[0].dec == 62.25055927777777
    assert facets[1].name == "Patch_10_with_spaces"
    assert facets[2].name == "Patch_11"
    assert facets[3].name == "Patch_12"


def test_write_ds9_region_file(request, tmp_path):
    """
    Test writing a DS9 region file.
    """
    reg_in = request.config.resource_dir / "test.reg"
    reg_out = tmp_path / "test_region_write.reg"
    facets = facet.read_ds9_region_file(reg_in)
    facet.make_ds9_region_file(facets, reg_out)
    facets = facet.read_ds9_region_file(reg_out)
    assert len(facets) == 15
    assert facets[0].name == "Patch_1"
    assert facets[0].ra == 318.2026666666666
    assert facets[0].dec == 62.25055927777777
    assert facets[1].name == "Patch_10_with_spaces"
    assert facets[2].name == "Patch_11"
    assert facets[3].name == "Patch_12"

def test_find_astrometry_offsets_with_comparison_skymodel_does_not_download_from_panstarrs(facet, mocker):
    """
    Test that find_astrometry_offsets does not attempt to download data from PanSTARRS
    if a comparison skymodel is provided.
    """
    facet.skymodel = mocker.MagicMock()
    mock_download_panstarrs = mocker.patch.object(facet, "download_panstarrs")
    mock_comparison_skymodel = "mock_LSMTool_skymodel"
    facet.find_astrometry_offsets(mock_comparison_skymodel, min_number=1)
    mock_download_panstarrs.assert_not_called()

@pytest.mark.parametrize("mock_panstarrs", ([], # No sources found in PanSTARRS
                                            [0, 1, 2, 3, 4], # Default minimum number of sources is 5
                                            [0, 1, 2, 3] # Below the minimum number of sources
                                            ))
def test_find_astrometry_offsets_without_comparison_skymodel_downloads_from_panstarrs(facet, mocker, mock_panstarrs):
    """
    Test that find_astrometry_offsets attempts to download data from PanSTARRS
    if no comparison skymodel is provided.
    """
    min_number = 5 # Default minimum number of sources is 5
    facet.skymodel = mocker.MagicMock()
    mock_download_panstarrs = mocker.patch.object(facet, "download_panstarrs", return_value=mock_panstarrs)
    facet.find_astrometry_offsets(comparison_skymodel=None, min_number=min_number)
    mock_download_panstarrs.assert_called_once()
    if len(mock_panstarrs) < min_number:
        facet.skymodel.compare.assert_not_called()
    else:
        facet.skymodel.compare.assert_called_once()
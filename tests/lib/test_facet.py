"""
Test cases for the rapthor.lib.facet module.
"""

from rapthor.lib import facet


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

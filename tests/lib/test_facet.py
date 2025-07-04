"""
Test cases for the rapthor.lib.facet module.
"""

import os
import pytest

import numpy as np
from rapthor.lib import facet


def test_make_facet_polygons():
    """
    Test the make_facet_polygons function, using a region that encompasses the NCP.
    """
    # Test a region that encompasses the NCP.
    facet_points, facet_polys = facet.make_facet_polygons(
        [119.73, 138.08, 124.13, 115.74],
        [89.92, 89.91, 89.89, 89.89],
        126.52,
        90.0,
        0.3,
        0.3,
    )
    facet_points = np.round(facet_points, 1).tolist()
    facet_polys = [np.round(a, 1).tolist() for a in facet_polys]

    # Check the facet points
    assert facet_points == [[119.7, 89.9], [138.1, 89.9], [124.1, 89.9], [115.7, 89.9]]

    # Check the facet polygons. Since the start point of each polygon can
    # be different from the control but the polygons still be identical,
    # we check only that each vertex is present in the control (and
    # vice versa)
    facet_polys_control = [
        [
            [127.3, 89.9],
            [278.9, 89.8],
            [351.5, 89.8],
            [51.9, 89.8],
            [119.9, 89.9],
            [127.3, 89.9],
        ],
        [
            [127.3, 89.9],
            [278.9, 89.8],
            [261.5, 89.8],
            [171.5, 89.8],
            [146.9, 89.8],
            [127.3, 89.9],
        ],
        [[127.3, 89.9], [146.9, 89.8], [119.9, 89.8], [119.9, 89.9], [127.3, 89.9]],
        [[119.9, 89.9], [51.9, 89.8], [81.5, 89.8], [119.9, 89.8], [119.9, 89.9]],
    ]
    facet_polys_flat = []
    for poly in facet_polys:
        for point in poly:
            facet_polys_flat.append(point)
    facet_polys_control_flat = []
    for poly in facet_polys_control:
        for point in poly:
            facet_polys_control_flat.append(point)
    all_present = True
    for point in facet_polys_flat:
        if point not in facet_polys_control_flat:
            all_present = False
    for point in facet_polys_control_flat:
        if point not in facet_polys_flat:
            all_present = False
    assert all_present, "polys: {0}, control: {1}".format(
        facet_polys_flat, facet_polys_control_flat
    )


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

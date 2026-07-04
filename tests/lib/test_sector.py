"""Tests for the `rapthor.lib.sector` module."""

from pathlib import Path

import numpy as np
import pytest
from matplotlib.patches import Polygon

import rapthor.lib.sector as sector_module


def test_sector_initializes_closed_polygons_and_observation_copies(sector, field):
    assert sector.name == "test_sector"
    assert sector.vertices_file.endswith("regions/test_sector_vertices.npy")
    assert sector.poly.exterior.is_ring
    assert sector.initial_poly.exterior.is_ring
    assert sector.poly_padded.area > sector.poly.area
    assert len(sector.observations) == len(field.observations)
    assert sector.observations[0] is not field.observations[0]


def test_set_prediction_parameters_forwards_sector_name_and_patches(sector, monkeypatch):
    calls = []
    sector.patches = ["[patch_0]", "[patch_1]"]

    for observation in sector.observations:
        monkeypatch.setattr(
            observation,
            "set_prediction_parameters",
            lambda name, patches, observation=observation: calls.append(
                (observation.name, name, patches)
            ),
        )

    sector.set_prediction_parameters()

    assert calls == [
        (observation.name, sector.name, sector.patches) for observation in sector.observations
    ]


def test_get_nwavelengths_scales_with_cell_size_and_time(sector):
    cellsize_deg = 1.0 / 3600.0
    timestep_sec = 10.0

    result = sector.get_nwavelengths(cellsize_deg, timestep_sec)

    max_baseline = 1 / (3 * cellsize_deg * np.pi / 180)
    expected = int(max_baseline * 2 * np.pi * timestep_sec / (24 * 60 * 60) / 4)
    assert result == expected


def test_filter_skymodel_delegates_to_lsmtool_facet_filter(sector, monkeypatch):
    calls = []
    skymodel = object()
    filtered_skymodel = object()

    def fake_filter_skymodel(poly, skymodel_arg, wcs, invert=False):
        calls.append((poly, skymodel_arg, wcs, invert))
        return filtered_skymodel

    monkeypatch.setattr(sector_module.facet, "filter_skymodel", fake_filter_skymodel)

    assert sector.filter_skymodel(skymodel, invert=True) is filtered_skymodel
    assert calls == [(sector.poly, skymodel, sector.field.wcs, True)]


def test_get_obs_parameters_returns_values_for_each_observation(sector):
    for index, observation in enumerate(sector.observations):
        observation.parameters["image_timestep"] = index + 1

    assert sector.get_obs_parameters("image_timestep") == [
        index + 1 for index in range(len(sector.observations))
    ]


def test_get_vertices_radec_returns_closed_sector_boundary(sector):
    ra_vertices, dec_vertices = sector.get_vertices_radec()

    assert len(ra_vertices) == len(dec_vertices)
    assert ra_vertices[0] == pytest.approx(ra_vertices[-1])
    assert dec_vertices[0] == pytest.approx(dec_vertices[-1])


def test_make_vertices_file_writes_radec_vertices(sector):
    sector.make_vertices_file()

    vertices = np.load(sector.vertices_file)
    assert vertices.shape == (5, 2)
    assert vertices[0, 0] == pytest.approx(vertices[-1, 0])
    assert vertices[0, 1] == pytest.approx(vertices[-1, 1])


def test_make_region_file_writes_ds9_and_casa_formats(sector, tmp_path):
    ds9_file = tmp_path / "sector.reg"
    casa_file = tmp_path / "sector.crtf"

    sector.make_region_file(ds9_file)
    sector.make_region_file(casa_file, region_format="casa")

    assert "polygon(" in ds9_file.read_text(encoding="utf-8")
    assert f"text={{{sector.name}}}" in ds9_file.read_text(encoding="utf-8")
    assert casa_file.read_text(encoding="utf-8").startswith("#CRTFv0")
    assert "poly[" in casa_file.read_text(encoding="utf-8")


def test_get_matplotlib_patch_returns_polygon_with_sector_label(sector):
    patch = sector.get_matplotlib_patch()

    assert isinstance(patch, Polygon)
    assert patch.get_label() == sector.name
    assert patch.get_xy().shape[1] == 2


def test_get_distance_to_obs_center_includes_center_and_vertices(sector):
    min_distance, max_distance = sector.get_distance_to_obs_center()

    assert min_distance == pytest.approx(0.0)
    assert max_distance > min_distance


def test_initialize_vertices_can_rebuild_polygon_after_size_change(sector):
    old_area = sector.poly.area
    sector.width_ra *= 2

    sector.intialize_vertices()

    assert sector.poly.area > old_area
    assert sector.poly_padded.area > sector.poly.area
    assert Path(sector.vertices_file).name == "test_sector_vertices.npy"

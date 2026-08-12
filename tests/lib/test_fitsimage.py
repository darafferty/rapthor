"""
Tests for the `rapthor.lib.fitsimage` module.
"""

import numpy as np
import pytest

from rapthor.lib.fitsimage import EmptyFacetSelectionError, FITSCube, FITSImage


class TestFITSCube:
    """
    Test cases for the FITSCube class.
    """

    def test_find_beam(self):
        pass

    def test_find_freq(self):
        pass

    def test_flatten(self):
        pass

    def test_write(self, filename=None):
        pass

    def test_get_beam(self):
        pass

    def test_get_wcs(self):
        pass

    def test_blank(self, vertices_file=None):
        pass

    def test_calc_noise(self, niter=1000, eps=None, sampling=4):
        pass

    def test_apply_shift(self, dra=0.0, ddec=0.0):
        pass

    def test_calc_weight(self):
        pass


def _to_facet_number(facet):
    return int(facet.name.split("_")[1])


class TestFITSImage:
    """
    Test cases for the FITSImage class.
    """

    def test_check_channel_images(self):
        pass

    def test_order_channel_images(self):
        pass

    def test_make_header(self):
        pass

    def test_make_data(self):
        pass

    def test_write(self, filename=None):
        pass

    def test_write_frequencies(self, filename=None):
        pass

    def test_write_beams(self, filename=None):
        pass

    def test_select_facet(self, facets, rendered_regions):
        image = FITSImage(rendered_regions)

        for facet in facets:
            selected_facet = image.select_facet(facet)
            facet_number = _to_facet_number(facet)
            assert facet_number in np.unique(selected_facet)
            difference_in_pixels = abs(
                np.count_nonzero(~np.isnan(selected_facet))
                - np.count_nonzero(image.img_data == facet_number)
            )
            assert difference_in_pixels < facet.polygon.length

    @pytest.mark.parametrize(
        "vertices",
        [
            pytest.param([(-5, 2), (-3, 2), (-3, 7), (-5, 7)], id="left"),
            pytest.param([(12, 2), (14, 2), (14, 7), (12, 7)], id="right"),
            pytest.param([(2, -5), (7, -5), (7, -3), (2, -3)], id="below"),
            pytest.param([(2, 12), (7, 12), (7, 14), (2, 14)], id="above"),
            pytest.param([(0.1, 0.1), (0.2, 0.1), (0.2, 0.2), (0.1, 0.2)], id="tiny"),
        ],
    )
    def test_select_facet_reports_empty_selection(
        self, identity_wcs_image, facet_factory, vertices
    ):
        facet = facet_factory("empty", vertices)

        with pytest.raises(EmptyFacetSelectionError, match="does not contain any image pixels"):
            identity_wcs_image.select_facet(facet)

    def test_select_facet_clips_facet_at_image_edge(self, identity_wcs_image, facet_factory):
        facet = facet_factory("partial", [(-2, 2), (3, 2), (3, 7), (-2, 7)])

        selected_facet = identity_wcs_image.select_facet(facet)

        assert selected_facet.shape == (6, 4)
        assert np.isfinite(selected_facet).any()

    def test_select_facet_supports_integer_image_data(self, identity_wcs_image, facet_factory):
        identity_wcs_image.img_data = identity_wcs_image.img_data.astype(int)
        facet = facet_factory("partial", [(-2, 2), (3, 2), (3, 7), (-2, 7)])

        selected_facet = identity_wcs_image.select_facet(facet)

        assert np.issubdtype(selected_facet.dtype, np.floating)
        assert np.isnan(selected_facet).any()

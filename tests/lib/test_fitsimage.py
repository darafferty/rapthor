"""
Tests for the `rapthor.lib.fitsimage` module.
"""

from rapthor.lib.fitsimage import FITSCube, FITSImage

import numpy as np


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
                np.count_nonzero(~np.isnan(selected_facet)) -
                np.count_nonzero(image.img_data == facet_number))
            assert difference_in_pixels < facet.polygon.length
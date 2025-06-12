"""
Tests for the `rapthor.lib.fitsimage` module.
"""

from rapthor.lib.fitsimage import FITSCube, FITSImage


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

"""
Test cases for the `rapthor.lib.sector` module.
"""

from rapthor.lib.sector import Sector


class TestSector:
    """
    Test cases for the `Sector` class in the `rapthor.lib.sector` module.
    """

    def test_set_prediction_parameters(self):
        pass

    def test_set_imaging_parameters(
        self,
        do_multiscale=False,
        recalculate_imsize=False,
        imaging_parameters=None,
        preapply_dde_solutions=False,
    ):
        pass

    def test_get_nwavelengths(self, cellsize_deg=None, timestep_sec=None):
        pass

    def test_make_skymodel(self, index=None):
        pass

    def test_filter_skymodel(self, skymodel=None):
        pass

    def test_get_obs_parameters(self, parameter=None):
        pass

    def test_intialize_vertices(self):
        pass

    def test_get_vertices_radec(self):
        pass

    def test_make_vertices_file(self):
        pass

    def test_make_region_file(self, outputfile=None, region_format="ds9"):
        pass

    def test_get_matplotlib_patch(self, wcs=None):
        pass

    def test_get_distance_to_obs_center(self):
        pass

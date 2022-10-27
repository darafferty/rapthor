import unittest
from rapthor.lib import facet
import numpy as np


class TestFacet(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    def test_make_facet_polygons(self):
        # Test a region that encompasses the NCP.
        facet_points, facet_polys = facet.make_facet_polygons([119.73, 138.08, 124.13, 115.74],
                                                              [89.92, 89.91, 89.89, 89.89],
                                                              126.52, 90.0, 0.3, 0.3)
        facet_points = np.round(facet_points, 2).tolist()
        facet_polys = [np.round(a, 2).tolist() for a in facet_polys]

        # Check the facet points
        self.assertEqual(facet_points, [[119.73, 89.92],
                                        [138.08, 89.91],
                                        [124.13, 89.89],
                                        [115.74, 89.89]])

        # Check the facet polygons. Since the start point of each polygon can
        # be different from the control but the polygons still be identical,
        # we check only that each vertex is present in the control (and
        # vice versa)
        facet_polys_control = [[[127.29, 89.91],
                                [278.87, 89.83],
                                [351.52, 89.79],
                                [51.92, 89.84],
                                [119.94, 89.9],
                                [127.29, 89.91]],
                               [[127.29, 89.91],
                                [278.87, 89.83],
                                [261.52, 89.79],
                                [171.52, 89.79],
                                [146.88, 89.84],
                                [127.29, 89.91]],
                               [[127.29, 89.91],
                                [146.88, 89.84],
                                [119.94, 89.85],
                                [119.94, 89.9],
                                [127.29, 89.91]],
                               [[119.94, 89.9],
                                [51.92, 89.84],
                                [81.52, 89.79],
                                [119.94, 89.85],
                                [119.94, 89.9]]]
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
        self.assertTrue(all_present)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestFacet('test_make_facet_polygons'))
    return suite


if __name__ == '__main__':
    unittest.main()

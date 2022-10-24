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
        # Test a region that encompasses the NCP
        facet_points, facet_polys = facet.make_facet_polygons([119.73, 138.08, 124.13, 115.74],
                                                              [89.92, 89.91, 89.89, 89.89],
                                                              126.52, 90.0, 0.3, 0.3)
        facet_points = np.round(facet_points, 2).tolist()
        facet_polys = [np.round(a, 2).tolist() for a in facet_polys]
        self.assertEqual(facet_points, [[119.73, 89.92],
                                        [138.08, 89.91],
                                        [124.13, 89.89],
                                        [115.74, 89.89]])
        self.assertEqual(facet_polys, [[[127.29, 89.91],
                                        [278.87, 89.83],
                                        [351.52, 89.79],
                                        [51.92, 89.84],
                                        [119.94, 89.9],
                                        [127.29, 89.91]],
                                       [[146.88, 89.84],
                                        [127.29, 89.91],
                                        [119.94, 89.9],
                                        [119.93, 89.85],
                                        [146.88, 89.84]],
                                       [[119.93, 89.85],
                                        [81.52, 89.79],
                                        [51.92, 89.84],
                                        [119.94, 89.9],
                                        [119.93, 89.85]],
                                       [[146.88, 89.84],
                                        [171.52, 89.79],
                                        [261.52, 89.79],
                                        [278.87, 89.83],
                                        [127.29, 89.91],
                                        [146.88, 89.84]]])


def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestFacet('test_make_facet_polygons'))
    return suite


if __name__ == '__main__':
    unittest.main()

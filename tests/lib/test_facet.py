import unittest
from rapthor.lib import facet
import numpy as np
import os


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
        facet_points = np.round(facet_points, 1).tolist()
        facet_polys = [np.round(a, 1).tolist() for a in facet_polys]

        # Check the facet points
        self.assertEqual(facet_points, [[119.7, 89.9],
                                        [138.1, 89.9],
                                        [124.1, 89.9],
                                        [115.7, 89.9]])

        # Check the facet polygons. Since the start point of each polygon can
        # be different from the control but the polygons still be identical,
        # we check only that each vertex is present in the control (and
        # vice versa)
        facet_polys_control = [[[127.3, 89.9],
                                [278.9, 89.8],
                                [351.5, 89.8],
                                [51.9, 89.8],
                                [119.9, 89.9],
                                [127.3, 89.9]],
                               [[127.3, 89.9],
                                [278.9, 89.8],
                                [261.5, 89.8],
                                [171.5, 89.8],
                                [146.9, 89.8],
                                [127.3, 89.9]],
                               [[127.3, 89.9],
                                [146.9, 89.8],
                                [119.9, 89.8],
                                [119.9, 89.9],
                                [127.3, 89.9]],
                               [[119.9, 89.9],
                                [51.9, 89.8],
                                [81.5, 89.8],
                                [119.9, 89.8],
                                [119.9, 89.9]]]
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
        self.assertTrue(all_present, msg='polys: {0}, control: {1}'.format(facet_polys_flat, facet_polys_control_flat))

    def test_read_ds9_region_file(self):
        facets = facet.read_ds9_region_file('resources/test.reg')
        self.assertEqual(len(facets), 15)
        self.assertEqual(facets[0].name, 'Patch_1')
        self.assertEqual(facets[0].ra, 318.2026666666666)
        self.assertEqual(facets[0].dec, 62.25055927777777)
        self.assertEqual(facets[1].name, 'Patch_10_with_spaces')
        self.assertEqual(facets[2].name, 'Patch_11')
        self.assertEqual(facets[3].name, 'Patch_12')

    def test_write_ds9_region_file(self):
        facets = facet.read_ds9_region_file('resources/test.reg')
        facet.make_ds9_region_file(facets, 'test_region_write.reg')
        facets = facet.read_ds9_region_file('test_region_write.reg')
        self.assertEqual(len(facets), 15)
        self.assertEqual(facets[0].name, 'Patch_1')
        self.assertEqual(facets[0].ra, 318.2026666666666)
        self.assertEqual(facets[0].dec, 62.25055927777777)
        self.assertEqual(facets[1].name, 'Patch_10_with_spaces')
        self.assertEqual(facets[2].name, 'Patch_11')
        self.assertEqual(facets[3].name, 'Patch_12')
        os.system('rm test_region_write.reg')


def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestFacet('test_make_facet_polygons'))
    suite.addTest(TestFacet('test_read_ds9_region_file'))
    suite.addTest(TestFacet('test_write_ds9_region_file'))
    return suite


if __name__ == '__main__':
    unittest.main()

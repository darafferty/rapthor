import unittest
import os
import requests
from rapthor.lib.parset import parset_read


class TestParset(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.parset_missing_parameter = """
                                        [global]
                                        dir_working = /data/rapthor
                                        # input_ms = /data/ms/*.ms  # missing required parameter
                                        """
        self.parset_misspelled_parameter = """
                                           [global]
                                           dir_working = /data/rapthor
                                           input_ms = /data/ms/*.ms

                                           [imaging]
                                           dd_method  # misspelled optional parameter
                                           """
        self.parset_missing_section = """
                                      # [global]  # missing required section
                                      dir_working = /data/rapthor
                                      input_ms = /data/ms/*.ms
                                      """
        self.parset_misspelled_section = """
                                         # [global]
                                         dir_working = /data/rapthor
                                         input_ms = /data/ms/*.ms

                                         [imging]  # misspelled optional section
                                         dde_method = facets
                                         """

    @classmethod
    def tearDownClass(self):
        pass

    def test_missing_parameter(self):
        parset_read(self.parset_missing_parameter)

    def test_misspelled_parameter(self):
        parset_read(self.parset_misspelled_parameter)

    def test_missing_section(self):
        parset_read(self.parset_missing_section)

    def test_misspelled_section(self):
        parset_read(self.parset_misspelled_section)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestParset('test_missing_parameter'))
    suite.addTest(TestParset('test_misspelled_parameter'))
    suite.addTest(TestParset('test_missing_section'))
    suite.addTest(TestParset('test_misspelled_section'))
    return suite


if __name__ == '__main__':
    unittest.main()

import unittest
import os
import requests
from rapthor.lib.parset import parset_read


class TestParset(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.parset_missing_parameter = 'missing_parameter.txt'
        with open(self.parset_missing_parameter, 'w') as f:
            f.write("""
                    [global]
                    dir_working = .
                    # input_ms = /data/ms/*.ms  # missing required parameter
                    """)
        self.parset_misspelled_parameter = 'misspelled_parameter.txt'
        with open(self.parset_misspelled_parameter, 'w') as f:
            f.write("""
                    [global]
                    dir_working = .
                    input_ms = /data/ms/*.ms

                    [imaging]
                    dd_method = facets  # misspelled optional parameter
                    """)
        self.parset_missing_section = 'missing_section.txt'
        with open(self.parset_missing_section, 'w') as f:
            f.write("""
                    # [global]  # missing required section
                    dir_working = .
                    input_ms = /data/ms/*.ms

                    [imging]  # misspelled optional section
                    dde_method = facets
                    """)
        self.parset_misspelled_section = 'misspelled_section.txt'
        with open(self.parset_misspelled_section, 'w') as f:
            f.write("""
                    [global]
                    dir_working = .
                    input_ms = /data/ms/*.ms

                    [imging]  # misspelled optional section
                    dde_method = facets
                    """)

    @classmethod
    def tearDownClass(self):
        os.system('rm *.txt')

    def test_missing_parameter(self):
        self.assertRaises(KeyError, parset_read, self.parset_missing_parameter)

    def test_misspelled_parameter(self):
        with self.assertLogs(logger='rapthor:parset', level='WARN') as cm:
            try:
                parset_read(self.parset_misspelled_parameter)
            except FileNotFoundError:
                # This is expected because the input MS file does not exist. We
                # ignore it as it happens after the check for misspelled parameters
                pass
            self.assertEqual(cm.output, ['WARNING:rapthor:parset:Option "dd_method" was '
                                         'given in the [imaging] section of the parset '
                                         'but is not a valid imaging option'])

    def test_missing_section(self):
        self.assertWarns(KeyError, parset_read, self.parset_missing_section)

    def test_misspelled_section(self):
        with self.assertLogs(logger='rapthor:parset', level='WARN') as cm:
            try:
                parset_read(self.parset_misspelled_parameter)
            except FileNotFoundError:
                # This is expected because the input MS file does not exist. We
                # ignore it as it happens after the check for misspelled sections
                pass
            self.assertEqual(cm.output, ['WARNING:rapthor:parset:Section "imging" was '
                                         'given in the parset but is not a valid '
                                         'section name'])


def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestParset('test_missing_parameter'))
    suite.addTest(TestParset('test_misspelled_parameter'))
    suite.addTest(TestParset('test_missing_section'))
    suite.addTest(TestParset('test_misspelled_section'))
    return suite


if __name__ == '__main__':
    unittest.main()

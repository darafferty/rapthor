import unittest
import os
import ast
import mock
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

                    [imaging]
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

        os.mkdir('dummy.ms')

    @classmethod
    def tearDownClass(self):
        os.system('rm *.txt')
        os.rmdir('dummy.ms')


    def test_missing_parset_file(self):
        self.assertRaises(FileNotFoundError, parset_read, 'this.parset.file.does.not.exist')

    def test_missing_parameter(self):
        # self.assertRaises(KeyError, parset_read, self.parset_missing_parameter)
        self.assertRaisesRegex(ValueError, 
            r"Missing required option\(s\) in section \[global\]:", 
            parset_read, self.parset_missing_parameter
        )

    def test_misspelled_parameter(self):
        with self.assertLogs(logger='rapthor:parset', level='WARN') as cm:
            try:
                parset_read(self.parset_misspelled_parameter)
            except FileNotFoundError:
                # This is expected because the input MS file does not exist. We
                # ignore it as it happens after the check for misspelled parameters
                pass
            self.assertEqual(cm.output, ["WARNING:rapthor:parset:Option 'dd_method' "
                                         "in section [imaging] is invalid"])

    def test_missing_section(self):
        self.assertRaisesRegex(ValueError,
            f"Parset file '{self.parset_missing_section}' could not be parsed correctly.",
            parset_read, self.parset_missing_section
        )

    def test_misspelled_section(self):
        with self.assertLogs(logger='rapthor:parset', level='WARN') as cm:
            try:
                parset_read(self.parset_misspelled_section)
            except FileNotFoundError:
                # This is expected because the input MS file does not exist. We
                # ignore it as it happens after the check for misspelled sections
                pass
            self.assertEqual(cm.output, ["WARNING:rapthor:parset:Section [imging] "
                                         "is invalid"])

    # Fix value of `cpu_count`, because `parset_read` does some smart things with it.
    @mock.patch("rapthor.lib.parset.multiprocessing.cpu_count", return_value=8)
    def test_minimal_parset(self, cpu_count):
        self.maxDiff = None
        parset = parset_read('resources/rapthor_minimal.parset')
        with open('resources/rapthor_minimal.parset.dict', 'r') as f:
            ref_parset = ast.literal_eval(f.read())
        self.assertEqual(parset, ref_parset)

    # Fix value of `cpu_count`, because `parset_read` does some smart things with it.
    @mock.patch("rapthor.lib.parset.multiprocessing.cpu_count", return_value=8)
    def test_complete_parset(self, cpu_count):
        self.maxDiff = None
        parset = parset_read('resources/rapthor_complete.parset')
        with open('resources/rapthor_complete.parset.dict', 'r') as f:
            ref_parset = ast.literal_eval(f.read())
        self.assertEqual(parset, ref_parset)


if __name__ == '__main__':
    unittest.main()

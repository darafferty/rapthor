"""
This module contains tests for the module `rapthor.lib.parset`
"""

import ast
import os
import string
import tempfile
import textwrap
import unittest
from unittest import mock
from rapthor.lib.parset import parset_read


class TestParset(unittest.TestCase):
    """
    This class contains tests for the public function `parset_read` in the module
    `rapthor.lib.parset`, which implicitly tests much of the `Parset` class in the same
    module.
    """

    @classmethod
    def setUpClass(cls):
        # Create dummy MS, an empty directory suffices
        cls.input_ms = tempfile.TemporaryDirectory(suffix=".ms")

        # Create an empty working directory
        cls.dir_working = tempfile.TemporaryDirectory()

        # Change directory to the tests directory (one level up from this file),
        # because that's where these tests need to be run from.
        os.chdir(os.path.join(os.path.dirname(__file__), ".."))

    @classmethod
    def tearDownClass(cls):
        cls.input_ms.cleanup()
        cls.dir_working.cleanup()

    def setUp(self):
        self.parset = tempfile.NamedTemporaryFile(
            suffix=".parset",
            dir=self.dir_working.name,
            delete=False,
        )
        with open(self.parset.name, "w") as f:
            f.write(
                textwrap.dedent(
                    f"""
                    [global]
                    input_ms = {self.input_ms.name}
                    dir_working = {self.dir_working.name}
                    """
                )
            )

    def test_missing_parset_file(self):
        os.unlink(self.parset.name)
        with self.assertRaises(FileNotFoundError):
            parset_read(self.parset.name)

    def test_empty_parset_file(self):
        with open(self.parset.name, "w") as f:
            pass
        with self.assertRaisesRegex(
            ValueError, r"Missing required option\(s\) in section \[global\]:"
        ):
            parset_read(self.parset.name)

    def test_minimal_parset(self):
        parset_dict = parset_read(self.parset.name)
        self.assertEqual(parset_dict["dir_working"], self.dir_working.name)
        self.assertEqual(parset_dict["input_ms"], self.input_ms.name)

    def test_misspelled_section(self):
        section = "misspelled_section"
        with open(self.parset.name, "a") as f:
            f.write(f"[{section}]")
        with self.assertLogs(logger="rapthor:parset", level="WARN") as cm:
            parset_read(self.parset.name)
            self.assertEqual(
                cm.output, [f"WARNING:rapthor:parset:Section [{section}] is invalid"]
            )

    def test_misspelled_option(self):
        option = "misspelled_option"
        with open(self.parset.name, "a") as f:
            f.write(f"{option} = some value")
        with self.assertLogs(logger="rapthor:parset", level="WARN") as cm:
            parset_read(self.parset.name)
            self.assertEqual(
                cm.output,
                [
                    f"WARNING:rapthor:parset:Option '{option}' in section [global] is invalid"
                ],
            )

    def test_deprecated_option(self):
        section = "[cluster]"
        option = "dir_local"
        with open(self.parset.name, "a") as f:
            f.write(f"{section}\n")
            f.write(f"{option} = some value")
        with self.assertLogs(logger="rapthor:parset", level="WARN") as cm:
            parset_read(self.parset.name)
            self.assertIn(
                f"WARNING:rapthor:parset:Option '{option}' in section {section} is deprecated",
                cm.output[0],
            )

    def test_fraction_out_of_range(self):
        option = "selfcal_data_fraction"
        value = 1.1
        with open(self.parset.name, "a") as f:
            f.write(f"{option} = {value}")
        with self.assertRaisesRegex(
            ValueError, f"The {option} parameter is {value}; it must be > 0 and <= 1"
        ):
            parset_read(self.parset.name)

    def test_invalid_idg_mode(self):
        option = "idg_mode"
        value = "invalid"
        with open(self.parset.name, "a") as f:
            f.write("[imaging]\n")
            f.write(f"{option} = {value}")
        with self.assertRaisesRegex(
            ValueError, f"The option '{option}' must be one of"
        ):
            parset_read(self.parset.name)

    def test_unequal_sector_list_lengths(self):
        with open(self.parset.name, "a") as f:
            f.write("[imaging]\n")
            f.write("sector_center_ra_list = [1]")
        with self.assertRaisesRegex(
            ValueError, "The options .* must all have the same number of entries"
        ):
            parset_read(self.parset.name)

    # Fix value of `cpu_count`, because `parset_read` does some smart things with it.
    @mock.patch("rapthor.lib.parset.misc.nproc", return_value=8)
    def test_default_parset_contents(self, cpu_count):
        self.maxDiff = None
        with open(self.parset.name, "w") as f:
            f.write(
                string.Template(
                    open("resources/rapthor_minimal.parset.template").read()
                ).substitute(
                    dir_working=self.dir_working.name, input_ms=self.input_ms.name
                )
            )
        parset = parset_read(self.parset.name)
        ref_parset = ast.literal_eval(
            string.Template(
                open("resources/rapthor_minimal.parset_dict.template").read()
            ).substitute(dir_working=self.dir_working.name, input_ms=self.input_ms.name)
        )
        self.assertEqual(parset, ref_parset)

    # Fix value of `cpu_count`, because `parset_read` does some smart things with it.
    @mock.patch("rapthor.lib.parset.misc.nproc", return_value=8)
    def test_complete_parset_contents(self, cpu_count):
        self.maxDiff = None
        with open(self.parset.name, "w") as f:
            f.write(
                string.Template(
                    open("resources/rapthor_complete.parset.template").read()
                ).substitute(
                    dir_working=self.dir_working.name, input_ms=self.input_ms.name
                )
            )
        parset = parset_read(self.parset.name)
        ref_parset = ast.literal_eval(
            string.Template(
                open("resources/rapthor_complete.parset_dict.template").read()
            ).substitute(dir_working=self.dir_working.name, input_ms=self.input_ms.name)
        )
        self.assertEqual(parset, ref_parset)


if __name__ == "__main__":
    unittest.main()

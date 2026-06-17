"""
This module contains tests for the module `rapthor.lib.parset`
"""

import ast
import logging
import os
import string
import tempfile
import textwrap
import unittest

from rapthor.testing import assert_logged

try:
    import mock
except ImportError:
    from unittest import mock

from rapthor.lib.parset import check_and_adjust_skymodel_settings, parset_read


def assert_warning_logged(caplog, expected_messages=(), expected_message=None):
    return assert_logged(
        caplog,
        "rapthor:parset",
        logging.WARNING,
        expected_messages,
        expected_message=expected_message,
    )


def assert_info_logged(caplog, expected_messages=(), expected_message=None):
    return assert_logged(
        caplog,
        "rapthor:parset",
        logging.INFO,
        expected_messages,
        expected_message=expected_message,
    )


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

    def test_misspelled_section(self, caplog):
        section = "misspelled_section"
        with open(self.parset.name, "a") as f:
            f.write(f"[{section}]")
        with assert_warning_logged(
            caplog, f"WARNING:rapthor:parset:Section [{section}] is invalid"
        ):
            parset_read(self.parset.name)

    def test_misspelled_option(self, caplog):
        option = "misspelled_option"
        with open(self.parset.name, "a") as f:
            f.write(f"{option} = some value")
        with assert_warning_logged(
            caplog, f"WARNING:rapthor:parset:Option '{option}' in section [global] is invalid"
        ):
            parset_read(self.parset.name)

    def test_deprecated_option(self, caplog):
        section = "[cluster]"
        option = "dir_local"
        with open(self.parset.name, "a") as f:
            f.write(f"{section}\n")
            f.write(f"{option} = some value")
        with assert_warning_logged(
            caplog, f"WARNING:rapthor:parset:Option '{option}' in section {section} is deprecated"
        ):
            parset_read(self.parset.name)

    def test_fraction_out_of_range(self):
        option = "selfcal_data_fraction"
        value = 1.1
        with open(self.parset.name, "a") as f:
            f.write(f"{option} = {value}")
        with self.assertRaisesRegex(
            ValueError,
            f"The {option} parameter is {value}; it must be > 0 and <= 1",
        ):
            parset_read(self.parset.name)

    def test_invalid_idg_mode(self):
        option = "idg_mode"
        value = "invalid"
        with open(self.parset.name, "a") as f:
            f.write("[imaging]\n")
            f.write(f"{option} = {value}")
        with self.assertRaisesRegex(ValueError, f"The option '{option}' must be one of"):
            parset_read(self.parset.name)

    def test_unequal_sector_list_lengths(self):
        with open(self.parset.name, "a") as f:
            f.write("[imaging]\n")
            f.write("sector_center_ra_list = [1]")
        with self.assertRaisesRegex(
            ValueError,
            "The options .* must all have the same number of entries",
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
                    dir_working=self.dir_working.name,
                    input_ms=self.input_ms.name,
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
                    dir_working=self.dir_working.name,
                    input_ms=self.input_ms.name,
                )
            )
        parset = parset_read(self.parset.name)
        ref_parset = ast.literal_eval(
            string.Template(
                open("resources/rapthor_complete.parset_dict.template").read()
            ).substitute(dir_working=self.dir_working.name, input_ms=self.input_ms.name)
        )
        self.assertEqual(parset, ref_parset)


class TestCheckSkymodelSettings(unittest.TestCase):
    """
    Tests for the `check_and_adjust_skymodel_settings` function.
    """

    def _make_parset_dict(self, **overrides):
        """Helper to create a minimal parset_dict with sensible defaults."""
        parset_dict = {
            "input_skymodel": None,
            "generate_initial_skymodel": False,
            "download_initial_skymodel": False,
            "apparent_skymodel": None,
            "cluster_specific": {"allow_internet_access": True},
            "imaging_specific": {
                "astrometry_skymodel": None,
                "photometry_skymodel": None,
                "normalization_skymodels": None,
                "normalization_reference_frequencies": None,
            },
        }
        for key, value in overrides.items():
            if key in ("cluster_specific", "imaging_specific"):
                parset_dict[key].update(value)
            else:
                parset_dict[key] = value
        return parset_dict

    # ---- input_skymodel given ----

    def test_input_skymodel_not_found_raises(self):
        parset_dict = self._make_parset_dict(input_skymodel="/nonexistent/skymodel.txt")
        with self.assertRaises(FileNotFoundError):
            check_and_adjust_skymodel_settings(parset_dict)

    def test_input_skymodel_exists(self):
        with tempfile.NamedTemporaryFile(suffix=".skymodel") as f:
            parset_dict = self._make_parset_dict(input_skymodel=f.name)
            # Should not raise
            check_and_adjust_skymodel_settings(parset_dict)

    def test_input_skymodel_with_generate_disables_download(self, caplog):
        with tempfile.NamedTemporaryFile(suffix=".skymodel") as f:
            parset_dict = self._make_parset_dict(
                input_skymodel=f.name,
                generate_initial_skymodel=True,
                download_initial_skymodel=True,
            )
            with assert_warning_logged(caplog, "Sky model generation requested"):
                check_and_adjust_skymodel_settings(parset_dict)
            self.assertFalse(parset_dict["download_initial_skymodel"])

    def test_input_skymodel_with_download_disables_download(self, caplog):
        with tempfile.NamedTemporaryFile(suffix=".skymodel") as f:
            parset_dict = self._make_parset_dict(
                input_skymodel=f.name,
                download_initial_skymodel=True,
            )
            with assert_warning_logged(
                caplog, ["Sky model download requested", "Disabling download"]
            ):
                check_and_adjust_skymodel_settings(parset_dict)
            self.assertFalse(parset_dict["download_initial_skymodel"])

    # ---- no input_skymodel, generate requested ----

    def test_no_input_generate_requested(self, caplog):
        parset_dict = self._make_parset_dict(generate_initial_skymodel=True)
        with assert_info_logged(caplog, "Will automatically generate sky model"):
            check_and_adjust_skymodel_settings(parset_dict)

    def test_no_input_generate_requested_with_apparent_warns(self, caplog):
        parset_dict = self._make_parset_dict(
            generate_initial_skymodel=True,
            apparent_skymodel="some_apparent.skymodel",
        )
        with assert_warning_logged(caplog, "apparent sky model will not be used"):
            check_and_adjust_skymodel_settings(parset_dict)

    # ---- no input_skymodel, download requested ----

    def test_no_input_download_requested(self, caplog):
        parset_dict = self._make_parset_dict(download_initial_skymodel=True)
        with assert_info_logged(caplog, "Will automatically download sky model"):
            check_and_adjust_skymodel_settings(parset_dict)

    def test_no_input_download_requested_with_apparent_warns(self, caplog):
        parset_dict = self._make_parset_dict(
            download_initial_skymodel=True,
            apparent_skymodel="some_apparent.skymodel",
        )
        with assert_warning_logged(caplog, "apparent sky model will not be used"):
            check_and_adjust_skymodel_settings(parset_dict)

    # ---- no input_skymodel, neither generate nor download ----

    def test_no_input_no_generate_no_download_warns(self, caplog):
        parset_dict = self._make_parset_dict()
        with assert_warning_logged(caplog, "neither generation nor download"):
            check_and_adjust_skymodel_settings(parset_dict)

    # ---- internet access checks ----

    def test_download_no_internet_raises(self):
        parset_dict = self._make_parset_dict(
            download_initial_skymodel=True,
            cluster_specific={"allow_internet_access": False},
        )
        with self.assertRaises(ValueError):
            check_and_adjust_skymodel_settings(parset_dict)

    def test_internet_allowed_download_ok(self):
        parset_dict = self._make_parset_dict(
            download_initial_skymodel=True,
            cluster_specific={"allow_internet_access": True},
        )
        # Should not raise
        check_and_adjust_skymodel_settings(parset_dict)

    # ---- diagnostic and normalization skymodel checks (no internet) ----

    def test_astrometry_skymodel_missing_no_internet_raises(self):
        parset_dict = self._make_parset_dict(
            generate_initial_skymodel=True,
            cluster_specific={"allow_internet_access": False},
            imaging_specific={
                "astrometry_skymodel": "/nonexistent/astro.skymodel",
                "photometry_skymodel": None,
                "normalization_skymodels": None,
                "normalization_reference_frequencies": None,
            },
        )
        with self.assertRaises(FileNotFoundError):
            check_and_adjust_skymodel_settings(parset_dict)

    def test_photometry_skymodel_missing_no_internet_raises(self):
        parset_dict = self._make_parset_dict(
            generate_initial_skymodel=True,
            cluster_specific={"allow_internet_access": False},
            imaging_specific={
                "astrometry_skymodel": None,
                "photometry_skymodel": "/nonexistent/photo.skymodel",
                "normalization_skymodels": None,
                "normalization_reference_frequencies": None,
            },
        )
        with self.assertRaises(FileNotFoundError):
            check_and_adjust_skymodel_settings(parset_dict)

    def test_normalization_skymodel_missing_no_internet_raises(self):
        parset_dict = self._make_parset_dict(
            generate_initial_skymodel=True,
            cluster_specific={"allow_internet_access": False},
            imaging_specific={
                "astrometry_skymodel": None,
                "photometry_skymodel": None,
                "normalization_skymodels": [
                    "/nonexistent/norm.skymodel",
                    "/nonexistent/norm2.skymodel",
                ],
                "normalization_reference_frequencies": None,
            },
        )
        with self.assertRaises(FileNotFoundError):
            check_and_adjust_skymodel_settings(parset_dict)

    def test_astrometry_skymodel_exists_no_internet_ok(self, caplog):
        with tempfile.NamedTemporaryFile(suffix=".skymodel") as f:
            parset_dict = self._make_parset_dict(
                cluster_specific={"allow_internet_access": False},
                imaging_specific={
                    "astrometry_skymodel": f.name,
                    "photometry_skymodel": None,
                },
            )
            # Should not raise (warning about no skymodel is expected)
            with assert_warning_logged(caplog, "The photometry check will be skipped"):
                check_and_adjust_skymodel_settings(parset_dict)

    def test_photometry_skymodel_exists_no_internet_ok(self, caplog):
        with tempfile.NamedTemporaryFile(suffix=".skymodel") as f:
            parset_dict = self._make_parset_dict(
                cluster_specific={"allow_internet_access": False},
                imaging_specific={
                    "astrometry_skymodel": None,
                    "photometry_skymodel": f.name,
                },
            )
            # Should not raise (warning about no skymodel is expected)
            with assert_warning_logged(caplog, "The astrometry check will be skipped"):
                check_and_adjust_skymodel_settings(parset_dict)

    def test_normalization_skymodel_exists_no_internet_ok(self):
        with tempfile.NamedTemporaryFile(suffix=".skymodel") as f:
            parset_dict = self._make_parset_dict(
                cluster_specific={"allow_internet_access": False},
                imaging_specific={
                    "normalization_skymodels": [f.name, f.name],
                    "normalization_reference_frequencies": [str(142000000.0), str(142001000.0)],
                },
            )
            # Should not raise (warning about no skymodel is expected)
            check_and_adjust_skymodel_settings(parset_dict)

    def test_single_normalization_skymodel_raises_error(self):
        with tempfile.NamedTemporaryFile(suffix=".skymodel") as f:
            parset_dict = self._make_parset_dict(
                cluster_specific={"allow_internet_access": False},
                imaging_specific={
                    "normalization_skymodels": f.name,
                    "normalization_reference_frequencies": [str(142000000.0)],
                },
            )
            # Should raise ValueError because only one normalization skymodel is provided
            with self.assertRaises(ValueError):
                check_and_adjust_skymodel_settings(parset_dict)

    def test_only_one_existing_path_for_normalization_skymodels_raises_error(self):
        with tempfile.NamedTemporaryFile(suffix=".skymodel") as f:
            parset_dict = self._make_parset_dict(
                cluster_specific={"allow_internet_access": False},
                imaging_specific={
                    "normalization_skymodels": [f.name, "/nonexistent/norm.skymodel"],
                    "normalization_reference_frequencies": [str(142000000.0), str(142001000.0)],
                },
            )
            with self.assertRaises(FileNotFoundError):
                check_and_adjust_skymodel_settings(parset_dict)

    def test_normalization_reference_frequencies_missing_raises_error(self):
        with tempfile.NamedTemporaryFile(suffix=".skymodel") as f:
            parset_dict = self._make_parset_dict(
                cluster_specific={"allow_internet_access": False},
                imaging_specific={
                    "normalization_skymodels": [f.name, f.name],
                    "normalization_reference_frequencies": None,
                },
            )
            with self.assertRaises(ValueError):
                check_and_adjust_skymodel_settings(parset_dict)

    def test_normalization_reference_frequencies_wrong_length_raises_error(self):
        with tempfile.NamedTemporaryFile(suffix=".skymodel") as f:
            parset_dict = self._make_parset_dict(
                cluster_specific={"allow_internet_access": False},
                imaging_specific={
                    "normalization_skymodels": [f.name, f.name],
                    "normalization_reference_frequencies": [str(142000000.0)],
                },
            )
            with self.assertRaises(ValueError):
                check_and_adjust_skymodel_settings(parset_dict)

    def test_normalization_parameters_tuple(self):
        with tempfile.NamedTemporaryFile(suffix=".skymodel") as f:
            parset_dict = self._make_parset_dict(
                cluster_specific={"allow_internet_access": False},
                imaging_specific={
                    "normalization_skymodels": (f.name, f.name),
                    "normalization_reference_frequencies": (142000000.0, 142001000.0),
                },
            )
            check_and_adjust_skymodel_settings(parset_dict)

    def test_normalization_parameters_set_raises_error(self):
        with tempfile.NamedTemporaryFile(suffix=".skymodel") as f:
            parset_dict = self._make_parset_dict(
                cluster_specific={"allow_internet_access": False},
                imaging_specific={
                    "normalization_skymodels": {f.name, f.name},
                    "normalization_reference_frequencies": {142000000.0, 142001000.0},
                },
            )
            with self.assertRaises(ValueError):
                check_and_adjust_skymodel_settings(parset_dict)

    def test_diagnostic_skymodel_empty_no_internet_ok(self, caplog):
        parset_dict = self._make_parset_dict(
            cluster_specific={"allow_internet_access": False},
            imaging_specific={
                "astrometry_skymodel": None,
                "photometry_skymodel": None,
                "normalization_skymodels": None,
                "normalization_reference_frequencies": None,
            },
        )
        # Should not raise
        with assert_warning_logged(
            caplog, ["The astrometry check will be skipped", "The photometry check will be skipped"]
        ):
            check_and_adjust_skymodel_settings(parset_dict)


if __name__ == "__main__":
    unittest.main()

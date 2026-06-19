"""
This module contains tests for the module `rapthor.lib.parset`
"""

import ast
import contextlib
import logging
import os
import string
import tempfile
import textwrap
import unittest
from unittest import mock

import pytest

from rapthor.lib.parset import check_and_adjust_skymodel_settings, parset_read
from rapthor.testing import assert_logged


def assert_warning_logged(caplog, *expected_messages):
    return assert_logged(
        caplog,
        "rapthor:parset",
        logging.WARNING,
        *expected_messages,
    )


def assert_info_logged(caplog, *expected_messages):
    return assert_logged(
        caplog,
        "rapthor:parset",
        logging.INFO,
        *expected_messages,
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

    def test_misspelled_section(self):
        section = "misspelled_section"
        with open(self.parset.name, "a") as f:
            f.write(f"[{section}]")
        with self.assertLogs(logger="rapthor:parset", level="WARN") as cm:
            parset_read(self.parset.name)
            self.assertEqual(
                cm.output,
                [f"WARNING:rapthor:parset:Section [{section}] is invalid"],
            )

    def test_misspelled_option(self):
        option = "misspelled_option"
        with open(self.parset.name, "a") as f:
            f.write(f"{option} = some value")
        with self.assertLogs(logger="rapthor:parset", level="WARN") as cm:
            parset_read(self.parset.name)
            self.assertEqual(
                cm.output,
                [f"WARNING:rapthor:parset:Option '{option}' in section [global] is invalid"],
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


class TestCheckSkymodelSettings:
    """Tests for the `check_and_adjust_skymodel_settings` function."""

    def _make_parset_dict(self, **overrides):
        """
        Helper to create a minimal parset_dict with sensible defaults.
        """
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

    @pytest.fixture
    def mock_skymodel_path(self, tmp_path):
        path = tmp_path / "mock.skymodel"
        path.touch()
        return path

    # ---- input_skymodel given ----

    @pytest.mark.parametrize(
        "resolve_fixture_values, context",
        [
            # Test nominal case where input skymodel is provided and exists.
            (mock_skymodel_path, contextlib.nullcontext()),
            # Test that a non-existent input skymodel raises FileNotFoundError.
            ("/nonexistent/skymodel.txt", pytest.raises(FileNotFoundError)),
        ],
        indirect=["resolve_fixture_values"],
    )
    def test_input_skymodel_existence(self, resolve_fixture_values, context):
        """
        Test that providing an existing input skymodel works, and that a
        non-existent input skymodel raises FileNotFoundError.
        """
        parset_dict = self._make_parset_dict(input_skymodel=resolve_fixture_values)
        with context:
            check_and_adjust_skymodel_settings(parset_dict)

    @pytest.mark.parametrize(
        "generate, expected_warnings",
        [
            (True, ["Sky model generation requested"]),
            (False, ["Sky model download requested", "Disabling download"]),
        ],
    )
    def test_input_skymodel_disables_download(
        self, caplog, mock_skymodel_path, generate, expected_warnings
    ):
        """
        Test that download is disabled when an input skymodel is provided.
        """
        parset_dict = self._make_parset_dict(
            input_skymodel=mock_skymodel_path,
            generate_initial_skymodel=generate,
            download_initial_skymodel=True,
        )
        with assert_warning_logged(caplog, *expected_warnings):
            check_and_adjust_skymodel_settings(parset_dict)

        assert parset_dict["download_initial_skymodel"] is False

    # ---- no input_skymodel, generate / requested ----

    @pytest.mark.parametrize(
        "config, expected_warning",
        [
            (
                {"generate_initial_skymodel": True},
                "Will automatically generate sky model",
            ),
            (
                {
                    "generate_initial_skymodel": True,
                    "apparent_skymodel": "some_apparent.skymodel",
                },
                "apparent sky model will not be used",
            ),
            (
                {"download_initial_skymodel": True},
                "Will automatically download sky model",
            ),
            (
                {
                    "download_initial_skymodel": True,
                    "apparent_skymodel": "some_apparent.skymodel",
                },
                "apparent sky model will not be used",
            ),
            ({}, "neither generation nor download"),
        ],
    )
    def test_no_input_skymodel_generate_or_download_requested(
        self, caplog, config, expected_warning
    ):
        """
        Test that the correct messages are logged when no input skymodel is
        given and generate/download is requested.
        """
        parset_dict = self._make_parset_dict(**config)
        with assert_info_logged(caplog, expected_warning):
            check_and_adjust_skymodel_settings(parset_dict)

    # ---- internet access checks ----

    @pytest.mark.parametrize(
        "allow_internet_access, context",
        [
            (True, contextlib.nullcontext()),
            (False, pytest.raises(ValueError)),
        ],
    )
    def test_download_with_internet_access(self, allow_internet_access, context):
        """
        Test that download requires internet access to be allowed.
        """
        parset_dict = self._make_parset_dict(
            download_initial_skymodel=True,
            cluster_specific={"allow_internet_access": allow_internet_access},
        )
        with context:
            check_and_adjust_skymodel_settings(parset_dict)

    # ---- diagnostic and normalization skymodel checks (no internet) ----

    @pytest.mark.parametrize(
        "skymodel_name, skymodel_path",
        [
            ("astrometry_skymodel", "/nonexistent/astro.skymodel"),
            ("photometry_skymodel", "/nonexistent/photo.skymodel"),
            (
                "normalization_skymodels",
                ["/nonexistent/norm1.skymodel", "/nonexistent/norm2.skymodel"],
            ),
        ],
    )
    def test_skymodel_missing_no_internet_raises(self, skymodel_name, skymodel_path):
        """
        Test that missing diagnostic/normalization skymodels raise
        FileNotFoundError without internet.
        """
        parset_dict = self._make_parset_dict(
            generate_initial_skymodel=True,
            cluster_specific={"allow_internet_access": False},
            imaging_specific={skymodel_name: skymodel_path},
        )
        with pytest.raises(FileNotFoundError):
            check_and_adjust_skymodel_settings(parset_dict)

    @pytest.mark.parametrize(
        "diagnostic",
        ["astrometry", "photometry"],
    )
    def test_skymodel_exists_no_internet_ok(self, caplog, mock_skymodel_path, diagnostic):
        """
        Test that existing diagnostic skymodels are accepted without internet access.
        """
        parset_dict = self._make_parset_dict(
            cluster_specific={"allow_internet_access": False},
            imaging_specific={f"{diagnostic}_skymodel": mock_skymodel_path},
        )
        # Should not raise (warning about no skymodel is expected)
        other = ({"astrometry", "photometry"} - {diagnostic}).pop()
        with assert_warning_logged(
            caplog,
            f"The {other} check will be skipped",
        ):
            check_and_adjust_skymodel_settings(parset_dict)

    def test_normalization_skymodel_exists_no_internet_ok(self, caplog, mock_skymodel_path):
        """
        Test that existing normalization skymodels are accepted without internet access.
        """
        parset_dict = self._make_parset_dict(
            cluster_specific={"allow_internet_access": False},
            imaging_specific={
                "normalization_skymodels": [mock_skymodel_path, mock_skymodel_path],
                "normalization_reference_frequencies": [str(142000000.0), str(142001000.0)],
            },
        )
        # Should not raise (warning about no skymodel is expected)
        with assert_warning_logged(
            caplog,
            "Comparison sky model for astrometry check not provided while "
            "`allow_internet_access` is False. The astrometry check will be skipped.",
            "Comparison sky model for photometry check not provided while "
            "`allow_internet_access` is False. The photometry check will be skipped.",
        ):
            check_and_adjust_skymodel_settings(parset_dict)

    def test_diagnostic_skymodel_empty_no_internet_ok(self, caplog):
        """
        Test that unset diagnostic skymodels produce skip warnings without internet.
        """
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
            caplog,
            "The astrometry check will be skipped",
            "The photometry check will be skipped",
        ):
            check_and_adjust_skymodel_settings(parset_dict)

    @pytest.mark.parametrize(
        "resolve_fixture_values, context",
        [
            # Nominal case: skymodels and frequencies are given as lists
            pytest.param(
                {
                    "normalization_skymodels": [mock_skymodel_path, mock_skymodel_path],
                    "normalization_reference_frequencies": [142000000.0, 142001000.0],
                },
                contextlib.nullcontext(),
                id="normalization_parameters_list",
            ),
            # Tuple case: skymodels and frequencies can be tuples as well.
            pytest.param(
                {
                    "normalization_skymodels": (mock_skymodel_path, mock_skymodel_path),
                    "normalization_reference_frequencies": (142000000.0, 142001000.0),
                },
                contextlib.nullcontext(),
                id="normalization_parameters_tuple",
            ),
            # Error cases
            # ---------------------------------------------------------------- #
            # Only one normalization skymodel provided, should raise ValueError
            pytest.param(
                {
                    "normalization_skymodels": mock_skymodel_path,
                    "normalization_reference_frequencies": [str(142000000.0)],
                },
                pytest.raises(ValueError),
                id="single_normalization_skymodel_raises_error",
            ),
            # One of the normalization skymodels does not exist, should raise FileNotFoundError
            pytest.param(
                {
                    "normalization_skymodels": [mock_skymodel_path, "/nonexistent/norm.skymodel"],
                    "normalization_reference_frequencies": [str(142000000.0), str(142001000.0)],
                },
                pytest.raises(FileNotFoundError),
                id="one_normalization_skymodel_missing_raises_error",
            ),
            # Normalization reference frequencies missing, should raise ValueError
            pytest.param(
                {
                    "normalization_skymodels": [mock_skymodel_path, mock_skymodel_path],
                    "normalization_reference_frequencies": None,
                },
                pytest.raises(ValueError),
                id="normalization_reference_frequencies_missing_raises_error",
            ),
            # Normalization reference frequencies length mismatch, should raise ValueError
            pytest.param(
                {
                    "normalization_skymodels": [mock_skymodel_path, mock_skymodel_path],
                    "normalization_reference_frequencies": [str(142000000.0)],
                },
                pytest.raises(ValueError),
                id="normalization_reference_frequencies_wrong_length_raises_error",
            ),
            # Invalid case: only one normalization skymodel provided as tuple, should raise ValueError
            pytest.param(
                {
                    "normalization_skymodels": (mock_skymodel_path),
                    "normalization_reference_frequencies": (142000000.0, 142001000.0),
                },
                pytest.raises(ValueError),
                id="normalization_parameters_set_raises_error",
            ),
        ],
        indirect=["resolve_fixture_values"],
    )
    def test_normalization_skymodel_input(self, resolve_fixture_values, context):
        """
        Test validation of normalization skymodel and reference frequency inputs.
        """
        parset_dict = self._make_parset_dict(
            cluster_specific={"allow_internet_access": False},
            imaging_specific=resolve_fixture_values,
        )
        with context:
            check_and_adjust_skymodel_settings(parset_dict)

"""
This module contains tests for the module `rapthor.lib.parset`
"""

import ast
import configparser
import contextlib
import logging
import os
import string
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

import pytest

from rapthor.lib.parset import check_and_adjust_skymodel_settings, parset_read


def _generate_parset(template_parset=None, config=None, output_path=None, **kws):

    parset = configparser.ConfigParser()
    if isinstance(template_parset, configparser.ConfigParser):
        parset = template_parset
    elif isinstance(template_parset, (str, Path)):
        parset.read(template_parset)
    elif template_parset is not None:
        raise TypeError(
            "Invalid type for template_parset. Expected str, Path, or ConfigParser.",
        )

    config = config or {}
    if kws:
        config["global"].update(kws)
    for section, options in config.items():
        if section is not None and section not in parset:
            parset.add_section(section)

        for option, value in options.items():
            parset.set(section, str(option), str(value))

    if output_path:
        with output_path.open("w") as fp:
            parset.write(fp)

    return parset


@contextlib.contextmanager
def assert_message_logged(caplog, logger, level, expected_message):

    with caplog.at_level(level, logger=logger):
        yield

        for record in caplog.records:
            if expected_message in record.message:
                return

        raise AssertionError(f'Expected message "{expected_message}" not found in logs')


def _test_parset_read_logs_warning(caplog, parset, expected_message):
    with assert_message_logged(
        caplog,
        logger="rapthor:parset",
        level=logging.WARNING,
        expected_message=expected_message,
    ):
        parset_read(parset)


class TestParset:
    """
    This class contains tests for the public function `parset_read` in the
    module `rapthor.lib.parset`, which implicitly tests much of the `Parset`
    class in the same module.
    """

    @pytest.fixture()
    def minimal_parset(self, tmp_path):

        # Create an empty file to represent the measurement set
        mock_input_ms = tmp_path / "test.ms"
        mock_input_ms.touch()

        return _generate_parset(
            config={
                "global": {
                    "input_ms": mock_input_ms,
                    "dir_working": tmp_path,
                }
            }
        )

    @pytest.fixture()
    def parset(self, tmp_path, minimal_parset, request):

        # Create the parset
        config = {}
        if getattr(request, "param", None):
            section, option, value = request.param
            config = {section: {option: value}}

        parset_path = tmp_path / "test.parset"
        _generate_parset(minimal_parset, config, parset_path)
        return parset_path

    # ------------------------------------------------------------------------ #
    # Tests 

    def test_missing_parset_file(self):
        """
        Test that reading a non-existent parset file raises FileNotFoundError.
        """
        with pytest.raises(FileNotFoundError):
            parset_read("non-existent-file")

    def test_empty_parset_file(self, tmp_path):
        """
        Test that reading an empty parset file raises ValueError due to missing
        required options.
        """
        parset = tmp_path / "empty.parset"
        parset.touch()  # Create an empty file

        with pytest.raises(
            ValueError, match=r"Missing required option\(s\) in section \[global\]:"
        ):
            parset_read(parset)

    def test_minimal_parset(self, tmp_path, parset):
        """
        Test that a minimal valid parset is read correctly with expected
        dir_working and input_ms values.
        """
        parset_dict = parset_read(parset)
        assert parset_dict["dir_working"] == str(tmp_path)
        assert parset_dict["input_ms"] == str(tmp_path / "test.ms")

    @pytest.mark.parametrize(
        "parset, expected_message",
        [
            pytest.param(
                ("misspelled_section", None, None),
                "Section [misspelled_section] is invalid",
                id="misspelled_section",
            ),
            pytest.param(
                ("global", "misspelled_option", "some value"),
                "Option 'misspelled_option' in section [global] is invalid",
                id="misspelled_option",
            ),
        ],
        indirect=["parset"],
    )
    def test_misconfigured(self, parset, caplog, expected_message):
        """
        Test that invalid sections or options in the parset produce appropriate
        warning log messages.
        """
        _test_parset_read_logs_warning(caplog, parset, expected_message)

    @pytest.mark.parametrize(
        "parset, expected_message",
        [
            (
                (section, option, "some value"),
                f"Option '{option}' in section [{section}] is deprecated",
            )
            for section, option in [
                ("cluster", "dir_local"),
            ]
        ],
        indirect=["parset"],
    )
    def test_deprecated_option(self, parset, caplog, expected_message):
        """
        Test that using deprecated options logs a deprecation warning.
        """
        _test_parset_read_logs_warning(caplog, parset, expected_message)

    @pytest.mark.parametrize(
        "parset, expected_message",
        [
            (
                (section, option, value),
                message.format(option=option, value=value),
            )
            for (section, option, value), message in [
                (
                    ("global", "selfcal_data_fraction", 1.1),
                    "The {option} parameter is {value}; it must be > 0 and <= 1",
                ),
                (
                    ("imaging", "idg_mode", "invalid"),
                    "The option '{option}' must be one of",
                ),
                (
                    ("imaging", "sector_center_ra_list", "[1]"),
                    "The options .* must all have the same number of entries",
                ),
            ]
        ],
        indirect=["parset"],
    )
    def test_validation(self, parset, expected_message):
        """
        Test that invalid parameter values (out-of-range, invalid enum,
        mismatched list lengths) raise ValueError.
        """
        with pytest.raises(ValueError, match=expected_message):
            parset_read(parset)

    @pytest.mark.parametrize("template_id", ["minimal", "complete"])
    def test_default_parset_contents(self, tmp_path, mocker, request, minimal_parset, template_id):
        """
        Test that reading a template parset produces a dict matching the
        expected reference output.
        """
        
        resources = request.config.resource_dir
        template_path = resources / f"rapthor_{template_id}.parset.template"
        reference_path = resources / f"rapthor_{template_id}.parset_dict.template"

        # Fix value of `cpu_count`, because `parset_read` does some smart things with it.
        mocker.patch("rapthor.lib.parset.misc.nproc", return_value=8)

        # Read the template
        template_parset = configparser.ConfigParser()
        template_parset.read(template_path)

        parset_path = tmp_path / "test.parset"
        _generate_parset(template_parset, minimal_parset, parset_path)

        parset = parset_read(parset_path)

        mock_input_ms = str(tmp_path / "test.ms")
        reference_dict = ast.literal_eval(reference_path.read_text())
        reference_dict.update(
            dir_working=str(tmp_path), input_ms=mock_input_ms, mss=[mock_input_ms]
        )
        assert parset == reference_dict



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
        with pytest.raises(FileNotFoundError):
            check_and_adjust_skymodel_settings(parset_dict)

    def test_input_skymodel_exists(self):
        with tempfile.NamedTemporaryFile(suffix=".skymodel") as f:
            parset_dict = self._make_parset_dict(input_skymodel=f.name)
            # Should not raise
            check_and_adjust_skymodel_settings(parset_dict)

    def test_input_skymodel_with_generate_disables_download(self):
        with tempfile.NamedTemporaryFile(suffix=".skymodel") as f:
            parset_dict = self._make_parset_dict(
                input_skymodel=f.name,
                generate_initial_skymodel=True,
                download_initial_skymodel=True,
            )
            with self.assertLogs(logger="rapthor:parset", level="WARN") as cm:
                check_and_adjust_skymodel_settings(parset_dict)
            self.assertFalse(parset_dict["download_initial_skymodel"])
            self.assertTrue(any("Sky model generation requested" in msg for msg in cm.output))

    def test_input_skymodel_with_download_disables_download(self):
        with tempfile.NamedTemporaryFile(suffix=".skymodel") as f:
            parset_dict = self._make_parset_dict(
                input_skymodel=f.name,
                download_initial_skymodel=True,
            )
            with self.assertLogs(logger="rapthor:parset", level="WARN") as cm:
                check_and_adjust_skymodel_settings(parset_dict)
            self.assertFalse(parset_dict["download_initial_skymodel"])
            self.assertTrue(any("Sky model download requested" in msg for msg in cm.output))
            self.assertTrue(any("Disabling download" in msg for msg in cm.output))

    # ---- no input_skymodel, generate requested ----

    def test_no_input_generate_requested(self):
        parset_dict = self._make_parset_dict(generate_initial_skymodel=True)
        with self.assertLogs(logger="rapthor:parset", level="INFO") as cm:
            check_and_adjust_skymodel_settings(parset_dict)
        self.assertTrue(any("Will automatically generate sky model" in msg for msg in cm.output))

    def test_no_input_generate_requested_with_apparent_warns(self):
        parset_dict = self._make_parset_dict(
            generate_initial_skymodel=True,
            apparent_skymodel="some_apparent.skymodel",
        )
        with self.assertLogs(logger="rapthor:parset", level="WARN") as cm:
            check_and_adjust_skymodel_settings(parset_dict)
        self.assertTrue(any("apparent sky model will not be used" in msg for msg in cm.output))

    # ---- no input_skymodel, download requested ----

    def test_no_input_download_requested(self):
        parset_dict = self._make_parset_dict(download_initial_skymodel=True)
        with self.assertLogs(logger="rapthor:parset", level="INFO") as cm:
            check_and_adjust_skymodel_settings(parset_dict)
        self.assertTrue(any("Will automatically download sky model" in msg for msg in cm.output))

    def test_no_input_download_requested_with_apparent_warns(self):
        parset_dict = self._make_parset_dict(
            download_initial_skymodel=True,
            apparent_skymodel="some_apparent.skymodel",
        )
        with self.assertLogs(logger="rapthor:parset", level="WARN") as cm:
            check_and_adjust_skymodel_settings(parset_dict)
        self.assertTrue(any("apparent sky model will not be used" in msg for msg in cm.output))

    # ---- no input_skymodel, neither generate nor download ----

    def test_no_input_no_generate_no_download_warns(self):
        parset_dict = self._make_parset_dict()
        with self.assertLogs(logger="rapthor:parset", level="WARN") as cm:
            check_and_adjust_skymodel_settings(parset_dict)
        self.assertTrue(any("neither generation nor download" in msg for msg in cm.output))

    # ---- internet access checks ----

    def test_download_no_internet_raises(self):
        parset_dict = self._make_parset_dict(
            download_initial_skymodel=True,
            cluster_specific={"allow_internet_access": False},
        )
        with pytest.raises(ValueError):
            check_and_adjust_skymodel_settings(parset_dict)

    def test_internet_allowed_download_ok(self):
        parset_dict = self._make_parset_dict(
            download_initial_skymodel=True,
            cluster_specific={"allow_internet_access": True},
        )
        # Should not raise
        with self.assertLogs(logger="rapthor:parset", level="INFO"):
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
        with pytest.raises(FileNotFoundError):
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
        with pytest.raises(FileNotFoundError):
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
        with pytest.raises(FileNotFoundError):
            check_and_adjust_skymodel_settings(parset_dict)

    def test_astrometry_skymodel_exists_no_internet_ok(self):
        with tempfile.NamedTemporaryFile(suffix=".skymodel") as f:
            parset_dict = self._make_parset_dict(
                cluster_specific={"allow_internet_access": False},
                imaging_specific={
                    "astrometry_skymodel": f.name,
                    "photometry_skymodel": None,
                },
            )
            # Should not raise (warning about no skymodel is expected)
            with self.assertLogs(logger="rapthor:parset", level="WARN") as cm:
                check_and_adjust_skymodel_settings(parset_dict)

            self.assertTrue(any("The photometry check will be skipped" in msg for msg in cm.output))

    def test_photometry_skymodel_exists_no_internet_ok(self):
        with tempfile.NamedTemporaryFile(suffix=".skymodel") as f:
            parset_dict = self._make_parset_dict(
                cluster_specific={"allow_internet_access": False},
                imaging_specific={
                    "astrometry_skymodel": None,
                    "photometry_skymodel": f.name,
                },
            )
            # Should not raise (warning about no skymodel is expected)
            with self.assertLogs(logger="rapthor:parset", level="WARN") as cm:
                check_and_adjust_skymodel_settings(parset_dict)

            self.assertTrue(any("The astrometry check will be skipped" in msg for msg in cm.output))

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
            with self.assertLogs(logger="rapthor:parset", level="WARN") as cm:
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
            with pytest.raises(ValueError):
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
            with pytest.raises(FileNotFoundError):
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
            with pytest.raises(ValueError):
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
            with pytest.raises(ValueError):
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
            with pytest.raises(ValueError):
                check_and_adjust_skymodel_settings(parset_dict)

    def test_diagnostic_skymodel_empty_no_internet_ok(self):
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
        with self.assertLogs(logger="rapthor:parset", level="WARN") as cm:
            check_and_adjust_skymodel_settings(parset_dict)

        self.assertTrue(any("The astrometry check will be skipped" in msg for msg in cm.output))
        self.assertTrue(any("The photometry check will be skipped" in msg for msg in cm.output))


if __name__ == "__main__":
    unittest.main()

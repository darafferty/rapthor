"""
This module contains tests for the module `rapthor.lib.parset`
"""

import ast
import configparser
import contextlib
import logging
import re
from collections.abc import MutableMapping, Sequence
from pathlib import Path

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
def assert_logged(caplog, logger, level, expected_messages=(), *, expected_message=None):
    """
    Context manager for asserting the presence of specific messages in the
    captured log records.

    Since this function is decorated as a context manager, the underlying
    object created is a ContextDecorator so this function can be used as a
    as decorators as well as in with statements. It is also re-entrant, so
    the same context manager

    Parameters
    ----------
    caplog : pytest.LogCaptureFixture
        The pytest caplog fixture for capturing log records.
    logger : str
        Logger name
    level : int
        Logging level
    expected_messages : str or tuple of str, optional
        Sequence of expected messages, by default ()

    Other Parameters
    ----------------
    expected_message : str, optional
        Keyword-only parameter to support user passing a single expected
        message, by default None

    Examples
    --------
    The following example will pass, because the expected message is present in
    the logs:

    >>> with assert_logged(caplog, "my_logger", logging.INFO, expected_message="Hello World"):
    ...    logging.getLogger("my_logger").info('Hello World')

    Failure case:
    >>> with assert_logged(caplog, "my_logger", logging.INFO, expected_message="Hello World"):
    ...    logging.getLogger("my_logger").info('Nope')

    Checking for multiple messages:
    >>> with assert_logged(caplog, "my_logger", logging.INFO, expected_messages=["Hello", "World"]):
    ...    logger = logging.getLogger("my_logger")
    ...    logger.info('Hello')
    ...    logger.info('World')

    Raises
    ------
    TypeError
        If the expected_messages parameter is not a string or a sequence of strings.
    AssertionError
        If any of the expected messages are not found in the logs.
    """

    # validate inputs
    if isinstance(expected_messages, str):
        expected_messages = [expected_messages]

    if not isinstance(expected_messages, Sequence):
        raise TypeError("expected_messages parameter should be a string or a sequence of strings")

    if expected_message is not None:
        expected_messages = [*expected_messages, expected_message]

    if not expected_messages:
        raise ValueError("expected messages list is empty")

    for msg in expected_messages:
        if not isinstance(msg, str):
            raise TypeError(f"expected message must be a str, not {type(msg).__name__}")

    # Capture logs at the level for the given logger
    with caplog.at_level(level, logger=logger) as context:
        yield context

        for expected_message in expected_messages:
            if expected_message not in caplog.text:
                raise AssertionError(
                    f"Expected message(s) not found in logs:\n\t{expected_message!r}"
                )


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


@pytest.fixture
def dynamic_fixture_lookup(request):
    """
    Helper fixture that retrieves fixture values dynamically for test case
    parameters containing direct fixture function references.

    This enables more flexible parametrization of test cases that require a mix
    of literal values and fixture values. Fixture functions are resolved to the
    required fixture values at runtime. The requested parameters may be a
    single value or a nested structure, in which case the value resolution will
    be done recursively.

    Examples
    --------
    >>> @pytest.fixture
    ... def my_fixture():
    ...     'do some setup here'
    ...     yield "fixture value"
    ...
    ... @pytest.mark.parametrize('dynamic_fixture_lookup', [my_fixture], indirect=True)
    ... def test_something(dynamic_fixture_lookup):
    ...     assert dynamic_fixture_lookup == "fixture value"
    """
    return _get_fixture_value(request, request.param)


def _get_fixture_value(request, item):
    # If the item is a fixture function, resolve it to the fixture value
    if callable(item) and hasattr(item, "_pytestfixturefunction"):
        return request.getfixturevalue(item.__name__)

    if isinstance(item, MutableMapping):
        return {key: _get_fixture_value(request, subitem) for key, subitem in item.items()}

    if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
        return [_get_fixture_value(request, subitem) for subitem in item]

    return item


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
            ValueError, match=re.escape("Missing required option(s) in section [global]:")
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
            pytest.param(
                ("cluster", "dir_local", "some value"),
                "Option 'dir_local' in section [cluster] is deprecated",
                id="deprecated_option",
            ),
        ],
        indirect=["parset"],
    )
    def test_misconfigured(self, parset, caplog, expected_message):
        """
        Test that invalid sections or options in the parset produce appropriate
        warning log messages.
        """
        with assert_warning_logged(caplog, expected_message):
            parset_read(parset)

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


class TestCheckSkymodelSettings:
    """
    Tests for the `check_and_adjust_skymodel_settings` function.
    """

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
        "dynamic_fixture_lookup, context",
        [
            # Test nominal case where input skymodel is provided and exists.
            (mock_skymodel_path, contextlib.nullcontext()),
            # Test that a non-existent input skymodel raises FileNotFoundError.
            ("/nonexistent/skymodel.txt", pytest.raises(FileNotFoundError)),
        ],
        indirect=["dynamic_fixture_lookup"],
    )
    def test_input_skymodel_not_found_raises(self, dynamic_fixture_lookup, context):
        """
        Test that providing an existing input skymodel works, and that a
        non-existent input skymodel raises FileNotFoundError.
        """
        parset_dict = self._make_parset_dict(input_skymodel=dynamic_fixture_lookup)
        with context:
            check_and_adjust_skymodel_settings(parset_dict)

    @pytest.mark.parametrize(
        "generate, expected_warning",
        [
            (True, "Sky model generation requested"),
            (False, ["Sky model download requested", "Disabling download"]),
        ],
    )
    def test_input_skymodel_disables_download(
        self, caplog, mock_skymodel_path, generate, expected_warning
    ):
        """
        Test that download is disabled when an input skymodel is provided.
        """
        parset_dict = self._make_parset_dict(
            input_skymodel=mock_skymodel_path,
            generate_initial_skymodel=generate,
            download_initial_skymodel=True,
        )
        with assert_warning_logged(caplog, expected_warning):
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
        with assert_info_logged(caplog, expected_message=expected_warning):
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
        [
            "astrometry",
            "photometry",
        ],
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
            expected_message=f"The {other} check will be skipped",
        ):
            check_and_adjust_skymodel_settings(parset_dict)

    def test_normalization_skymodel_exists_no_internet_ok(self, mock_skymodel_path):
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
            expected_messages=[
                "The astrometry check will be skipped",
                "The photometry check will be skipped",
            ],
        ):
            check_and_adjust_skymodel_settings(parset_dict)

    @pytest.mark.parametrize(
        "dynamic_fixture_lookup, context",
        [
            # Nominal case: skymodels and frequencies can be given as tuples
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
                    "normalization_skymodels": [mock_skymodel_path],
                    "normalization_reference_frequencies": (142000000.0, 142001000.0),
                },
                pytest.raises(ValueError),
                id="normalization_parameters_set_raises_error",
            ),
        ],
        indirect=["dynamic_fixture_lookup"],
    )
    def test_normalization_skymodel_input(self, dynamic_fixture_lookup, context):
        """
        Test validation of normalization skymodel and reference frequency inputs.
        """
        parset_dict = self._make_parset_dict(
            cluster_specific={"allow_internet_access": False},
            imaging_specific=dynamic_fixture_lookup,
        )
        with context:
            check_and_adjust_skymodel_settings(parset_dict)

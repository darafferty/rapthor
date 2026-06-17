"""
This module contains helper code for testing rapthor.
"""

import configparser
import contextlib
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT_DIR = Path(__file__).parent.parent


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


def _get_test_run_root():
    """Keep CI integration runs inside the project so GitLab can upload logs."""
    if ci_project_dir := os.environ.get("CI_PROJECT_DIR"):
        # Keep the path short enough for multiprocessing AF_UNIX socket names.
        run_root = Path(ci_project_dir) / "ci" / "i"
        run_root.mkdir(parents=True, exist_ok=True)
        return run_root
    return Path("/tmp")


def generate_parset(
    template_parset_path,
    input_ms,
    input_skymodel_path=None,
    apparent_skymodel_path=None,
    normalization_skymodel_paths=None,
):
    """
    Generate a complete parset from a template, optionally update the input
    skymodel paths and return the parset as a configparser.ConfigParser object.

    This function creates a temporary working folder and scratch folder, and
    updates the provided template parset with
     - `dir_working` to a temporary directory
     - `local_scratch_dir` to a temporary directory
     - `global_scratch_dir` to a temporary directory

    If either skymodel is provided, the following keys in the parset will be
    updated:
     - `input_skymodel` in [global] to the provided sky model path
     - `apparent_skymodel` in [global] to the provided sky model path
     - `photometry_skymodel` in [imaging] to the provided sky model path
     - `astrometry_skymodel` in [imaging] to the provided sky model path

    Parameters
    ----------
    template_parset_path : str
        Path to the template parset file.
    input_ms : str
        Path to the input measurement set to set in the parset.
    input_skymodel_path : str, optional (default=None)
        Path to the input skymodel file to set in the parset.
    apparent_skymodel_path : str, optional (default=None)
        Path to the apparent skymodel file to set in the parset.
    normalization_skymodel_paths : list of str, optional (default=None)
        List of paths to the normalization skymodel files to set in the parset.

    Returns
    -------
    configparser.ConfigParser
        The updated parset as a ConfigParser object.
    """
    parset_path = REPO_ROOT_DIR / template_parset_path
    if input_skymodel_path:
        input_skymodel_path = REPO_ROOT_DIR / input_skymodel_path
    if apparent_skymodel_path:
        apparent_skymodel_path = REPO_ROOT_DIR / apparent_skymodel_path
    if normalization_skymodel_paths:
        normalization_skymodel_paths = [
            REPO_ROOT_DIR / path for path in normalization_skymodel_paths
        ]

    # Keep runtime paths short to avoid AF_UNIX socket path length limits
    # in multiprocessing-based tooling (e.g. PyBDSF). In CI, place runs under
    # the project directory so the generated logs can be collected as artifacts.
    run_dir = Path(tempfile.mkdtemp(prefix="ical-", dir=_get_test_run_root()))
    work_dir = run_dir / "work"
    scratch_dir = run_dir / "scratch"
    work_dir.mkdir()
    scratch_dir.mkdir()

    parset = configparser.ConfigParser()
    parset.read(parset_path)
    parset["global"].update(
        dir_working=str(work_dir),
        input_ms=str(input_ms),
    )
    if input_skymodel_path:
        parset["global"]["input_skymodel"] = str(input_skymodel_path)
        parset["imaging"]["photometry_skymodel"] = str(input_skymodel_path)
        parset["imaging"]["astrometry_skymodel"] = str(input_skymodel_path)
    if apparent_skymodel_path:
        parset["global"]["apparent_skymodel"] = str(apparent_skymodel_path)
    if normalization_skymodel_paths:
        parset["imaging"]["normalization_skymodels"] = (
            "["
            + ", ".join([str(path) for path in normalization_skymodel_paths if path is not None])
            + "]"
        )
        parset["imaging"]["normalization_reference_frequencies"] = (
            "["
            + ", ".join(
                [
                    str(120000000.0 + i * 60000000.0)
                    for i, _ in enumerate(normalization_skymodel_paths)
                    if _ is not None
                ]
            )
            + "]"
        )
    else:
        parset["imaging"]["normalization_reference_frequencies"] = "None"
    parset["cluster"].update(
        local_scratch_dir=str(scratch_dir),
        global_scratch_dir=str(scratch_dir),
    )
    return parset


def generate_parset_path(
    template_path,
    output_path,
    test_ms,
    input_skymodel_path,
    apparent_skymodel_path,
    normalization_skymodel_paths=None,
):
    """
    Fixture to generate a complete parset from a template and return the path.

    This fixture is used to read in and update a template parset file. It is
    parametrised using the pytest request fixture and expects a tuple
    containing three paths to the following files:

    1. Template parset (e.g. in tests/resources/parsets/)
    2. True sky model (e.g. in tests/resources/)
    3. Apparent sky model (e.g. in tests/resources/)

    This fixture can be used to test rapthor runs end to end on a small input
    measurement set with different strategies and sky models.
    For further details see `generate_parset` function.
    """
    parset_path = REPO_ROOT_DIR / template_path
    parset = generate_parset(
        parset_path,
        test_ms,
        input_skymodel_path,
        apparent_skymodel_path,
        normalization_skymodel_paths,
    )

    with output_path.open("w") as fp:
        parset.write(fp)

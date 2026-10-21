"""
Module that modifies the state of the pipeline
"""

import filecmp
import logging
import os
import re
import shutil
import sys
from contextlib import contextmanager

from rapthor.lib.parset import parset_read

log = logging.getLogger("rapthor:state")

# Operations in the order in which they are listed within a cycle
OPERATION_ORDER = (
    "concatenate",
    "initial_image",
    "predict_nc",
    "calibrate",
    "predict_di",
    "calibrate_di",
    "predict",
    "normalize",
    "image",
    "mosaic",
)
# The DI and DD calibration operations of a cycle run in the order set by its
# calibration strategy and use each other's solutions, so they are reset together
CALIBRATION_OPERATIONS = {"predict_nc", "calibrate", "predict_di", "calibrate_di"}
_OPERATION_DIRECTORY = re.compile(r"(?P<operation>[a-z_]+?)(?:_(?P<index>[0-9]+))?")


@contextmanager
def _suppressed_loggers(*names, level=logging.ERROR):
    """Raise the level of the named loggers and restore it afterwards.

    Loggers are process-wide, so the previous levels are restored even when the
    block exits through ``sys.exit``. Otherwise the suppression would leak into
    the rest of the process, for example into tests that run afterwards.
    """
    loggers = [logging.getLogger(name) for name in names]
    previous_levels = [logger.level for logger in loggers]
    for logger in loggers:
        logger.setLevel(level)
    try:
        yield
    finally:
        for logger, previous_level in zip(loggers, previous_levels):
            logger.setLevel(previous_level)


def run(parset_file):
    """
    Modifies the state of one or more operations

    Parameters
    ----------
    parset_file : str
        Filename of parset containing processing parameters
    """
    # The parset messages were already shown when the run started, so suppress
    # them while the state is read and modified.
    with _suppressed_loggers("rapthor:parset"):
        _run(parset_file)


def _parse_operation(name):
    """Return the cycle index and operation of an operation directory, or None."""
    match = _OPERATION_DIRECTORY.fullmatch(name)
    if match is None or match["operation"] not in OPERATION_ORDER:
        return None
    # Only the initial sky-model image, made before the first cycle, has no index
    if (match["index"] is None) != (match["operation"] == "initial_image"):
        return None
    return int(match["index"] or 1), match["operation"]


def find_operations(pipelines_dir):
    """
    Return the operation directories in pipelines_dir, ordered by cycle.

    The list comes from the directories rather than from the current strategy,
    so that it includes cycles that the current settings would no longer run,
    such as repeated final cycles.
    """
    if not os.path.isdir(pipelines_dir):
        return []
    names = os.listdir(pipelines_dir)
    operations = {name: _parse_operation(name) for name in names}
    return sorted(
        (name for name, parsed in operations.items() if parsed),
        key=lambda name: (operations[name][0], OPERATION_ORDER.index(operations[name][1]), name),
    )


def _reset_start(pipelines, selected):
    """Return the position from which to reset when an operation is selected."""
    index, operation = _parse_operation(pipelines[selected])
    if operation not in CALIBRATION_OPERATIONS:
        return selected
    calibration = {(index, name) for name in CALIBRATION_OPERATIONS}
    return next(
        position for position, name in enumerate(pipelines) if _parse_operation(name) in calibration
    )


def _run(parset_file):
    # Read parset
    log.info("Reading parset and checking state...")
    parset = parset_read(parset_file, use_log_file=False)
    pipelines_dir = os.path.join(parset["dir_working"], "pipelines")

    while True:
        # Check for started operations (workflows)
        pipelines = find_operations(pipelines_dir)

        # List operations and query user
        print(f"\nCurrent strategy: {parset['strategy']}")
        print("\nOperations:")
        i = 0
        if len(pipelines) == 0:
            print("    None")
            print("No reset can be done.")
            sys.exit(0)
        else:
            for p in pipelines:
                i += 1
                print(f"    {i}) {p}")
        try:
            while True:
                p_number_raw = input('Enter number of operation to reset or "q" to quit: ')
                try:
                    if p_number_raw.lower() == "q":
                        sys.exit(0)
                    elif int(p_number_raw) > 0 and int(p_number_raw) <= i:
                        break
                    else:
                        print(f"Please enter a number between 1 and {i}")
                except ValueError:
                    pass
        except KeyboardInterrupt:
            sys.exit(0)
        start = _reset_start(pipelines, int(p_number_raw) - 1)
        pipeline = pipelines[start]
        if start != int(p_number_raw) - 1:
            print(
                "The calibration operations of a cycle are reset together, as they "
                "depend on each other's solutions."
            )

        # Ask for confirmation
        try:
            while True:
                answer = input(f"Reset all operations from {pipeline} onwards (y/n)? ")
                if (
                    answer.lower() == "n"
                    or answer.lower() == "no"
                    or answer.lower() == "y"
                    or answer.lower() == "yes"
                ):
                    break
                else:
                    print('Please enter "y" or "n"')
        except KeyboardInterrupt:
            sys.exit(0)

        # Reset operation states as requested
        if answer.lower() == "y" or answer.lower() == "yes":
            print("Reseting state...")
            for pipeline in pipelines[start:]:
                # Remove the operation working directory to ensure files from previous
                # runs are not kept and used in subsequent ones. Other associated
                # files are removed as well.
                path = os.path.join(pipelines_dir, pipeline)
                shutil.rmtree(path, ignore_errors=True)

            # Now remove any sub-directories in the other output directories, that
            # are _not_ present in the 'pipelines' directory.
            for dirname in (
                "skymodels",
                "solutions",
                "logs",
                "plots",
                "regions",
                "images",
                "visibilities",
            ):
                output_dir = os.path.join(parset["dir_working"], dirname)
                if not os.path.exists(output_dir):
                    continue
                dcmp = filecmp.dircmp(
                    os.path.join(parset["dir_working"], "pipelines"),
                    output_dir,
                )
                for path in (os.path.join(dcmp.right, item) for item in dcmp.right_only):
                    shutil.rmtree(path, ignore_errors=True)

            print("Reset complete.")

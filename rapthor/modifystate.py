"""
Module that modifies the state of the pipeline
"""

import filecmp
import glob
import json
import logging
import os
import re
import shutil
import sys
import tempfile
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone

from rapthor.execution.working_dir_lock import WorkingDirectoryLockError, working_directory_lock
from rapthor.execution.workspace import remove_operation_workspace
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
# The DD (calibrate) and DI (predict_di, calibrate_di) calibration operations of a
# cycle run in the order set by the calibration strategy, and the one that runs
# second uses the solutions of the first. The list of operations has a fixed order
# and the strategy is not read, so selecting any of them resets all of them.
CALIBRATION_OPERATIONS = {"predict_nc", "calibrate", "predict_di", "calibrate_di"}
# Logs with one JSON record per line, each naming its operation
OPERATION_RECORD_LOGS = ("commands.jsonl", "tasks.jsonl")
MAIN_LOG = "rapthor.log"
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
    # An operation stopped while moving to or from shared scratch may only have
    # its recovery record, ".<operation>.scratch"
    names = {re.sub(r"^\.(.+)\.scratch$", r"\1", name) for name in os.listdir(pipelines_dir)}
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


def archive_reset_logs(logs_dir, operations, now=None):
    """
    Move the log records of reset operations to files named after the reset.

    The records are moved from commands.jsonl and tasks.jsonl to, for example,
    commands.reset-20250102T030405Z.jsonl, so that the logs only describe the work
    whose products are kept. rapthor.log cannot be split by operation, so it is
    renamed in the same way. Returns the name of the reset attempt, or None if
    nothing was archived.
    """
    now = now or datetime.now(timezone.utc)
    stamp = attempt = f"reset-{now:%Y%m%dT%H%M%SZ}"
    number = 1
    # Two resets within the same second get distinct names
    while any(
        os.path.exists(_archive_path(logs_dir, filename, attempt))
        for filename in (*OPERATION_RECORD_LOGS, MAIN_LOG)
    ):
        number += 1
        attempt = f"{stamp}-{number}"

    archived_any = False
    for filename in OPERATION_RECORD_LOGS:
        path = os.path.join(logs_dir, filename)
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as log_file:
            lines = log_file.read().splitlines()
        kept, moved = [], []
        for line in lines:
            (moved if _record_operation(line) in operations else kept).append(line)
        if not moved:
            continue
        # A reset that stopped after writing an archive but before replacing the
        # log left its records in both, so archive each record once
        archived = Counter()
        for archive in glob.glob(_archive_path(glob.escape(logs_dir), filename, "reset-*")):
            with open(archive, encoding="utf-8") as archive_file:
                archived.update(archive_file.read().splitlines())
        moved = list((Counter(moved) - archived).elements())
        if moved:
            with open(_archive_path(logs_dir, filename, attempt), "a", encoding="utf-8") as archive:
                archive.writelines(f"{line}\n" for line in moved)
            archived_any = True
        # Replace the log in one step, so it is never left partly written
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=logs_dir, prefix=f".{filename}.", delete=False
        ) as replacement:
            replacement.writelines(f"{line}\n" for line in kept)
        try:
            shutil.copymode(path, replacement.name)
            os.replace(replacement.name, path)
        except BaseException:
            os.remove(replacement.name)
            raise

    main_log = os.path.join(logs_dir, MAIN_LOG)
    if os.path.isfile(main_log):
        os.replace(main_log, _archive_path(logs_dir, MAIN_LOG, attempt))
        archived_any = True
    return attempt if archived_any else None


def _archive_path(logs_dir, filename, attempt):
    stem, extension = os.path.splitext(filename)
    return os.path.join(logs_dir, f"{stem}.{attempt}{extension}")


def _record_operation(line):
    try:
        record = json.loads(line)
    except ValueError:
        return None
    return record.get("operation") if isinstance(record, dict) else None


def _run(parset_file):
    # Read parset
    log.info("Reading parset and checking state...")
    parset = parset_read(parset_file, use_log_file=False)

    # Hold the working directory while operations are listed and removed
    try:
        with working_directory_lock(parset["dir_working"], "a Rapthor reset", required=True):
            _reset(parset)
    except WorkingDirectoryLockError as error:
        print(f"{error}\nNo reset can be done.")
        sys.exit(1)


def _reset(parset):
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
                "The calibration operations of a cycle are reset together: the one that "
                "ran second used the solutions of the first, in the order set by the "
                "calibration strategy."
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
            # Move the logs of the reset work first, so that a reset that stops part
            # way still lists the operations and can be repeated
            logs_dir = os.path.join(parset["dir_working"], "logs")
            if os.path.isdir(logs_dir):
                attempt = archive_reset_logs(logs_dir, pipelines[start:])
                if attempt is not None:
                    print(f"Logs of the reset operations were moved to logs/*.{attempt}.*")

            for pipeline in pipelines[start:]:
                # Remove the operation working directory to ensure files from previous
                # runs are not kept and used in subsequent ones. Its shared scratch
                # workspace and recovery record are removed as well.
                remove_operation_workspace(os.path.join(pipelines_dir, pipeline))

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

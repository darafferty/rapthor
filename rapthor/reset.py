"""Reset operations of a Rapthor run."""

import filecmp
import json
import logging
import re
import shutil
import sys
import tempfile
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from rapthor.execution.working_dir_lock import WorkingDirectoryLockError, working_directory_lock
from rapthor.execution.workspace import remove_operation_workspace
from rapthor.lib.parset import Parset

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


def _parse_operation(name):
    """
    Extract the operation and cycle index from an operation directory name.

    Returns (cycle index, operation) or None if they were not found.
    """
    match = _OPERATION_DIRECTORY.fullmatch(name)
    if match is None or match["operation"] not in OPERATION_ORDER:
        return None
    index, operation = match["index"], match["operation"]
    if operation == "initial_image":
        # The initial sky-model image is made before the first cycle and has no index
        return (1, operation) if index is None else None
    if index is None:
        return None
    return int(index), operation


def find_operations(pipelines_dir):
    """
    Return the operation directories in pipelines_dir, ordered by cycle.

    The list comes from the directories rather than from the current strategy,
    so that it includes cycles that the current settings would no longer run,
    such as repeated final cycles.
    """
    pipelines_dir = Path(pipelines_dir)
    if not pipelines_dir.is_dir():
        return []
    # An operation stopped while moving to or from shared scratch may only have
    # its recovery record, ".<operation>.scratch"
    names = {re.sub(r"^\.(.+)\.scratch$", r"\1", path.name) for path in pipelines_dir.iterdir()}
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
    # The first calibration operation of the same cycle
    for position, name in enumerate(pipelines):
        operation_index, operation_name = _parse_operation(name)
        if operation_index == index and operation_name in CALIBRATION_OPERATIONS:
            return position


def archive_reset_logs(logs_dir, operations, now=None):
    """
    Move the log records of reset operations to files named after the reset.

    commands.jsonl and tasks.jsonl hold one record per line for the whole run,
    each naming its operation. The records of the reset operations are moved to,
    for example, commands.reset-20250102T030405Z.jsonl and the others stay, so
    that the logs only describe the work whose products are kept. rapthor.log has
    no record per operation and cannot be split, so it is renamed in the same way.
    Returns the name of the reset attempt, or None if nothing was archived.
    """
    logs_dir = Path(logs_dir)
    now = now or datetime.now(timezone.utc)
    # The Z marks the time as UTC. A second reset within the same second is named
    # after the next second, so that it gets its own files.
    while True:
        attempt = f"reset-{now:%Y%m%dT%H%M%SZ}"
        if not any(
            _archive_path(logs_dir, filename, attempt).exists()
            for filename in (*OPERATION_RECORD_LOGS, MAIN_LOG)
        ):
            break
        now += timedelta(seconds=1)

    archived_any = False
    for filename in OPERATION_RECORD_LOGS:
        path = logs_dir / filename
        if path.is_file():
            archive_path = _archive_path(logs_dir, filename, attempt)
            archived_any |= _move_records(path, archive_path, operations)
    main_log = logs_dir / MAIN_LOG
    if main_log.is_file():
        main_log.replace(_archive_path(logs_dir, MAIN_LOG, attempt))
        archived_any = True
    return attempt if archived_any else None


def _move_records(path, archive_path, operations):
    """
    Move the records of the given operations from a record log to an archive.

    Returns whether any record was archived. A reset that stopped after writing
    its archive but before replacing the log left its records in both, so a record
    that an earlier archive of this log already holds is removed from the log
    without being archived again.
    """
    kept, moved = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        (moved if _record_operation(line) in operations else kept).append(line)
    if not moved:
        return False
    archived = Counter()
    for archive in path.parent.glob(f"{path.stem}.reset-*{path.suffix}"):
        archived.update(archive.read_text(encoding="utf-8").splitlines())
    moved = list((Counter(moved) - archived).elements())
    if moved:
        with archive_path.open("a", encoding="utf-8") as archive:
            archive.writelines(f"{line}\n" for line in moved)
    # Replace the log in one step, so it is never left partly written
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as replacement:
        replacement.writelines(f"{line}\n" for line in kept)
    try:
        shutil.copymode(path, replacement.name)
        Path(replacement.name).replace(path)
    except BaseException:
        Path(replacement.name).unlink()
        raise
    return bool(moved)


def _archive_path(logs_dir, filename, attempt):
    path = Path(logs_dir, filename)
    return path.with_stem(f"{path.stem}.{attempt}")


def _record_operation(line):
    try:
        record = json.loads(line)
    except ValueError:
        return None
    return record.get("operation") if isinstance(record, dict) else None


def reset(parset_file):
    """
    Reset operations of a run interactively.

    Lists the operations started in the working directory, asks which one to
    reset, and removes it and the operations after it together with their
    products, so that the next run makes them again.

    Parameters
    ----------
    parset_file : str
        Filename of parset containing processing parameters
    """
    # The parset messages were already shown when the run started, so suppress
    # them while the state is read and modified. A reset needs the settings only,
    # not the input data or the log file.
    log.info("Reading parset and checking state...")
    with _suppressed_loggers("rapthor:parset"):
        parset = Parset(parset_file).as_parset_dict()
        # Hold the working directory while operations are listed and removed
        try:
            with working_directory_lock(parset["dir_working"], "a Rapthor reset", required=True):
                _reset(parset)
        except WorkingDirectoryLockError as error:
            print(f"{error}\nNo reset can be done.")
            sys.exit(1)


def _reset(parset):
    working_dir = Path(parset["dir_working"])
    pipelines_dir = working_dir / "pipelines"

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
            logs_dir = working_dir / "logs"
            if logs_dir.is_dir():
                attempt = archive_reset_logs(logs_dir, pipelines[start:])
                if attempt is not None:
                    print(f"Logs of the reset operations were moved to logs/*.{attempt}.*")

            for pipeline in pipelines[start:]:
                # Remove the operation working directory to ensure files from previous
                # runs are not kept and used in subsequent ones. Its shared scratch
                # workspace and recovery record are removed as well.
                remove_operation_workspace(pipelines_dir / pipeline)

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
                output_dir = working_dir / dirname
                if not output_dir.exists():
                    continue
                dcmp = filecmp.dircmp(pipelines_dir, output_dir)
                for item in dcmp.right_only:
                    shutil.rmtree(output_dir / item, ignore_errors=True)

            print("Reset complete.")

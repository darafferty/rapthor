"""
Test cases for the `rapthor.reset` module.
"""

import errno
import json
import logging
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

import rapthor.reset as reset_module
from rapthor.execution import working_dir_lock
from tests.execution.scratch_transitions import exit_at_scratch_transition

OUTPUT_DIRS = (
    "skymodels",
    "solutions",
    "logs",
    "plots",
    "regions",
    "images",
    "visibilities",
)


def _prepare_working_dir(tmp_path, omitted_output_dirs=()):
    working_dir = tmp_path / "working"
    (working_dir / "pipelines").mkdir(parents=True)
    for dirname in OUTPUT_DIRS:
        if dirname in omitted_output_dirs:
            continue
        (working_dir / dirname).mkdir()
    return working_dir


def _write_prefect_operation_state(working_dir, operation_name):
    operation_dir = working_dir / "pipelines" / operation_name
    operation_dir.mkdir(parents=True)
    (operation_dir / ".done").touch()
    (operation_dir / "pipeline_outputs.json").write_text(
        json.dumps(
            {
                "product": {
                    "class": "File",
                    "path": str(operation_dir / f"{operation_name}.txt"),
                }
            }
        )
    )
    (operation_dir / "pipeline_inputs.json").write_text("{}")
    return operation_dir


def _write_output_subdir(working_dir, dirname, operation_name):
    output_dir = working_dir / dirname / operation_name
    output_dir.mkdir(parents=True)
    (output_dir / "product.txt").write_text("product")
    return output_dir


def _patch_reset_lifecycle(monkeypatch, working_dir):
    parset = {"dir_working": str(working_dir), "strategy": "test-strategy"}

    monkeypatch.setattr(
        reset_module, "Parset", lambda parset_file: SimpleNamespace(as_parset_dict=lambda: parset)
    )


@pytest.fixture
def working_dir(tmp_path, monkeypatch):
    working_dir = _prepare_working_dir(tmp_path)
    _patch_reset_lifecycle(monkeypatch, working_dir)
    return working_dir


def _reset(monkeypatch, *answers):
    """Run a reset with the given answers to its prompts and return its exit code."""
    remaining = iter(answers)
    monkeypatch.setattr("builtins.input", lambda prompt: next(remaining))
    with pytest.raises(SystemExit) as exc:
        reset_module.reset("input.parset")
    return exc.value.code


def _write_jsonl(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_reset_resets_from_selected_operation_and_preserves_upstream_prefect_state(
    tmp_path, monkeypatch, capsys
):
    working_dir = _prepare_working_dir(tmp_path)
    _patch_reset_lifecycle(monkeypatch, working_dir)

    upstream_calibrate = _write_prefect_operation_state(working_dir, "calibrate_1")
    upstream_predict = _write_prefect_operation_state(working_dir, "predict_1")
    downstream_image = _write_prefect_operation_state(working_dir, "image_1")
    downstream_mosaic = _write_prefect_operation_state(working_dir, "mosaic_1")

    upstream_solutions = _write_output_subdir(working_dir, "solutions", "calibrate_1")
    upstream_visibilities = _write_output_subdir(working_dir, "visibilities", "predict_1")
    downstream_images = _write_output_subdir(working_dir, "images", "image_1")
    downstream_logs = _write_output_subdir(working_dir, "logs", "mosaic_1")

    assert _reset(monkeypatch, ["3", "y", "q"]) == 0
    assert "Reset complete." in capsys.readouterr().out

    assert upstream_calibrate.is_dir()
    assert (upstream_calibrate / ".done").is_file()
    upstream_outputs = json.loads((upstream_calibrate / "pipeline_outputs.json").read_text())
    assert upstream_outputs["product"]["class"] == "File"
    assert upstream_predict.is_dir()
    assert (upstream_predict / ".done").is_file()
    assert (upstream_predict / "pipeline_outputs.json").is_file()
    assert upstream_solutions.is_dir()
    assert upstream_visibilities.is_dir()

    assert not downstream_image.exists()
    assert not downstream_mosaic.exists()
    assert not downstream_images.exists()
    assert not downstream_logs.exists()


def test_reset_restores_logger_levels_after_exit(tmp_path, monkeypatch):
    """Suppressing the parset logger must not leak out of reset()."""
    _patch_reset_lifecycle(monkeypatch, tmp_path)
    logger = logging.getLogger("rapthor:parset")
    monkeypatch.setattr(logger, "level", logging.INFO)

    # Without any pipeline directory there is nothing to reset and reset() exits.
    with pytest.raises(SystemExit):
        reset_module.reset("input.parset")

    assert logger.level == logging.INFO


def test_reset_ignores_missing_optional_output_directories(tmp_path, monkeypatch, capsys):
    working_dir = _prepare_working_dir(tmp_path, omitted_output_dirs={"plots", "visibilities"})
    _patch_reset_lifecycle(monkeypatch, working_dir)

    image_state = _write_prefect_operation_state(working_dir, "image_1")
    image_output = _write_output_subdir(working_dir, "images", "image_1")

    assert _reset(monkeypatch, ["1", "y", "q"]) == 0
    assert "Reset complete." in capsys.readouterr().out
    assert not image_state.exists()
    assert not image_output.exists()
    assert not (working_dir / "plots").exists()
    assert not (working_dir / "visibilities").exists()


def test_reset_exits_without_reset_when_no_pipeline_state_exists(tmp_path, monkeypatch, capsys):
    working_dir = _prepare_working_dir(tmp_path)
    _patch_reset_lifecycle(monkeypatch, working_dir)

    with pytest.raises(SystemExit) as exc:
        reset_module.reset("input.parset")

    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "Operations:" in output
    assert "None" in output
    assert "No reset can be done." in output


def test_find_operations_orders_directories_by_cycle_and_operation(tmp_path):
    names = ["mosaic_10", "image_2", "calibrate_10", "initial_image", "predict_di_2"]
    names += ["calibrate_2", "concatenate_1", "unrelated", "image", "initial_image_1"]
    for name in names:
        (tmp_path / name).mkdir()
    # An operation stopped while moving to or from scratch may only have this record
    (tmp_path / ".calibrate_1.scratch").mkdir()

    assert reset_module.find_operations(str(tmp_path)) == [
        "concatenate_1",
        "initial_image",
        "calibrate_1",
        "calibrate_2",
        "predict_di_2",
        "image_2",
        "calibrate_10",
        "mosaic_10",
    ]


def test_reset_resets_repeated_final_cycles_found_on_disk(working_dir, monkeypatch, capsys):
    """The list comes from the cycles on disk, not from the current strategy."""
    operations = [
        _write_prefect_operation_state(working_dir, f"{name}_{cycle}")
        for cycle in (1, 2, 3)
        for name in ("calibrate", "image")
    ]
    final_images = _write_output_subdir(working_dir, "images", "image_3")

    assert _reset(monkeypatch, "4", "y", "q") == 0

    assert "6) image_3" in capsys.readouterr().out
    assert [path.exists() for path in operations] == [True, True, True, False, False, False]
    assert not final_images.exists()


@pytest.mark.parametrize("selected", ["predict_di_2", "calibrate_di_2"])
def test_reset_resets_all_calibration_operations_of_a_cycle_together(
    working_dir, monkeypatch, capsys, selected
):
    """A DI-first cycle runs predict_di and calibrate_di before calibrate, although
    calibrate is listed first, and its DD solve uses the DI solutions."""
    names = [
        f"{operation}_{cycle}"
        for cycle in (1, 2)
        for operation in ("calibrate", "predict_di", "calibrate_di", "image")
    ]
    for name in names:
        _write_prefect_operation_state(working_dir, name)

    _reset(monkeypatch, str(names.index(selected) + 1), "y", "q")

    assert "calibration operations of a cycle are reset together" in capsys.readouterr().out
    remaining = sorted(path.name for path in (working_dir / "pipelines").iterdir())
    assert remaining == sorted(names[:4])


def test_reset_resets_operations_with_their_scratch_workspaces(tmp_path, working_dir, monkeypatch):
    pipelines_dir = working_dir / "pipelines"
    scratch = tmp_path / "scratch"
    exit_at_scratch_transition(pipelines_dir / "calibrate_1", scratch, "flow")
    kept_workspace = (pipelines_dir / "calibrate_1").resolve()
    # The image operation stopped while staging, so only its recovery record is left
    exit_at_scratch_transition(pipelines_dir / "image_1", scratch, "backup")

    assert _reset(monkeypatch, "2", "y", "q") == 0

    assert sorted(path.name for path in pipelines_dir.iterdir()) == [
        ".calibrate_1.scratch",
        "calibrate_1",
    ]
    assert list(scratch.iterdir()) == [kept_workspace.parent]


RESET_TIME = datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


def test_archive_reset_logs_moves_records_to_a_file_per_attempt(tmp_path):
    commands = [{"operation": name} for name in ("calibrate_1", "image_1", "mosaic_1")]
    _write_jsonl(tmp_path / "commands.jsonl", commands)
    with open(tmp_path / "commands.jsonl", "a") as log_file:
        log_file.write('not json\n{"operation": null}\n')
    tasks = [{"operation": "calibrate_1"}, {"operation": "image_1"}]
    _write_jsonl(tmp_path / "tasks.jsonl", tasks)
    (tmp_path / "rapthor.log").write_text("attempt 1\n")

    attempt = reset_module.archive_reset_logs(str(tmp_path), ["image_1", "mosaic_1"], RESET_TIME)

    assert attempt == "reset-20250102T030405Z"
    assert (tmp_path / "commands.jsonl").read_text() == (
        '{"operation": "calibrate_1"}\nnot json\n{"operation": null}\n'
    )
    assert _read_jsonl(tmp_path / f"commands.{attempt}.jsonl") == commands[1:]
    assert _read_jsonl(tmp_path / "tasks.jsonl") == tasks[:1]
    assert _read_jsonl(tmp_path / f"tasks.{attempt}.jsonl") == tasks[1:]
    assert (tmp_path / f"rapthor.{attempt}.log").read_text() == "attempt 1\n"
    assert not (tmp_path / "rapthor.log").exists()

    # A second reset within the same second is named after the next second
    (tmp_path / "rapthor.log").write_text("attempt 2\n")
    attempt = reset_module.archive_reset_logs(str(tmp_path), ["image_1"], RESET_TIME)
    assert attempt == "reset-20250102T030406Z"
    assert (tmp_path / f"rapthor.{attempt}.log").read_text() == "attempt 2\n"
    assert reset_module.archive_reset_logs(str(tmp_path), ["image_1"], RESET_TIME) is None


def test_archive_reset_logs_archives_each_record_once_after_an_interrupted_reset(
    tmp_path, monkeypatch
):
    commands = [{"operation": name} for name in ("calibrate_1", "image_1", "mosaic_1")]
    _write_jsonl(tmp_path / "commands.jsonl", commands)

    def fail_to_replace(source, destination):
        raise OSError("disk error")

    with monkeypatch.context() as patch:
        patch.setattr(reset_module.Path, "replace", fail_to_replace)
        with pytest.raises(OSError, match="disk error"):
            reset_module.archive_reset_logs(str(tmp_path), ["image_1", "mosaic_1"], RESET_TIME)
    assert _read_jsonl(tmp_path / "commands.jsonl") == commands

    reset_module.archive_reset_logs(str(tmp_path), ["image_1", "mosaic_1"], RESET_TIME)

    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "commands.jsonl",
        "commands.reset-20250102T030405Z.jsonl",
    ]
    assert _read_jsonl(tmp_path / "commands.reset-20250102T030405Z.jsonl") == commands[1:]
    assert _read_jsonl(tmp_path / "commands.jsonl") == commands[:1]


def test_archive_reset_logs_counts_each_archived_copy_of_a_record_once(tmp_path):
    record = '{"operation": "image_1"}\n'
    (tmp_path / "commands.reset-20250101T000000Z.jsonl").write_text(record)
    (tmp_path / "commands.jsonl").write_text(record + record)

    attempt = reset_module.archive_reset_logs(str(tmp_path), ["image_1"], RESET_TIME)

    assert (tmp_path / f"commands.{attempt}.jsonl").read_text() == record
    assert (tmp_path / "commands.jsonl").read_text() == ""


def test_reset_can_be_repeated_after_failing_part_way(working_dir, monkeypatch, capsys):
    operations = ("calibrate_1", "image_1", "mosaic_1")
    for name in operations:
        _write_prefect_operation_state(working_dir, name)
    logs_dir = working_dir / "logs"
    _write_jsonl(logs_dir / "commands.jsonl", [{"operation": name} for name in operations])
    (logs_dir / "rapthor.log").write_text("first attempt\n")
    remove_operation_workspace = reset_module.remove_operation_workspace

    def fail_on_mosaic(path):
        if Path(path).name == "mosaic_1":
            raise OSError("disk error")
        remove_operation_workspace(path)

    with monkeypatch.context() as patch:
        patch.setattr(reset_module, "remove_operation_workspace", fail_on_mosaic)
        with pytest.raises(OSError, match="disk error"):
            _reset(patch, "2", "y")
    assert "Logs of the reset operations were moved to logs/*.reset-" in capsys.readouterr().out

    # The retry still lists the operation that was not removed
    assert _reset(monkeypatch, "2", "y", "q") == 0

    assert "2) mosaic_1" in capsys.readouterr().out
    assert sorted(path.name for path in (working_dir / "pipelines").iterdir()) == ["calibrate_1"]
    assert _read_jsonl(logs_dir / "commands.jsonl") == [{"operation": "calibrate_1"}]
    (archived_commands,) = logs_dir.glob("commands.reset-*.jsonl")
    assert _read_jsonl(archived_commands) == [{"operation": "image_1"}, {"operation": "mosaic_1"}]
    assert [path.read_text() for path in logs_dir.glob("rapthor*.log")] == ["first attempt\n"]


@pytest.mark.parametrize("cause", ["held-by-a-run", "flock-unsupported"])
def test_reset_refuses_to_reset_without_the_working_directory_lock(
    working_dir, monkeypatch, capsys, cause
):
    image = _write_prefect_operation_state(working_dir, "image_1")
    if cause == "held-by-a-run":
        lock = working_dir_lock.working_directory_lock(str(working_dir), "a Rapthor test")
        expected = "is locked by a Rapthor test"
    else:
        lock = nullcontext()
        expected = "Could not lock the working directory"

        def flock(file, operation):
            raise OSError(errno.ENOLCK, "No locks available")

        monkeypatch.setattr(working_dir_lock.fcntl, "flock", flock)

    with lock:
        assert _reset(monkeypatch, "1", "y", "q") == 1

    output = capsys.readouterr().out
    assert expected in output
    assert "No reset can be done." in output
    assert image.is_dir()

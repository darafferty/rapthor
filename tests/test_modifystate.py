"""
Test cases for the `rapthor.modifystate` module.
"""

import json
from pathlib import Path

import pytest

import rapthor.modifystate as modifystate

OUTPUT_DIRS = (
    "skymodels",
    "solutions",
    "logs",
    "plots",
    "regions",
    "images",
    "visibilities",
)


class FakeField:
    def __init__(self, parset, minimal=False):
        self.parset = parset
        self.minimal = minimal


def _prepare_working_dir(tmp_path):
    working_dir = tmp_path / "working"
    (working_dir / "pipelines").mkdir(parents=True)
    for dirname in OUTPUT_DIRS:
        (working_dir / dirname).mkdir()
    return working_dir


def _write_prefect_operation_state(working_dir, operation_name):
    operation_dir = working_dir / "pipelines" / operation_name
    operation_dir.mkdir(parents=True)
    (operation_dir / ".done").touch()
    (operation_dir / ".outputs.json").write_text(
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


def _patch_modifystate_lifecycle(monkeypatch, working_dir, strategy_steps=None):
    parset = {"dir_working": str(working_dir), "strategy": "test-strategy"}

    monkeypatch.setattr(
        modifystate,
        "parset_read",
        lambda parset_file, use_log_file=False: parset,
    )
    monkeypatch.setattr(modifystate, "Field", FakeField)
    monkeypatch.setattr(modifystate, "set_strategy", lambda field: strategy_steps or [{}])


def test_run_resets_from_selected_operation_and_preserves_upstream_prefect_state(
    tmp_path, monkeypatch, capsys
):
    working_dir = _prepare_working_dir(tmp_path)
    _patch_modifystate_lifecycle(monkeypatch, working_dir)

    upstream_calibrate = _write_prefect_operation_state(working_dir, "calibrate_1")
    upstream_predict = _write_prefect_operation_state(working_dir, "predict_1")
    downstream_image = _write_prefect_operation_state(working_dir, "image_1")
    downstream_mosaic = _write_prefect_operation_state(working_dir, "mosaic_1")

    upstream_solutions = _write_output_subdir(working_dir, "solutions", "calibrate_1")
    upstream_visibilities = _write_output_subdir(working_dir, "visibilities", "predict_1")
    downstream_images = _write_output_subdir(working_dir, "images", "image_1")
    downstream_logs = _write_output_subdir(working_dir, "logs", "mosaic_1")

    answers = iter(["3", "y", "q"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))

    with pytest.raises(SystemExit) as exc:
        modifystate.run("input.parset")

    assert exc.value.code == 0
    assert "Reset complete." in capsys.readouterr().out

    assert upstream_calibrate.is_dir()
    assert (upstream_calibrate / ".done").is_file()
    upstream_outputs = json.loads((upstream_calibrate / ".outputs.json").read_text())
    assert upstream_outputs["product"]["class"] == "File"
    assert upstream_predict.is_dir()
    assert (upstream_predict / ".done").is_file()
    assert (upstream_predict / ".outputs.json").is_file()
    assert upstream_solutions.is_dir()
    assert upstream_visibilities.is_dir()

    assert not downstream_image.exists()
    assert not downstream_mosaic.exists()
    assert not downstream_images.exists()
    assert not downstream_logs.exists()


def test_run_exits_without_reset_when_no_pipeline_state_exists(tmp_path, monkeypatch, capsys):
    working_dir = _prepare_working_dir(tmp_path)
    _patch_modifystate_lifecycle(monkeypatch, working_dir)

    with pytest.raises(SystemExit) as exc:
        modifystate.run("input.parset")

    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "Operations:" in output
    assert "None" in output
    assert "No reset can be done." in output

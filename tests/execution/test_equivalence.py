import json
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from rapthor.execution.equivalence import (
    assert_backend_equivalent,
    collect_backend_summary,
    compare_backend_runs,
    compare_fits_product,
    compare_h5parm_product,
    format_differences,
    run_equivalence_pair,
)


def _write_operation_state(
    working_dir,
    operation_name="image_1",
    output_basename="final-image.fits",
    done=True,
    outputs_filename=".outputs.json",
):
    working_dir = Path(working_dir)
    operation_dir = working_dir / "pipelines" / operation_name
    operation_dir.mkdir(parents=True, exist_ok=True)
    if done:
        (operation_dir / ".done").touch()
    outputs = {
        "image": {"class": "File", "path": str(working_dir / "products" / output_basename)},
        "visibilities": [{"class": "Directory", "path": str(working_dir / "products" / "vis.ms")}],
    }
    (operation_dir / outputs_filename).write_text(json.dumps(outputs, indent=2))


def _write_backend_state(working_dir, output_basename="final-image.fits", done=True):
    working_dir = Path(working_dir)
    _write_operation_state(working_dir, "predict_1", "predict.ms", done=done)
    _write_operation_state(working_dir, "image_1", output_basename, done=done)
    (working_dir / "operation_order.json").write_text(json.dumps(["predict_1", "image_1"]))
    (working_dir / "logs").mkdir(parents=True, exist_ok=True)
    (working_dir / "logs" / "diagnostics.txt").write_text("Rapthor has finished :)\n")
    (working_dir / "field_state.json").write_text(
        json.dumps({"cycle_number": 1, "h5parm_filename": "solutions.h5"}, indent=2)
    )


def test_run_equivalence_pair_invokes_isolated_backend_runners(tmp_path):
    source_parset = tmp_path / "input.parset"
    source_parset.write_text("strategy = smoke\n")
    calls = []

    def fake_runner(parset_file, working_dir, logging_level):
        calls.append((parset_file.name, working_dir.name, logging_level))
        assert parset_file.parent == working_dir
        _write_backend_state(working_dir)
        return {"backend": working_dir.name}

    cwl_run, prefect_run = run_equivalence_pair(
        source_parset,
        tmp_path / "runs",
        cwl_runner=fake_runner,
        prefect_runner=fake_runner,
        logging_level="debug",
    )

    assert cwl_run.backend == "cwl"
    assert prefect_run.backend == "prefect"
    assert cwl_run.result == {"backend": "cwl"}
    assert prefect_run.result == {"backend": "prefect"}
    assert calls == [("input.parset", "cwl", "debug"), ("input.parset", "prefect", "debug")]
    assert cwl_run.parset_file.read_text() == source_parset.read_text()
    assert prefect_run.parset_file.read_text() == source_parset.read_text()
    assert compare_backend_runs(cwl_run.working_dir, prefect_run.working_dir) == []


def test_collect_backend_summary_normalizes_product_paths(tmp_path):
    cwl_dir = tmp_path / "cwl"
    prefect_dir = tmp_path / "prefect"
    _write_backend_state(cwl_dir)
    _write_backend_state(prefect_dir)

    cwl_summary = collect_backend_summary(cwl_dir)
    prefect_summary = collect_backend_summary(prefect_dir)

    assert cwl_summary == prefect_summary
    assert cwl_summary["operations"]["image_1"]["outputs"]["image"] == {
        "class": "File",
        "basename": "final-image.fits",
    }


def test_compare_backend_runs_reports_output_differences(tmp_path):
    cwl_dir = tmp_path / "cwl"
    prefect_dir = tmp_path / "prefect"
    _write_backend_state(cwl_dir, output_basename="final-image.fits")
    _write_backend_state(prefect_dir, output_basename="renamed-image.fits")

    differences = compare_backend_runs(cwl_dir, prefect_dir)

    assert [
        (difference.path, difference.reference, difference.candidate) for difference in differences
    ] == [
        (
            "$.operations.image_1.outputs.image.basename",
            "final-image.fits",
            "renamed-image.fits",
        )
    ]
    assert "renamed-image.fits" in format_differences(differences)


def test_assert_backend_equivalent_reports_missing_done_marker(tmp_path):
    cwl_dir = tmp_path / "cwl"
    prefect_dir = tmp_path / "prefect"
    _write_backend_state(cwl_dir, done=True)
    _write_backend_state(prefect_dir, done=False)

    with pytest.raises(AssertionError) as exc:
        assert_backend_equivalent(cwl_dir, prefect_dir)

    assert "$.operations.image_1.done" in str(exc.value)
    assert "$.operations.predict_1.done" in str(exc.value)


def test_collect_operation_state_accepts_cwl_pipeline_outputs_until_cutover(tmp_path):
    cwl_dir = tmp_path / "cwl"
    prefect_dir = tmp_path / "prefect"
    _write_operation_state(cwl_dir, outputs_filename="pipeline_outputs.json")
    _write_operation_state(prefect_dir, outputs_filename=".outputs.json")

    assert compare_backend_runs(cwl_dir, prefect_dir) == []


def test_compare_fits_product_uses_numeric_tolerances(tmp_path):
    reference_fits = tmp_path / "reference.fits"
    candidate_fits = tmp_path / "candidate.fits"
    fits.writeto(reference_fits, np.array([[1.0, 2.0], [3.0, np.nan]]), overwrite=True)
    fits.writeto(candidate_fits, np.array([[1.0, 2.0005], [3.0, np.nan]]), overwrite=True)

    assert compare_fits_product(reference_fits, candidate_fits, rtol=0.0, atol=1e-3) == []

    differences = compare_fits_product(reference_fits, candidate_fits, rtol=0.0, atol=1e-6)

    assert differences[0].path == "$.fits[0].data"
    assert differences[0].candidate["max_absolute_difference"] == pytest.approx(5e-4)


def test_compare_h5parm_product_uses_numeric_tolerances(tmp_path):
    h5py = pytest.importorskip("h5py")

    reference_h5 = tmp_path / "reference.h5"
    candidate_h5 = tmp_path / "candidate.h5"
    with h5py.File(reference_h5, "w") as h5_file:
        h5_file.create_dataset("sol000/amplitude/val", data=np.array([1.0, 2.0, 3.0]))
        h5_file.create_dataset("sol000/amplitude/label", data=b"ignored")
    with h5py.File(candidate_h5, "w") as h5_file:
        h5_file.create_dataset("sol000/amplitude/val", data=np.array([1.0, 2.0005, 3.0]))
        h5_file.create_dataset("sol000/amplitude/label", data=b"ignored")

    assert compare_h5parm_product(reference_h5, candidate_h5, rtol=0.0, atol=1e-3) == []

    differences = compare_h5parm_product(reference_h5, candidate_h5, rtol=0.0, atol=1e-6)

    assert differences[0].path == "$.h5parm.sol000/amplitude/val"
    assert differences[0].candidate["max_absolute_difference"] == pytest.approx(5e-4)

import json
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from rapthor.execution.equivalence import (
    assert_backend_equivalent,
    check_reference_artifacts,
    collect_backend_summary,
    collect_product_summaries,
    compare_backend_runs,
    compare_fits_product,
    compare_h5parm_product,
    compare_saved_reference_equivalence_manifest,
    compare_saved_reference_equivalence_scenario,
    format_differences,
    materialize_scenario_parset,
    reference_artifact_dir,
    reference_artifact_root_from_environment,
    required_reference_artifact_items,
    run_equivalence_pair,
    scenario_parset_file,
    scenario_parset_materializer,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures"
REPO_ROOT = Path(__file__).resolve().parents[2]


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


def test_scenario_parset_file_resolves_first_parset_fixture(tmp_path):
    parset = tmp_path / "scenario.parset"
    parset.write_text("[global]\ndir_working = .\n")

    assert (
        scenario_parset_file(
            {
                "id": "smoke",
                "fixture_refs": ["notes.txt", "scenario.parset"],
            },
            repo_root=tmp_path,
        )
        == parset
    )


def test_scenario_parset_file_rejects_missing_or_absent_parset(tmp_path):
    with pytest.raises(ValueError, match="does not define a parset fixture"):
        scenario_parset_file({"id": "smoke", "fixture_refs": ["notes.txt"]}, repo_root=tmp_path)

    with pytest.raises(FileNotFoundError, match="parset fixture not found"):
        scenario_parset_file(
            {"id": "smoke", "fixture_refs": ["missing.parset"]}, repo_root=tmp_path
        )


def test_materialize_scenario_parset_fills_template_values_and_overrides(tmp_path):
    source_parset = tmp_path / "template.parset"
    source_parset.write_text(
        "[global]\n"
        "dir_working = \n"
        "input_ms = \n"
        "input_skymodel = \n"
        "apparent_skymodel = \n"
        "strategy = \n"
        "\n"
        "[imaging]\n"
        "photometry_skymodel = \n"
        "astrometry_skymodel = None\n"
        "shared_facet_rw = False\n"
        "\n"
        "[cluster]\n"
        "local_scratch_dir = \n"
        "global_scratch_dir = \n"
    )
    strategy = tmp_path / "strategy.py"
    strategy.write_text("strategy_steps = []\n")
    scenario = {
        "id": "smoke",
        "parset_overrides": {
            "global": {"strategy": "${STRATEGY_PATH}"},
            "imaging.shared_facet_rw": "True",
            "calibration.maxiter": 5,
        },
    }

    materialized = materialize_scenario_parset(
        scenario,
        source_parset,
        tmp_path / "candidate",
        environ={
            "RAPTHOR_EQUIVALENCE_INPUT_MS": "/data/input.ms",
            "RAPTHOR_EQUIVALENCE_INPUT_SKYMODEL": "/data/true.txt",
            "RAPTHOR_EQUIVALENCE_APPARENT_SKYMODEL": "/data/apparent.txt",
            "STRATEGY_PATH": str(strategy),
        },
    )

    text = materialized.read_text()
    assert "dir_working = " + str(tmp_path / "candidate") in text
    assert "input_ms = /data/input.ms" in text
    assert "input_skymodel = /data/true.txt" in text
    assert "apparent_skymodel = /data/apparent.txt" in text
    assert f"strategy = {strategy}" in text
    assert "photometry_skymodel = /data/true.txt" in text
    assert "astrometry_skymodel = /data/true.txt" in text
    assert "shared_facet_rw = True" in text
    assert "maxiter = 5" in text
    assert f"local_scratch_dir = {tmp_path / 'candidate' / 'scratch'}" in text


def test_materialize_scenario_parset_rejects_unresolved_override(tmp_path):
    source_parset = tmp_path / "template.parset"
    source_parset.write_text("[global]\ndir_working = \ninput_ms = input.ms\n")

    with pytest.raises(ValueError, match="unresolved environment reference"):
        materialize_scenario_parset(
            {
                "id": "smoke",
                "parset_overrides": {"global.strategy": "${MISSING_STRATEGY}"},
            },
            source_parset,
            tmp_path / "candidate",
            environ={},
        )


def test_scenario_parset_materializer_binds_scenario(tmp_path):
    source_parset = tmp_path / "template.parset"
    source_parset.write_text("[global]\ndir_working = \nstrategy = \n")
    materializer = scenario_parset_materializer(
        {"id": "smoke", "parset_overrides": {"global.strategy": "bound.py"}},
        environ={},
    )

    materialized = materializer(source_parset, tmp_path / "candidate")

    assert "strategy = bound.py" in materialized.read_text()


def test_compare_saved_reference_equivalence_scenario_runs_prefect_candidate(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    parset = repo_root / "scenario.parset"
    parset.write_text("[global]\ndir_working = .\n")
    scenario = {
        "id": "smoke",
        "cwl_reference_artifact_dir": "saved-smoke",
        "fixture_refs": ["scenario.parset"],
        "comparison_scopes": ["operations", "products"],
    }
    reference_root = tmp_path / "references"
    reference_dir = reference_artifact_dir(reference_root, scenario)
    _write_backend_state(reference_dir)
    calls = []

    def materialize(source_parset, working_dir):
        assert source_parset == parset
        candidate_parset = working_dir / "candidate.parset"
        candidate_parset.write_text(source_parset.read_text())
        return candidate_parset

    def fake_prefect_runner(parset_file, working_dir, logging_level):
        calls.append((parset_file, working_dir, logging_level))
        _write_backend_state(working_dir)
        return {"ran": True}

    result = compare_saved_reference_equivalence_scenario(
        scenario,
        reference_root,
        tmp_path / "runs",
        repo_root=repo_root,
        prefect_runner=fake_prefect_runner,
        logging_level="debug",
        parset_materializer=materialize,
    )

    assert result.ok
    assert result.differences == ()
    assert result.reference_run.backend == "cwl"
    assert result.reference_run.working_dir == reference_dir
    assert result.candidate_run.backend == "prefect"
    assert result.candidate_run.working_dir == tmp_path / "runs" / "smoke" / "prefect"
    assert result.candidate_run.result == {"ran": True}
    assert calls == [
        (
            tmp_path / "runs" / "smoke" / "prefect" / "candidate.parset",
            tmp_path / "runs" / "smoke" / "prefect",
            "debug",
        )
    ]


def test_compare_saved_reference_equivalence_scenario_reports_differences(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "scenario.parset").write_text("[global]\ndir_working = .\n")
    scenario = {
        "id": "smoke",
        "fixture_refs": ["scenario.parset"],
        "comparison_scopes": ["operations", "products"],
    }
    reference_root = tmp_path / "references"
    _write_backend_state(reference_artifact_dir(reference_root, scenario), "reference.fits")

    def fake_prefect_runner(parset_file, working_dir, logging_level):
        _write_backend_state(working_dir, "candidate.fits")

    result = compare_saved_reference_equivalence_scenario(
        scenario,
        reference_root,
        tmp_path / "runs",
        repo_root=repo_root,
        prefect_runner=fake_prefect_runner,
    )

    assert not result.ok
    assert result.differences[0].path == "$.operations.image_1.outputs.image.basename"


def test_compare_saved_reference_equivalence_scenario_uses_default_materializer(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    parset = repo_root / "scenario.parset"
    parset.write_text("[global]\ndir_working = old\ninput_ms = input.ms\n")
    scenario = {
        "id": "smoke",
        "fixture_refs": ["scenario.parset"],
        "comparison_scopes": ["operations", "products"],
    }
    reference_root = tmp_path / "references"
    _write_backend_state(reference_artifact_dir(reference_root, scenario))

    def fake_prefect_runner(parset_file, working_dir, logging_level):
        assert f"dir_working = {working_dir}" in parset_file.read_text()
        _write_backend_state(working_dir)

    result = compare_saved_reference_equivalence_scenario(
        scenario,
        reference_root,
        tmp_path / "runs",
        repo_root=repo_root,
        prefect_runner=fake_prefect_runner,
    )

    assert result.ok


def test_compare_saved_reference_equivalence_manifest_runs_all_scenarios(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "first.parset").write_text("[global]\ndir_working = .\n")
    (repo_root / "second.parset").write_text("[global]\ndir_working = .\n")
    scenarios = [
        {"id": "first", "fixture_refs": ["first.parset"], "comparison_scopes": ["operations"]},
        {"id": "second", "fixture_refs": ["second.parset"], "comparison_scopes": ["operations"]},
    ]
    reference_root = tmp_path / "references"
    for scenario in scenarios:
        _write_backend_state(reference_artifact_dir(reference_root, scenario))

    def fake_prefect_runner(parset_file, working_dir, logging_level):
        _write_backend_state(working_dir)

    results = compare_saved_reference_equivalence_manifest(
        scenarios,
        reference_root,
        tmp_path / "runs",
        repo_root=repo_root,
        prefect_runner=fake_prefect_runner,
    )

    assert [result.scenario_id for result in results] == ["first", "second"]
    assert all(result.ok for result in results)
    assert [result.candidate_run.working_dir.name for result in results] == ["prefect", "prefect"]


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


def test_restart_equivalence_accepts_prefect_produced_output_records(tmp_path):
    cwl_dir = tmp_path / "cwl"
    prefect_dir = tmp_path / "prefect"
    _write_operation_state(
        cwl_dir, operation_name="image_1", outputs_filename="pipeline_outputs.json"
    )
    _write_operation_state(prefect_dir, operation_name="image_1", outputs_filename=".outputs.json")
    (cwl_dir / "operation_order.json").write_text(json.dumps(["image_1"]))
    (prefect_dir / "operation_order.json").write_text(json.dumps(["image_1"]))

    assert compare_backend_runs(cwl_dir, prefect_dir) == []


def test_collect_backend_summary_includes_product_metadata(tmp_path):
    working_dir = tmp_path / "run"
    image_dir = working_dir / "images" / "image_1"
    skymodel_dir = working_dir / "skymodels" / "image_1"
    region_dir = working_dir / "regions" / "image_1"
    for directory in (image_dir, skymodel_dir, region_dir):
        directory.mkdir(parents=True)

    fits.writeto(image_dir / "sector_1-MFS-I-image.fits", np.array([[1.0, 2.0], [3.0, np.nan]]))
    (skymodel_dir / "sector_1.true_sky.txt").write_text(
        "FORMAT = Name, Type, Patch\n , , Patch_1\nsource_1, POINT, Patch_1\n",
        encoding="utf-8",
    )
    (region_dir / "sector_1.reg").write_text("fk5\ncircle(1,2,3)\n", encoding="utf-8")

    products = collect_product_summaries(working_dir)

    assert products["images/image_1/sector_1-MFS-I-image.fits"] == {
        "type": "fits",
        "hdus": [
            {
                "index": 0,
                "shape": [2, 2],
                "dtype": ">f8",
                "finite_count": 3,
                "nan_count": 1,
                "min": 1.0,
                "max": 3.0,
                "mean": 2.0,
                "std": pytest.approx(0.816496580928),
            }
        ],
    }
    assert products["skymodels/image_1/sector_1.true_sky.txt"] == {
        "type": "skymodel",
        "source_count": 1,
        "patch_count": 1,
    }
    assert products["regions/image_1/sector_1.reg"] == {
        "type": "region",
        "lines": ["fk5", "circle(1,2,3)"],
    }


def test_collect_product_summaries_includes_h5parm_metadata(tmp_path):
    h5py = pytest.importorskip("h5py")

    working_dir = tmp_path / "run"
    h5parm_dir = working_dir / "h5parms" / "calibrate_1"
    h5parm_dir.mkdir(parents=True)
    with h5py.File(h5parm_dir / "field-solutions.h5", "w") as h5_file:
        solset = h5_file.create_group("sol000")
        phase = solset.create_group("phase000")
        values = phase.create_dataset("val", data=np.ones((2, 3)))
        values.attrs["AXES"] = b"time,ant"
        solset.create_group("antenna")

    products = collect_product_summaries(working_dir)

    assert products["h5parms/calibrate_1/field-solutions.h5"] == {
        "type": "h5parm",
        "solsets": ["sol000"],
        "soltabs": {"sol000": ["phase000"]},
        "datasets": {"sol000/phase000/val": {"shape": [2, 3], "dtype": "float64"}},
        "axes": {"sol000/phase000/val": "time,ant"},
    }


def test_compare_backend_runs_reports_product_metadata_differences(tmp_path):
    cwl_dir = tmp_path / "cwl"
    prefect_dir = tmp_path / "prefect"
    cwl_image_dir = cwl_dir / "images" / "image_1"
    prefect_image_dir = prefect_dir / "images" / "image_1"
    cwl_image_dir.mkdir(parents=True)
    prefect_image_dir.mkdir(parents=True)
    fits.writeto(cwl_image_dir / "sector_1-MFS-I-image.fits", np.array([[1.0, 2.0]]))
    fits.writeto(prefect_image_dir / "sector_1-MFS-I-image.fits", np.array([[1.0, 3.0]]))

    differences = compare_backend_runs(cwl_dir, prefect_dir)

    paths = [difference.path for difference in differences]
    assert "$.products.images/image_1/sector_1-MFS-I-image.fits.hdus[0].max" in paths
    assert "$.products.images/image_1/sector_1-MFS-I-image.fits.hdus[0].mean" in paths


def test_equivalence_gate_scenarios_cover_supported_merge_matrix():
    matrix = json.loads((FIXTURE_DIR / "supported_merge_feature_matrix.json").read_text())
    scenarios = json.loads((FIXTURE_DIR / "equivalence_gate_scenarios.json").read_text())

    supported_equivalence_ids = {
        entry["id"] for entry in matrix["supported"] if "equivalence" in entry["test_types"]
    }
    scenario_ids = {scenario["matrix_id"] for scenario in scenarios["scenarios"]}

    assert scenario_ids == supported_equivalence_ids
    for scenario in scenarios["scenarios"]:
        assert "operations" in scenario["comparison_scopes"], scenario["id"]
        assert "products" in scenario["comparison_scopes"], scenario["id"]
        assert scenario["cwl_reference_artifact_dir"] == scenario["id"]
        assert scenario["fixture_refs"], scenario["id"]
        assert all((REPO_ROOT / fixture_ref).exists() for fixture_ref in scenario["fixture_refs"])
        if scenario["matrix_id"] == "mpi_wsclean":
            assert scenario.get("target_environment") is True


def test_reference_artifact_root_uses_environment_when_set(tmp_path):
    assert reference_artifact_root_from_environment({}) is None
    assert reference_artifact_root_from_environment({"RAPTHOR_CWL_REFERENCE_ROOT": ""}) is None
    assert (
        reference_artifact_root_from_environment({"RAPTHOR_CWL_REFERENCE_ROOT": str(tmp_path)})
        == tmp_path
    )


def test_required_reference_artifact_items_follow_comparison_scopes():
    assert required_reference_artifact_items(
        {
            "id": "full",
            "comparison_scopes": [
                "operations",
                "products",
                "h5parm",
                "fits",
                "skymodel",
                "regions",
            ],
        }
    ) == (
        "pipelines",
        "operation_order",
        "product_roots",
        "fits_products",
        "h5parm_products",
        "skymodel_products",
        "region_products",
    )
    assert required_reference_artifact_items(
        {"id": "restart", "comparison_scopes": ["operations", "products", "restart"]}
    ) == ("pipelines", "operation_order", "product_roots", "restart_state")


def test_check_reference_artifacts_reports_missing_saved_cwl_items(tmp_path):
    scenario = {
        "id": "smoke",
        "cwl_reference_artifact_dir": "smoke",
        "comparison_scopes": ["operations", "products", "fits", "restart"],
    }

    checks = check_reference_artifacts([scenario], root_dir=tmp_path)

    assert len(checks) == 1
    assert checks[0].scenario_id == "smoke"
    assert checks[0].artifact_dir == tmp_path / "smoke"
    assert checks[0].missing_items == (
        "artifact_dir",
        "pipelines",
        "operation_order",
        "product_roots",
        "fits_products",
        "restart_state",
    )
    assert not checks[0].ok


def test_check_reference_artifacts_accepts_minimal_saved_cwl_reference(tmp_path):
    scenario = {
        "id": "smoke",
        "cwl_reference_artifact_dir": "saved-smoke",
        "comparison_scopes": ["operations", "products", "fits", "restart"],
    }
    artifact_dir = reference_artifact_dir(tmp_path, scenario)
    operation_dir = artifact_dir / "pipelines" / "image_1"
    image_dir = artifact_dir / "images" / "image_1"
    operation_dir.mkdir(parents=True)
    image_dir.mkdir(parents=True)
    (operation_dir / ".done").touch()
    (operation_dir / ".outputs.json").write_text("{}")
    (artifact_dir / "operation_order.json").write_text(json.dumps(["image_1"]))
    fits.writeto(image_dir / "sector_1-MFS-I-image.fits", np.array([[1.0]]))

    checks = check_reference_artifacts([scenario], root_dir=tmp_path)

    assert checks == [
        checks[0].__class__(
            scenario_id="smoke",
            artifact_dir=artifact_dir,
            missing_items=(),
        )
    ]
    assert checks[0].ok


def test_check_reference_artifacts_is_disabled_without_root():
    assert (
        check_reference_artifacts(
            [{"id": "smoke", "comparison_scopes": ["operations"]}], environ={}
        )
        == []
    )


def test_saved_cwl_reference_artifacts_are_complete_when_configured():
    scenarios = json.loads((FIXTURE_DIR / "equivalence_gate_scenarios.json").read_text())[
        "scenarios"
    ]
    checks = check_reference_artifacts(scenarios)
    if not checks:
        pytest.skip("RAPTHOR_CWL_REFERENCE_ROOT is not set")

    missing = [check for check in checks if not check.ok]

    assert not missing, "\n".join(
        f"{check.scenario_id}: {check.artifact_dir} missing {', '.join(check.missing_items)}"
        for check in missing
    )


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

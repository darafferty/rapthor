"""Saved CWL-reference regression tests for the Prefect execution path."""

import json
import os
from pathlib import Path

import pytest

from rapthor.execution.equivalence import (
    REFERENCE_ARTIFACT_ROOT_ENV,
    check_reference_artifacts,
    compare_saved_reference_equivalence_manifest,
    format_differences,
    reference_artifact_root_from_environment,
    required_gate_scenarios,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "tests" / "execution" / "fixtures" / "equivalence_gate_scenarios.json"
RUN_SAVED_CWL_EQUIVALENCE_ENV = "RAPTHOR_RUN_SAVED_CWL_EQUIVALENCE"
SCENARIO_FILTER_ENV = "RAPTHOR_EQUIVALENCE_SCENARIOS"


def _load_manifest_scenarios():
    return json.loads(MANIFEST_PATH.read_text())["scenarios"]


def _selected_scenarios(scenarios):
    selected = os.environ.get(SCENARIO_FILTER_ENV)
    if selected in (None, ""):
        return required_gate_scenarios(scenarios)

    requested_ids = {scenario_id.strip() for scenario_id in selected.split(",")}
    requested_ids.discard("")
    assert requested_ids, f"{SCENARIO_FILTER_ENV} did not contain any scenario ids"
    scenarios_by_id = {str(scenario["id"]): scenario for scenario in scenarios}
    missing_ids = sorted(requested_ids - set(scenarios_by_id))
    assert not missing_ids, (
        f"{SCENARIO_FILTER_ENV} contains unknown scenario ids: {', '.join(missing_ids)}"
    )
    return [scenarios_by_id[scenario_id] for scenario_id in sorted(requested_ids)]


def _missing_artifact_message(checks):
    return "\n".join(
        f"{check.scenario_id}: {check.artifact_dir} missing {', '.join(check.missing_items)}"
        for check in checks
        if not check.ok
    )


def _difference_message(results):
    messages = []
    for result in results:
        if result.ok:
            continue
        messages.append(f"{result.scenario_id}:\n{format_differences(result.differences)}")
    return "\n\n".join(messages)


@pytest.mark.integration
def test_prefect_outputs_match_saved_cwl_references(tmp_path):
    """Run Prefect scenarios and compare outputs to saved CWL artifacts."""
    if os.environ.get(RUN_SAVED_CWL_EQUIVALENCE_ENV) != "1":
        pytest.skip(f"{RUN_SAVED_CWL_EQUIVALENCE_ENV}=1 is required")

    reference_root = reference_artifact_root_from_environment()
    if reference_root is None:
        pytest.skip(f"{REFERENCE_ARTIFACT_ROOT_ENV} is not set")

    scenarios = _selected_scenarios(_load_manifest_scenarios())
    checks = check_reference_artifacts(scenarios, root_dir=reference_root)
    assert all(check.ok for check in checks), _missing_artifact_message(checks)

    results = compare_saved_reference_equivalence_manifest(
        scenarios,
        reference_root,
        tmp_path / "prefect-runs",
        repo_root=REPO_ROOT,
    )

    assert all(result.ok for result in results), _difference_message(results)

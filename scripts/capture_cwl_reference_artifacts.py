#!/usr/bin/env python3
"""Capture saved CWL reference artifacts from a pre-cutover Rapthor checkout."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from rapthor.execution.equivalence import (
    EQUIVALENCE_APPARENT_SKYMODEL_ENV,
    EQUIVALENCE_INPUT_MS_ENV,
    EQUIVALENCE_INPUT_SKYMODEL_ENV,
    EQUIVALENCE_STRATEGY_ENV,
    REFERENCE_ARTIFACT_ROOT_ENV,
    check_reference_artifacts,
    reference_artifact_dir,
    scenario_parset_file,
    scenario_parset_materializer,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    REPO_ROOT / "tests" / "execution" / "fixtures" / "equivalence_gate_scenarios.json"
)
LEGACY_CWL_REPO_ENV = "RAPTHOR_LEGACY_CWL_REPO"


def _parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run equivalence scenarios with the legacy CWL implementation and save the "
            "working directories as reference artifacts."
        )
    )
    parser.add_argument(
        "--legacy-repo",
        default=os.environ.get(LEGACY_CWL_REPO_ENV),
        help=f"Path to a pre-cutover Rapthor checkout. Defaults to ${LEGACY_CWL_REPO_ENV}.",
    )
    parser.add_argument(
        "--reference-root",
        default=os.environ.get(REFERENCE_ARTIFACT_ROOT_ENV),
        help=(
            "Directory that will receive one artifact directory per scenario. "
            f"Defaults to ${REFERENCE_ARTIFACT_ROOT_ENV}."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Equivalence scenario manifest to capture.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Current checkout used to resolve manifest fixture_refs.",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        default=[],
        help="Scenario id to capture. May be passed multiple times; defaults to all scenarios.",
    )
    parser.add_argument(
        "--logging-level",
        default="info",
        choices=("debug", "info", "warning"),
        help="Logging level passed to rapthor.process.run in the legacy checkout.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove an existing scenario artifact directory before capturing it.",
    )
    return parser


def _load_scenarios(manifest_path):
    return json.loads(Path(manifest_path).read_text())["scenarios"]


def _select_scenarios(scenarios, selected_ids):
    if not selected_ids:
        return scenarios
    scenarios_by_id = {str(scenario["id"]): scenario for scenario in scenarios}
    missing_ids = sorted(set(selected_ids) - set(scenarios_by_id))
    if missing_ids:
        raise SystemExit(f"Unknown scenario ids: {', '.join(missing_ids)}")
    return [scenarios_by_id[scenario_id] for scenario_id in selected_ids]


def _required_directory(value, name):
    if value in (None, ""):
        raise SystemExit(f"{name} is required")
    path = Path(value).expanduser()
    if not path.is_dir():
        raise SystemExit(f"{name} is not a directory: {path}")
    return path


def _pythonpath_with_repo(repo):
    current_pythonpath = os.environ.get("PYTHONPATH")
    if current_pythonpath:
        return str(repo) + os.pathsep + current_pythonpath
    return str(repo)


def _run_legacy_cwl(legacy_repo, parset_file, logging_level):
    code = (
        "import sys\n"
        "from rapthor import process\n"
        "process.run(sys.argv[1], logging_level=sys.argv[2])\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = _pythonpath_with_repo(legacy_repo)
    result = subprocess.run(
        [sys.executable, "-c", code, str(parset_file), logging_level],
        cwd=legacy_repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Legacy CWL run failed for {parset_file}")
    return result


def _prepare_artifact_dir(artifact_dir, overwrite):
    if artifact_dir.exists() and any(artifact_dir.iterdir()):
        if not overwrite:
            raise RuntimeError(
                f"Artifact directory already exists and is not empty: {artifact_dir}. "
                "Pass --overwrite to replace it."
            )
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)


def _format_missing_artifacts(checks):
    return "\n".join(
        f"{check.scenario_id}: {check.artifact_dir} missing {', '.join(check.missing_items)}"
        for check in checks
        if not check.ok
    )


def _capture_scenario(scenario, reference_root, legacy_repo, repo_root, logging_level, overwrite):
    artifact_dir = reference_artifact_dir(reference_root, scenario)
    _prepare_artifact_dir(artifact_dir, overwrite)

    source_parset = scenario_parset_file(scenario, repo_root=repo_root)
    parset_materializer = scenario_parset_materializer(scenario, repo_root=repo_root)
    captured_parset = parset_materializer(source_parset, artifact_dir)

    print(f"Capturing {scenario['id']} into {artifact_dir}")
    _run_legacy_cwl(legacy_repo, captured_parset, logging_level)

    checks = check_reference_artifacts([scenario], root_dir=reference_root)
    if not all(check.ok for check in checks):
        raise RuntimeError(_format_missing_artifacts(checks))


def _print_environment_hint():
    print("Common required inputs are filled from these environment variables when blank:")
    print(f"  {EQUIVALENCE_INPUT_MS_ENV}")
    print(f"  {EQUIVALENCE_INPUT_SKYMODEL_ENV}")
    print(f"  {EQUIVALENCE_APPARENT_SKYMODEL_ENV}")
    print(f"  {EQUIVALENCE_STRATEGY_ENV}")


def main(argv=None):
    args = _parser().parse_args(argv)
    legacy_repo = _required_directory(args.legacy_repo, "--legacy-repo")
    reference_root = Path(args.reference_root).expanduser() if args.reference_root else None
    if reference_root is None:
        raise SystemExit(f"--reference-root or ${REFERENCE_ARTIFACT_ROOT_ENV} is required")
    reference_root.mkdir(parents=True, exist_ok=True)

    scenarios = _select_scenarios(_load_scenarios(args.manifest), args.scenario)
    _print_environment_hint()
    for scenario in scenarios:
        _capture_scenario(
            scenario,
            reference_root,
            legacy_repo,
            args.repo_root,
            args.logging_level,
            args.overwrite,
        )
    print(f"Captured {len(scenarios)} scenario(s) in {reference_root}")


if __name__ == "__main__":
    main()

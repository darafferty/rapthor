# AGENTS.md

Guidance for AI coding agents working in this repository.

## First Stops

1. Read this file for the repository contract and hard guardrails.
2. Use `.agents/README.md` as the index for focused decision guides.
3. Read `.agents/scientific_glossary.md` before changing calibration,
   prediction, imaging, sky-model, strategy, or scientific documentation logic.
4. Read `PLAN.md` before changing architecture boundaries, task granularity,
   benchmarking, runtime bootstrap, preflight behavior, scalability work, or
   contributor-facing development docs.

## Project Overview

Rapthor is a Python package and command-line pipeline for LOFAR
direction-dependent effect correction, with ongoing SKA-Low support. The
production execution path uses Prefect/Dask flows to run radio astronomy tools
such as DP3, WSClean, EveryBeam, IDG, and related utilities.

Most changes touch one or more of these areas:

- `rapthor/lib/`: domain objects such as `Field`, `Observation`, `Sector`,
  `Cluster`, `Operation`, strategy handling, and parset handling.
- `rapthor/operations/`: thin operation adapters that translate domain state
  into execution inputs and finalize outputs.
- `rapthor/execution/`: Prefect/Dask flows, command builders, payload contracts,
  validation, migrated helper-script logic, shell execution, artifacts, resource
  checks, and task-runner helpers.
- `rapthor/execution/image/`, `calibrate/`, `concatenate/`, `predict/`,
  `mosaic/`, and `pipeline/`: execution owner packages for operation-specific
  payloads, commands, outputs, flow wiring, and orchestration.
- `rapthor/settings/`: package defaults in `defaults.parset` and
  `defaults.json`.
- `docs/source/`, `examples/`, and `tests/`: user docs, example parsets and
  strategies, and focused test suites.
- `.agents/`: agent-facing playbooks.
- `PLAN.md`: current architecture stabilization, benchmarking, Dask
  scalability, runtime UX, and deferred-refactor roadmap.

Package console scripts are declared in `pyproject.toml`:

- `rapthor = rapthor.cli:main`
- `concat_linc_files = rapthor.execution.concatenate.linc_cli:main`

## Agent Playbooks

Use the focused guides instead of growing this file with duplicated detail:

- `.agents/repo_architecture.md`: ownership boundaries and change placement.
- `.agents/pipeline_contracts.md`: serializable payloads, output records,
  h5parm products, restart behavior, and command-builder contracts.
- `.agents/parset_strategy_guide.md`: user-facing option and strategy update
  workflow.
- `.agents/external_tools.md`: DP3, WSClean, EveryBeam, IDG/IDGCal, PyBDSF,
  Casacore, Prefect/Dask, containers, MPI, and cluster runtime notes.
- `.agents/testing_playbook.md`: focused commands and confidence checks by
  change type.
- `.agents/scientific_glossary.md`: scientific vocabulary, self-calibration
  reasoning, naming guidance, and configuration recommendations.

## Hard Guardrails

- Check the worktree before editing. Do not overwrite unrelated local changes,
  and do not commit unless explicitly asked.
- Keep patches scoped to the requested behavior. Avoid broad refactors while
  making behavioral fixes.
- Keep operation adapters thin. Production command mechanics, payload
  validation, output discovery, and migrated helper logic belong under the
  appropriate `rapthor/execution/<owner>/` package.
- Keep Prefect/Dask worker payloads plain and serializable. Do not send
  `Field`, `Observation`, `Sector`, operation instances, open file handles,
  table objects, or live subprocess state to workers.
- Keep command builders deterministic and testable. Use option dataclasses when
  they clarify stable argument groups, but do not hide scientific intent behind
  generic abstractions.
- Preserve module-level logging style, typically loggers named like
  `rapthor:image` or `rapthor:calibrate`.
- Keep generated data, large Measurement Sets, build artifacts, downloaded
  archives, integration/equivalence/demo run products, `.tox`, `.ruff_cache`,
  `htmlcov`, and `__pycache__` out of source decisions.
- Prefer existing small fixtures in `tests/resources/` over adding large files.
- After any code change, run `ruff check --fix --select I` and `ruff format`
  before final verification. If the commands cannot be run, report that and
  name the remaining formatting/import-order risk.

## Scientific And Strategy Guardrails

- `calibration_strategy` is the production interface for calibration solve type
  and solve order. Do not reintroduce legacy solve toggles such as
  `do_fulljones_solve` or `do_slowgain_solve`.
- Use `slow_gains` in strategy values. In prose, "slow gain" or "slow diagonal
  gain" is fine.
- DD slow-gain solves are currently diagonal, and the default DD strategy should
  remain explicit about the post-slow `medium_phase` solve rather than relying
  on hidden slot behavior.
- The built-in selfcal strategy runs early phase-only DD cycles when Rapthor did
  not generate the initial sky model. Do not flatten that behavior into an
  always-amplitude-first rule.
- Solutions from one calibration cycle must not be silently reused in later
  cycles unless the strategy and operation contract explicitly say so.
- When adding a user-facing option, update defaults, docs, examples, operation
  payloads, command builders, validators, templates, and tests together.
- Treat apparent-sky, true-sky, DI, DD, full-Jones, normalization, and screen
  products as distinct scientific states.

## Testing And Verification

Use `.agents/testing_playbook.md` for detailed test selection. Start focused,
then broaden when behavior crosses module, payload, runtime, or scientific
product boundaries.

Common commands:

```bash
tox -e lint
tox
python -m pytest -m "not integration" tests
python -m pytest tests/operations/test_image.py
```

To mirror the current tox split manually in the prepared dev-container
environment:

```bash
python -m pytest tests/lib/test_field.py -m "not integration"
python -m pytest -m "not integration and prefect" tests
python -m pytest -m "not integration and not prefect" -n auto --dist worksteal --ignore=tests/lib/test_field.py tests
```

Run integration tests only when the environment has the required external tools
and data access:

```bash
RAPTHOR_TEST_RUN_ROOT=/tmp/rapthor-integration-runs python -m pytest -m integration -vv -ra --durations=0 tests/integration tests/operations/integration
```

Use `scripts/dev/run_saved_cwl_equivalence.py` for heavier scientific
confidence after scientific logic changes, script-to-module migrations,
calibration strategy changes, or changes to FITS/h5parm/skymodel products.

## Development Environment

- Python support is declared as `>=3.9`.
- Build, dependency, lint, pytest, and tox metadata live in `pyproject.toml`.
- Test and lint dependencies are available through the `test` and `dev`
  dependency groups.
- Documentation dependencies are available through the `docs` dependency group
  and are installed by the dev container.
- External radio astronomy dependencies are not fully mocked everywhere. Some
  workflows require tools such as DP3, EveryBeam, IDG, WSClean, Casacore, and
  Python-Casacore.
- The prepared dev container is the preferred environment for formatting,
  tests, integration checks, equivalence checks, and demo runs. Isolated tox
  environments may try to rebuild compiled packages such as `python-casacore`
  or `everybeam`.

When installing locally, prefer an editable install with development
dependencies if your environment supports it:

```bash
python -m pip install -e ".[dev]"
```

## Documentation

Update docs when changing user-facing behavior, parset options, operation
semantics, command-line behavior, or installation/runtime requirements. The main
documentation source is under `docs/source/`, with README-level overview in
`README.md`.

Architecture notes and the current Prefect/Dask orchestration diagram live under
`docs/source/development/`. Keep them aligned with owner-package, task-runner,
runtime-bootstrap, and roadmap changes.

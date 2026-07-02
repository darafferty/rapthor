# AGENTS.md

Guidance for AI coding agents working in this repository.

## Project Overview

Rapthor is a Python package and command-line pipeline for LOFAR direction-dependent
effect correction, with ongoing SKA-Low support. The production execution path
uses Prefect/Dask flows to run radio astronomy tools such as DP3, WSClean,
EveryBeam, IDG, and related utilities.

Most changes affect one or more of these layers:

- Python domain model and orchestration code under `rapthor/lib/`
- Prefect/Dask execution owner packages under `rapthor/execution/`
- Thin pipeline operation adapters under `rapthor/operations/`
- Package console scripts declared in `pyproject.toml`
- Parset defaults under `rapthor/settings/`
- Unit, operation, execution, architecture, and integration tests under `tests/`

## Repository Layout

- `rapthor/cli.py`: installed `rapthor` command entry point.
- `rapthor/lib/`: core objects such as `Field`, `Observation`, `Sector`, `Cluster`,
  `Operation`, strategy handling, and parset handling.
- `rapthor/execution/pipeline/flow.py`: top-level Prefect pipeline flow and
  operation scheduling.
- `rapthor/execution/`: Prefect/Dask flows, command builders, payload contracts,
  validation, migrated helper-script logic, shell execution, artifacts, resource
  checks, and task-runner helpers.
- `rapthor/execution/image/`, `calibrate/`, `concatenate/`, `predict/`, `mosaic/`:
  owner packages for operation-specific payloads, commands, outputs, flow wiring,
  and execution-owned module adapters.
- `rapthor/operations/`: high-level operation adapters. Keep these thin; prefer
  putting execution mechanics in the corresponding `rapthor/execution/<owner>/`
  package.
- `scripts/dev/` and `scripts/prod/`: developer and deployment helpers only.
  Do not add production pipeline logic here.
- Package console scripts:
  - `rapthor = rapthor.cli:main`
  - `concat_linc_files = rapthor.execution.concatenate.linc_cli:main`
- `tests/lib/`, `tests/execution/`, `tests/operations/`,
  `tests/architecture/`: focused test suites for the corresponding code.
- `tests/integration/`: broader pipeline behavior tests. Treat these as heavier
  and more environment-sensitive.
- `tests/resources/`: small test parsets, sky models, FITS files, and templates.
- `docs/source/`: Sphinx documentation.
- `examples/`: example parsets and strategy files.

## Development Environment

- Python support is declared as `>=3.9`.
- Build and dependency metadata lives in `pyproject.toml`.
- Test and lint dependencies are available through the `test` and `dev`
  dependency groups.
- Documentation dependencies are available through the `docs` dependency group
  and are installed by the dev container.
- External radio astronomy dependencies are not fully mocked everywhere. Some
  workflows require tools such as DP3, EveryBeam, IDG, WSClean, Casacore, and
  Python-Casacore.

When installing locally, prefer an editable install with development dependencies
if your environment supports it:

```bash
python -m pip install -e ".[dev]"
```

Some dependencies are Git-based or compiled, so dependency installation may need
network access and system packages.

The dev container is the preferred environment for formatting, tests,
integration checks, equivalence checks, and demo runs. The prepared container
Python environment includes the external astronomy dependencies used by the
pipeline. Isolated tox Python environments may try to rebuild compiled packages
such as `python-casacore` or `everybeam`; make sure Casacore headers are
available before relying on those environments locally.

## Common Commands

Run lint checks:

```bash
tox -e lint
```

Run the default unit test suite through tox:

```bash
tox
```

Run non-integration tests directly:

```bash
python -m pytest -m "not integration" tests
```

To mirror the current tox split manually in the prepared dev-container
environment:

```bash
python -m pytest tests/lib/test_field.py -m "not integration"
python -m pytest -m "not integration and prefect" tests
python -m pytest -m "not integration and not prefect" -n auto --dist worksteal --ignore=tests/lib/test_field.py tests
```

Run a focused test file:

```bash
python -m pytest tests/operations/test_image.py
```

Run integration tests only when the environment has the required external tools
and data access:

```bash
tox -e test_integration
```

When running integration, equivalence, or demo commands locally, place generated
run roots on a filesystem with enough free space. WSClean writes large FITS
products, and the repository workspace can fill quickly:

```bash
RAPTHOR_TEST_RUN_ROOT=/tmp/rapthor-integration-runs python -m pytest -m integration -vv -ra --durations=0 tests/integration tests/operations/integration
python scripts/dev/run_saved_cwl_equivalence.py --run-root /tmp/rapthor-equivalence --stop-on-failure
python scripts/dev/run-rapthor-prefect-demo.py examples/generated/prefect_demo_rich/prefect_demo_rich.parset --run-dir /tmp/rapthor-prefect-demo --no-keep-server
```

The tox configuration runs `tests/lib/test_field.py` sequentially, Prefect-marked
tests serially, and the rest of the non-integration tests in parallel. If you
reproduce tox behavior manually, keep those special cases in mind.

## Coding Style

- Follow the existing style in the touched file before introducing new patterns.
- Ruff is configured in `pyproject.toml` with a line length of 100.
- The active lint rules are primarily `E` and `F`; `E501` is ignored.
- Use type hints when they clarify contracts, especially around operation inputs,
  parset-derived values, and helper functions.
- Keep payloads passed to Prefect/Dask tasks plain and serializable. Do not send
  `Field`, `Observation`, `Sector`, or operation instances to workers.
- Prefer shared execution payload validators for common string, basename, list,
  and file-record checks. Avoid local private aliases for shared helpers unless
  the local name adds real domain meaning.
- Keep command builders deterministic and easy to compare in tests. Use option
  dataclasses for stable argument groups, but avoid abstractions that hide the
  scientific intent of a DP3, WSClean, IDG, or helper command.
- Keep comments useful and sparse. Prefer explaining why a workflow needs a
  condition over restating what the next line does.
- Preserve the existing logging style, typically module-level loggers named like
  `rapthor:image` or `rapthor:calibrate`.
- Avoid broad refactors when making behavioral fixes. Many modules coordinate
  through operation output names, command builders, and test fixtures.
- New production helper logic should live in an execution owner package. Keep
  command-line adapters thin and package-owned where possible.

## Testing Guidance

- Add or update the narrowest relevant tests for the changed behavior.
- For `rapthor/lib/` changes, look first under `tests/lib/`.
- For operation changes, look under `tests/operations/`.
- For Prefect/Dask flow changes, look under `tests/execution/`.
- For execution-owned module adapters and migrated helper logic, look under
  `tests/execution/`.
- For installed console entry points, add focused CLI/adapter tests near the
  owning package or existing top-level CLI tests.
- For import-boundary or retired-script guard changes, update
  `tests/architecture/`.
- For end-to-end behavior, use `tests/integration/`, but expect external
  dependencies, longer runtimes, and possible downloaded test Measurement Sets.
- Tests marked `integration` are excluded from the default non-integration pytest
  command.
- Tests marked `internet` may require network access.

Prefer focused commands during development, then run the smallest broader suite
that gives confidence before handing work back.

## Parsets, Strategies, And Defaults

- Default settings are in `rapthor/settings/defaults.parset`,
  `rapthor/settings/defaults_skalow.parset`, and `rapthor/settings/defaults.json`.
- Example strategies live in `examples/`.
- Strategy behavior is implemented mainly in `rapthor/lib/strategy.py` and
  consumed by operation classes.
- Test parset templates live in `tests/resources/`.
- `calibration_strategy` is the production interface for calibration solve type
  and solve order. Do not reintroduce legacy solve toggles such as
  `do_fulljones_solve` or `do_slowgain_solve` into production behavior.
- DD slow-gain solves are currently diagonal, and the default DD strategy should
  remain explicit about any post-slow medium-phase solve rather than relying on
  hidden slot behavior.
- Solutions from one calibration cycle must not be silently reused in later
  cycles unless the strategy and operation contract explicitly say so. Existing
  tests and demo logs rely on cycle-local solution filtering.

When adding a new option, update the defaults, relevant docs/examples, operation
payloads, command builders, and tests together so command-line behavior and flow
payloads stay aligned.

## Scientific Regression Checks

- Use focused unit and execution tests for command contracts, payload contracts,
  output records, operation adapters, and failure handling.
- Use integration tests for representative DD, DI, mixed DI/DD, full-Jones,
  slow-gain, normalization, restart, and imaging scenarios.
- Use `scripts/dev/run_saved_cwl_equivalence.py` as the heavier confidence check
  after scientific logic changes, script-to-module migrations, calibration
  strategy changes, or changes to FITS/h5parm/skymodel products.
- Small WSClean beam-fit differences can occur in image-cube beam sidecars on
  tiny synthetic fixtures. The equivalence helper compares these with a
  documented tolerance while still comparing image data and other products.

## Data And External Resources

- Keep generated data, large Measurement Sets, build artifacts, and downloaded
  archives out of version control.
- Integration fixtures may copy or download test data into temporary directories.
- Keep integration, equivalence, and demo run products out of source decisions.
  They can be very large; prefer `/tmp` or another spacious filesystem for
  local runs.
- Do not assume internet access during normal unit tests.
- Prefer existing small fixtures in `tests/resources/` over adding large files.

## Documentation

Update docs when changing user-facing behavior, parset options, operation
semantics, command-line behavior, or installation/runtime requirements. The main
documentation source is under `docs/source/`, with README-level overview in
`README.md`.

Architecture notes and the current Prefect/Dask orchestration diagram live under
`docs/source/development/`. Keep them aligned with owner-package, task-runner,
and runtime-bootstrap changes.

## Git Hygiene

- Check the worktree before editing.
- Do not overwrite unrelated local changes.
- Keep patches scoped to the requested behavior.
- Do not commit unless explicitly asked.

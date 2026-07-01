# AGENTS.md

Guidance for AI coding agents working in this repository.

## Project Overview

Rapthor is a Python package and command-line pipeline for LOFAR direction-dependent
effect correction, with ongoing SKA-Low support. The production execution path
uses Prefect/Dask flows to run radio astronomy tools such as DP3, WSClean,
EveryBeam, IDG, and related utilities.

Most changes affect one or more of these layers:

- Python domain model and orchestration code under `rapthor/lib/`
- Prefect/Dask execution code under `rapthor/execution/`
- Pipeline operations under `rapthor/operations/`
- Script entry points under `rapthor/scripts/` and package console scripts
- Parset defaults under `rapthor/settings/`
- Unit, script, operation, execution, and integration tests under `tests/`

## Repository Layout

- `rapthor/process.py`: top-level processing flow and operation scheduling.
- `rapthor/lib/`: core objects such as `Field`, `Observation`, `Sector`, `Cluster`,
  `Operation`, strategy handling, and parset handling.
- `rapthor/execution/`: Prefect/Dask flows, command builders, shell execution,
  artifacts, resource checks, and task-runner helpers.
- `rapthor/operations/`: high-level operation classes for calibration, imaging,
  prediction, concatenation, and mosaics.
- `rapthor/scripts/`: standalone helper scripts invoked by execution flows and tests.
- `tests/lib/`, `tests/execution/`, `tests/operations/`, `tests/scripts/`: focused
  test suites for the corresponding code.
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

Run a focused test file:

```bash
python -m pytest tests/operations/test_image.py
```

Run integration tests only when the environment has the required external tools
and data access:

```bash
tox -e test_integration
```

The tox configuration runs `tests/lib/test_field.py` sequentially and the rest of
the non-integration tests in parallel. If you reproduce tox behavior manually,
keep that special case in mind.

## Coding Style

- Follow the existing style in the touched file before introducing new patterns.
- Ruff is configured in `pyproject.toml` with a line length of 100.
- The active lint rules are primarily `E` and `F`; `E501` is ignored.
- Use type hints when they clarify contracts, especially around operation inputs,
  parset-derived values, and helper functions.
- Keep comments useful and sparse. Prefer explaining why a workflow needs a
  condition over restating what the next line does.
- Preserve the existing logging style, typically module-level loggers named like
  `rapthor:image` or `rapthor:calibrate`.
- Avoid broad refactors when making behavioral fixes. Many modules coordinate
  through operation output names, command builders, and test fixtures.

## Testing Guidance

- Add or update the narrowest relevant tests for the changed behavior.
- For `rapthor/lib/` changes, look first under `tests/lib/`.
- For operation changes, look under `tests/operations/`.
- For Prefect/Dask flow changes, look under `tests/execution/`.
- For script changes, look under `tests/scripts/`.
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

When adding a new option, update the defaults, relevant docs/examples, operation
payloads, command builders, and tests together so command-line behavior and flow
payloads stay aligned.

## Data And External Resources

- Keep generated data, large Measurement Sets, build artifacts, and downloaded
  archives out of version control.
- Integration fixtures may copy or download test data into temporary directories.
- Do not assume internet access during normal unit tests.
- Prefer existing small fixtures in `tests/resources/` over adding large files.

## Documentation

Update docs when changing user-facing behavior, parset options, operation
semantics, command-line behavior, or installation/runtime requirements. The main
documentation source is under `docs/source/`, with README-level overview in
`README.md`.

## Git Hygiene

- Check the worktree before editing.
- Do not overwrite unrelated local changes.
- Keep patches scoped to the requested behavior.
- Do not commit unless explicitly asked.
